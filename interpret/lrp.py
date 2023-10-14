import copy
import torch
import torch.nn as nn
import numpy as np


class LRP(object):
    """
    The inspiration of this code comes from:
        - https://github.com/deepfindr/xai-series/
        - https://git.tu-berlin.de/gmontavon/lrp-tutorial
    """

    def __init__(self, device=None, *args, **kwargs):
        super(LRP, self).__init__(*args, **kwargs)
        CUDA_AVAILABLE = torch.cuda.is_available()
        if device is None:
            self.device = torch.device("cpu:0")
        else:
            self.device = device

    def layers_generator(self, model):
        """Iterate over models layers"""
        if hasattr(model, "get_layers"):
            layers = model.get_layers()
            for layer in layers:
                yield layer
        else:
            for _, layers in model._modules.items():
                if isinstance(layers, nn.Sequential):
                    for layer in layers:
                        yield layer
                else:
                    yield layers

    def new_layer(self, layer, g):
        """Clone a layer and pass its parameters through the function g."""
        layer = copy.deepcopy(layer)
        try:
            layer.weight = torch.nn.Parameter(g(layer.weight))
        except AttributeError:
            pass
        try:
            layer.bias = torch.nn.Parameter(g(layer.bias))
        except AttributeError:
            pass
        return layer

    def convert_vgg(self, layer, depth):
        """Converts a dense layer to a conv layer for a VGG16"""
        newlayer = None
        if depth == 0:
            # this part is due the transition between avgpool and the dense layer
            m, n = 512, layer.weight.shape[0]
            newlayer = nn.Conv2d(m, n, 7)
            newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 7, 7))
        else:
            m, n = layer.weight.shape[1], layer.weight.shape[0]
            newlayer = nn.Conv2d(m, n, 1)
            newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 1, 1))
        newlayer.bias = nn.Parameter(layer.bias)
        return newlayer

    def convert_convnet(self, layer, depth):
        """Converts a dense layer to a conv layer for a VGG16"""
        newlayer = None
        if depth == 0:
            # this part is due the transition between avgpool and the dense layer
            m, n = 32, layer.weight.shape[0]
            newlayer = nn.Conv2d(m, n, 3)
            newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 63, 63))
        else:
            m, n = layer.weight.shape[1], layer.weight.shape[0]
            newlayer = nn.Conv2d(m, n, 1)
            newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 1, 1))
            newlayer.bias = nn.Parameter(layer.bias)
        return newlayer

    def extract_layers(self, model, convert_callable=None):
        """
        This method extract all layers from a model, then it convert any linear layer into a convolutional form
        required for the rest of the calculations.
        """

        if convert_callable is None:
            convert_callable = self.convert_vgg

        layers = []
        linear_layers = []
        depth_linear = 0

        for layer_depth, layer in enumerate(self.layers_generator(model)):
            if isinstance(layer, nn.Linear):
                layer = convert_callable(layer, depth_linear)
                linear_layers.append(layer_depth)
                depth_linear += 1

            layers.append(layer)

        return layers, linear_layers

    def propagating_image_forward_vgg(self, layers, linear_layer_indices, image):
        # >>> Step 2: Propagate image through layers and store activations
        activations = [image] + [None] * len(layers)  # list of activations

        for n, layer in enumerate(layers):
            if n in linear_layer_indices:
                # this part is due the transition between avgpool and the dense layer
                if n == 32:
                    activations[n] = activations[n].reshape((1, 512, 7, 7))

            activation = layer.forward(activations[n])

            if isinstance(layer, torch.nn.modules.pooling.AdaptiveAvgPool2d):
                activation = torch.flatten(activation, start_dim=1)

            activations[n + 1] = activation
        return activations

    def propagating_image_forward_convnet(self, layers, linear_layer_indices, image):
        # >>> Step 2: Propagate image through layers and store activations
        activations = [image] + [None] * len(layers)  # list of activations

        for n, layer in enumerate(layers):
            activation = layer.forward(activations[n])
            activations[n + 1] = activation
        return activations

    def last_layer_one_hot(self, activations, target):
        # >>> Step 3: Replace last layer with one-hot-encoding

        output_activation = activations[-1]
        mask = torch.zeros(output_activation.shape)

        if target is None:
            max_idx = torch.argmax(output_activation)
            mask[0][max_idx][0][0] = 1
            output = torch.mul(mask, output_activation)
        else:
            mask[0][target][0][0] = 1
            output = torch.mul(mask, output_activation)

        activations[-1] = torch.FloatTensor(output).to(self.device)
        return activations

    def define_rho_inc_by_depth(self, depth):
        # Apply variants of LRP depending on the depth
        # see: https://link.springer.com/chapter/10.1007%2F978-3-030-28954-6_10
        # Lower layers, LRP-gamma >> Favor positive contributions (activations)

        if depth <= 16:
            rho = lambda p: p + 0.25 * p.clamp(min=0)
            incr = lambda z: z + 1e-9
        # Middle layers, LRP-epsilon >> Remove some noise / Only most salient factors survive
        if 17 <= depth <= 30:
            rho = lambda p: p
            incr = lambda z: z + 1e-9 + 0.25 * ((z**2).mean() ** 0.5).data
        # Upper Layers, LRP-0 >> Basic rule
        if depth >= 31:
            rho = lambda p: p
            incr = lambda z: z + 1e-9
        return rho, incr

    def propagating_relevance_backward(self, layers, activations):
        # >>> Step 4: Backpropagate relevance scores
        relevances = [None] * len(layers) + [activations[-1]]
        # Iterate over the layers in reverse order
        for n, layer in reversed(list(enumerate(layers))):
            # Treat max pooling layers as avg pooling
            if isinstance(layer, torch.nn.MaxPool2d):
                layer = torch.nn.AvgPool2d(2)

            if (
                isinstance(layer, torch.nn.Conv2d)
                or isinstance(layer, torch.nn.AvgPool2d)
                or isinstance(layer, torch.nn.Linear)
            ):
                activations[n] = activations[n].data.requires_grad_(True)
                rho, incr = self.define_rho_inc_by_depth(n)

                # Transform weights of layer and execute forward pass
                z = incr(self.new_layer(layers[n], rho).forward(activations[n]))
                # Element-wise division between relevance of the next layer and z
                s = (relevances[n + 1] / z).data
                # Calculate the gradient and multiply it by the activation
                (z * s).sum().backward()
                c = activations[n].grad
                # Assign new relevance values
                relevances[n] = (activations[n] * c).data
            else:
                relevances[n] = relevances[n + 1]
        return relevances

    def apply_relevances_to_image(self, relevances):
        # >>> Potential Step 5: Apply different propagation rule for pixels
        image_relevance = relevances[0].permute(0, 2, 3, 1).detach().cpu().numpy()[0]
        image_relevance = np.interp(
            image_relevance, (image_relevance.min(), image_relevance.max()), (0, 1)
        )
        return image_relevance

    def apply(self, model, image, target=None):
        image = torch.unsqueeze(image, 0)

        if "vgg" in model.name:
            layers, linear_layer_indices = self.extract_layers(model)
            activations = self.propagating_image_forward_vgg(
                layers, linear_layer_indices, image
            )
        elif "convnet" in model.name:
            layers, linear_layer_indices = self.extract_layers(
                model, convert_callable=self.convert_convnet
            )
            activations = self.propagating_image_forward_convnet(
                layers, linear_layer_indices, image
            )
        else:
            raise Exception("Unknown model")

        activations = self.last_layer_one_hot(activations, target)
        relevances = self.propagating_relevance_backward(layers, activations)
        image_relevance = self.apply_relevances_to_image(relevances)
        return image_relevance
