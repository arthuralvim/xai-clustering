import copy
import torch
import torch.nn as nn
import numpy as np


class LayerWiseRP(object):
    """
    This code was obtained and modified from:
    https://github.com/deepfindr/xai-series/
    """

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

    def dense_to_conv(self, layers):
        """Converts a dense layer to a conv layer"""
        newlayers = []
        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                newlayer = None
                if i == 0:
                    m, n = 512, layer.weight.shape[0]
                    newlayer = nn.Conv2d(m, n, 7)
                    newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 7, 7))
                else:
                    m, n = layer.weight.shape[1], layer.weight.shape[0]
                    newlayer = nn.Conv2d(m, n, 1)
                    newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 1, 1))
                newlayer.bias = nn.Parameter(layer.bias)
                newlayers += [newlayer]
            else:
                newlayers += [layer]
        return newlayers

    def get_linear_layer_indices(self, model):
        offset = len(model._modules["feature_extractor"]) + 1
        indices = []
        for i, layer in enumerate(model._modules["classifier"]):
            if isinstance(layer, nn.Linear):
                indices.append(i)
        indices = [offset + val for val in indices]
        return indices

    def apply(self, model, image, device):
        try:
            model._modules["feature_extractor"]
        except:
            model = model.model

        image = torch.unsqueeze(image, 0)

        # >>> Step 1: Extract layers
        layers = (
            list(model._modules["feature_extractor"])
            + [model._modules["avgpool"]]
            + self.dense_to_conv(list(model._modules["classifier"]))
        )
        linear_layer_indices = self.get_linear_layer_indices(model)

        # >>> Step 2: Propagate image through layers and store activations
        n_layers = len(layers)
        activations = [image] + [None] * n_layers  # list of activations

        for layer in range(n_layers):
            if layer in linear_layer_indices:
                if layer == 32:
                    activations[layer] = activations[layer].reshape((1, 512, 7, 7))
            activation = layers[layer].forward(activations[layer])
            if isinstance(layers[layer], torch.nn.modules.pooling.AdaptiveAvgPool2d):
                activation = torch.flatten(activation, start_dim=1)
            activations[layer + 1] = activation

        # >>> Step 3: Replace last layer with one-hot-encoding
        output_activation = activations[-1].detach().cpu().numpy()
        max_activation = output_activation.max()
        one_hot_output = np.array(
            [
                max_activation if val == max_activation else 0
                for val in output_activation[0]
            ]
        )
        # one_hot_output = [
        #     max_activation if val == max_activation else 0 for val in output_activation[0]
        # ]

        activations[-1] = torch.FloatTensor(one_hot_output).to(device)
        # activations[-1] = torch.FloatTensor([one_hot_output]).to(device)

        # >>> Step 4: Backpropagate relevance scores
        relevances = [None] * n_layers + [activations[-1]]
        # Iterate over the layers in reverse order
        for layer in range(0, n_layers)[::-1]:
            current = layers[layer]
            # Treat max pooling layers as avg pooling
            if isinstance(current, torch.nn.MaxPool2d):
                layers[layer] = torch.nn.AvgPool2d(2)
                current = layers[layer]
            if (
                isinstance(current, torch.nn.Conv2d)
                or isinstance(current, torch.nn.AvgPool2d)
                or isinstance(current, torch.nn.Linear)
            ):
                activations[layer] = activations[layer].data.requires_grad_(True)

                # Apply variants of LRP depending on the depth
                # see: https://link.springer.com/chapter/10.1007%2F978-3-030-28954-6_10
                # Lower layers, LRP-gamma >> Favor positive contributions (activations)
                if layer <= 16:
                    rho = lambda p: p + 0.25 * p.clamp(min=0)
                    incr = lambda z: z + 1e-9
                # Middle layers, LRP-epsilon >> Remove some noise / Only most salient factors survive
                if 17 <= layer <= 30:
                    rho = lambda p: p
                    incr = lambda z: z + 1e-9 + 0.25 * ((z**2).mean() ** 0.5).data
                # Upper Layers, LRP-0 >> Basic rule
                if layer >= 31:
                    rho = lambda p: p
                    incr = lambda z: z + 1e-9

                # Transform weights of layer and execute forward pass
                z = incr(self.new_layer(layers[layer], rho).forward(activations[layer]))
                # Element-wise division between relevance of the next layer and z
                s = (relevances[layer + 1] / z).data
                # Calculate the gradient and multiply it by the activation
                (z * s).sum().backward()
                c = activations[layer].grad
                # Assign new relevance values
                relevances[layer] = (activations[layer] * c).data
            else:
                relevances[layer] = relevances[layer + 1]

        # >>> Potential Step 5: Apply different propagation rule for pixels

        image_relevances = relevances[0].permute(0, 2, 3, 1).detach().cpu().numpy()[0]
        image_relevances = np.interp(
            image_relevances, (image_relevances.min(), image_relevances.max()), (0, 1)
        )
        return image_relevances
