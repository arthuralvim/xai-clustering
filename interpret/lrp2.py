import torch
from captum.attr._utils.lrp_rules import (
    EpsilonRule,
    GammaRule,
)


class InitLRP(object):
    def __init__(self, verbose=False, epsilon=100):
        self.verbose = verbose
        self.epsilon = epsilon
        self.rules = [
            {"depth": [0, 16], "rule": GammaRule(), "name": "LRP-y"},
            {"depth": [17, 30], "rule": EpsilonRule(epsilon=1e-9), "name": "LRP-e"},
            {"depth": [30, 100], "rule": EpsilonRule(epsilon=0), "name": "LRP-0"},
        ]

    def _get_model_layers(self, model: torch.nn.Module):
        # get children form model!
        children = list(model.children())
        flatt_children = []
        if children == []:
            # if model has no children; model is last child! :O
            return model
        else:
            # look for children from children... to the last child!
            for child in children:
                try:
                    flatt_children.extend(self._get_model_layers(child))
                except TypeError:
                    flatt_children.append(self._get_model_layers(child))
        return flatt_children

    def reset_rules(self, model):
        """
        run this code to re-set all the layers to default epsilon-rule
        """
        layers = self._get_model_layers(model)
        number_layers = len(layers)
        for l in range(0, number_layers)[::-1]:
            delattr(layers[l], "rule")
        return model

    def print_rules(self, model):
        layers = self._get_model_layers(model)
        number_layers = len(layers)
        for l in range(0, number_layers)[::-1]:
            try:
                rule = getattr(layers[l], "rule")
            except:
                rule = "default"

            print(
                "layer {:}| rule: {:} \n    structure: {:}\n----------------------------------------------------------------------------".format(
                    l, rule, layers[l]
                )
            )

    def set_rules(self, model):
        layers = self._get_model_layers(model)

        rules = []
        rules.append(["layer", "depth", "rule"])
        depth = 0
        for layer in layers:
            rule = "default"

            if (
                isinstance(layer, torch.nn.Linear)
                or isinstance(layer, torch.nn.Conv1d)
                or isinstance(layer, torch.nn.Conv2d)
                or isinstance(layer, torch.nn.AvgPool2d)
                or isinstance(layer, torch.nn.MaxPool2d)
            ):
                if depth < 16:
                    setattr(layer, "rule", GammaRule())
                    rule = "LRP-y"

                elif 17 <= depth <= 30:
                    setattr(layer, "rule", EpsilonRule(epsilon=1e-9))
                    rule = "LRP-e"

                elif depth > 30:
                    setattr(layer, "rule", EpsilonRule(epsilon=0))
                    rule = "LRP-0"

                depth += 1

            else:
                setattr(layer, "rule", EpsilonRule())
                rule = "none"

            rules.append([layer, depth, rule])

        if self.verbose:
            self.print_rules(model)
        return model, rules
