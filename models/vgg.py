import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from typing import Dict
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
from models.base import BaseNet


class VGG(BaseNet):
    def __init__(self, num_classes, lr=1e-4, dropout=0, freeze=True, *args, **kwargs):
        super().__init__(num_classes=num_classes, lr=lr, dropout=dropout)

        backbone = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.feature_extractor = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier
        top_layer_in_features = self.remove_top_layer(self.classifier)
        self.add_top_layer(self.classifier, top_layer_in_features)

        if freeze:
            self.freeze_layers(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_layers(self):
        return list(self.feature_extractor) + [self.avgpool] + list(self.classifier)

    def remove_top_layer(self, model):
        in_features = model.pop(-1).in_features
        return in_features

    def add_top_layer(self, model, in_features):
        model.append(nn.Linear(in_features, self.num_classes))

    def freeze_layers(self, model):
        model.eval()

        for name, param in model.named_parameters():
            if param.requires_grad and "classifier.6" not in name:
                param.requires_grad = False
