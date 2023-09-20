import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from typing import Dict
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy


class VGG(pl.LightningModule):
    __default_config__ = {
        "input_size": 64,
        "loss_function": "crossentropy",
        "dropout": 0.5,
        "optimizers": {"lr": 1e-4, "weight_decay": 1e-5},
    }

    num_classes = 4
    labels = {}

    def __init__(
        self, config: Dict = None, only_feature_extractor=False, *args, **kwargs
    ):
        super().__init__()

        self.config = self.__default_config__ if config is None else config
        self.only_feature_extractor = only_feature_extractor

        backbone = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.feature_extractor = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier
        in_features = self.classifier.pop(-1).in_features
        if not only_feature_extractor:
            self.classifier.append(nn.Linear(in_features, self.num_classes))
        self.freeze_layers(self)

        # loop functions
        self.loss_fn = self.define_loss_fn()
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def freeze_layers(self, model):
        # layers are frozen by using eval()
        model.eval()
        # freeze params
        for name, param in model.named_parameters():
            if param.requires_grad and "classifier.6" not in name:
                param.requires_grad = False

    def define_loss_fn(self):
        if self.config["loss_function"] == "crossentropy":
            return nn.CrossEntropyLoss()
        else:
            raise ValueError("Unknown criterion")

    def torch_loop(self, batch, batch_idx, metric_name, metric_call):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log(metric_name, loss, prog_bar=True)
        metric_call(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        return self.torch_loop(
            batch, batch_idx, metric_name="train_loss", metric_call=self.train_acc
        )

    def validation_step(self, batch, batch_idx):
        return self.torch_loop(
            batch, batch_idx, metric_name="val_loss", metric_call=self.val_acc
        )

    def test_step(self, batch, batch_idx):
        return self.torch_loop(
            batch, batch_idx, metric_name="test_loss", metric_call=self.test_acc
        )

    def configure_optimizers(self):
        if "optimizers" in self.config:
            optim_kwargs = self.config.get("optimizers")
        else:
            optim_kwargs = {}
        trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.Adam(trainable_parameters, **optim_kwargs)
        return optimizer
