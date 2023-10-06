import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from typing import Dict
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy, Precision, Recall


class VGG(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        lr=1e-4,
        weight_decay=1e-5,
        dropout=None,
        freeze=True,
        *args,
        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

        backbone = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.feature_extractor = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier
        top_layer_in_features = self.remove_top_layer(self.classifier)
        self.add_top_layer(self.classifier, top_layer_in_features)

        if freeze:
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

    def remove_top_layer(self, model):
        in_features = model.pop(-1).in_features
        return in_features

    def add_top_layer(self, model, in_features):
        model.append(nn.Linear(in_features, self.num_classes))

    def freeze_layers(self, model):
        # layers are frozen by using eval()
        model.eval()
        # freeze params
        for name, param in model.named_parameters():
            if param.requires_grad and "classifier.6" not in name:
                param.requires_grad = False

    def define_loss_fn(self):
        return nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optim_kwargs = {"lr": self.lr, "weight_decay": self.weight_decay}
        trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.Adam(trainable_parameters, **optim_kwargs)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)
        self.train_acc(y_pred, y)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc",
            self.train_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)
        self.val_acc(y_pred, y)

        self.log(
            "val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )
        self.log(
            "val_acc",
            self.val_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)
        self.test_acc(y_pred, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)
        y_pred = torch.argmax(y_hat, dim=1).detach().numpy()
        y_truth = y.detach().numpy()
        return y_truth, y_pred
