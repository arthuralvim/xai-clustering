import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from typing import Dict
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy, Precision, Recall


class BaseNet(pl.LightningModule):
    def __init__(self, num_classes, dropout=0.0, lr=1e-4, *args, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.learning_rate = lr
        self.dropout = dropout
        self.loss_fn = self.define_loss_fn()

        # metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.save_hyperparameters()

    @property
    def name(self):
        return self.__class__.__name__.lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def define_loss_fn(self):
        return nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optim_kwargs = {"lr": self.learning_rate}
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
