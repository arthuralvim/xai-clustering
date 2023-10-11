import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from typing import Dict
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy, Precision, Recall


class CNN(pl.LightningModule):
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
        self.learning_rate = lr
        self.weight_decay = weight_decay

        if dropout is None:
            self.dropout = 0.3
        else:
            self.dropout = dropout

        # loop functions
        self.loss_fn = self.define_loss_fn()
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.save_hyperparameters()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 63 * 63, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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
