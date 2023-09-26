import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from typing import Dict
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy, Precision, Recall


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

    def training_step(self, batch, batch_idx):
        # print(f"runnning training batch: {batch_idx}")
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)
        self.train_acc(y_pred, y)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # print(f"runnning validation batch: {batch_idx}")
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)
        self.val_acc(y_pred, y)

        self.log(
            "val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True, logger=True)
        return loss

    # def on_validation_end(self) -> None:
    #     if self.trainer.sanity_checking:
    #         return super().on_validation_end()
    #     train_loss = self.trainer.callback_metrics["train_loss"]
    #     val_loss = self.trainer.callback_metrics["val_loss"]
    #     train_acc = self.trainer.callback_metrics["train_acc"]
    #     val_acc = self.trainer.callback_metrics["val_acc"]
    #     epoch = self.trainer.current_epoch
    #     print(
    #         f"Epoch {epoch} train_loss={train_loss:.4f}/train_acc={train_acc:.4f} val_loss={val_loss:.4f}/val_acc={val_acc:.4f}"
    #     )
    #     return super().on_validation_end()

    def test_step(self, batch, batch_idx):
        # print(f"runnning test batch: {batch_idx}")
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)
        self.test_acc(y_pred, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # print(f"runnning prediction batch: {batch_idx}")
        x, y = batch
        y_hat = self(x)
        y_pred = torch.argmax(y_hat, dim=1).detach().numpy()
        y_truth = y.detach().numpy()
        return y_truth, y_pred

    def configure_optimizers(self):
        if "optimizers" in self.config:
            optim_kwargs = self.config.get("optimizers")
        else:
            optim_kwargs = {}
        trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.Adam(trainable_parameters, **optim_kwargs)
        return optimizer
