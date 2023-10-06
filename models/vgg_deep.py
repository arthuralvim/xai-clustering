import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import silhouette_score
from time import time
from collections import Counter, deque

from data.cluster import ReassignedDataset
from .vgg import VGG
from .timer import Timer
from .processors import pca_and_normalization
import joblib


class DVGG(pl.LightningModule):
    def __init__(
        self,
        dataset,
        num_classes,
        lr=1e-4,
        weight_decay=1e-5,
        dropout=None,
        preprocessing=False,
        pca=256,
        freeze=True,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

        self.model = VGG(
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            freeze=freeze,
        )
        self.dataset = dataset
        self.preprocessing = preprocessing
        self.pca = pca
        self.features = None

        # loop functions
        self.loss_fn = self.define_loss_fn()
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

        #  clustering info
        self.clustering_loss = deque([])
        self.clustering_centers = deque([])
        self.clustering_labels = deque([])
        self.timers = {
            "preprocessing": [],
            "clustering": [],
            "dataset-clustering": [],
            "feature-extraction": [],
        }
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def define_loss_fn(self):
        return nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optim_kwargs = {"lr": self.lr, "weight_decay": self.weight_decay}
        trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.Adam(trainable_parameters, **optim_kwargs)
        return optimizer

    @property
    def epoch_path(self):
        path = f"{self.logger.log_dir}/artifacts/{self.current_epoch}"
        Path(path).mkdir(parents=True, exist_ok=True)
        return path

    def preprocess_features(self, data, pca=256):
        return pca_and_normalization(data, pca)

    def run_kmeans(
        self,
        data,
        n_clusters,
        preprocess=False,
        pca=256,
        random_state=222,
    ):
        if preprocess:
            with Timer(
                name="preprocessing",
                text="Pre-processing elapsed time: {:0.4f} seconds.",
            ) as t1:
                data = self.preprocess_features(data, pca=pca)
            self.timers["preprocessing"].append(t1.timers["preprocessing"])
        clu = KMeans(
            init="k-means++",
            n_clusters=n_clusters,
            n_init=20,
            verbose=0,
        )

        with Timer(
            name="clustering", text="Clustering elapsed time: {:0.4f} seconds."
        ) as t2:
            clu.fit(data)
        self.timers["clustering"].append(t2.timers["clustering"])

        joblib.dump(clu, f"{self.epoch_path}/clustering.dump")

        return clu.inertia_, clu.cluster_centers_, clu.labels_

    def compute_features(self, dataloader):
        top_layer_in_features = self.model.remove_top_layer(self.model.classifier)
        self.model.eval()

        features = []
        targets = []
        for i, (data, target) in enumerate(dataloader):
            with torch.no_grad():
                aux = self.model(data).data.cpu().numpy()
            features.append(aux)
            targets.append(target)

        features = np.concatenate(features)
        targets = np.concatenate(targets)

        with open(f"{self.epoch_path}/features.dump", "wb") as f:
            np.save(f, features)

        with open(f"{self.epoch_path}/features-targets.dump", "wb") as f:
            np.save(f, targets)

        self.model.add_top_layer(self.model.classifier, top_layer_in_features)

        return features, targets

    def train_dataloader(self):
        if self.current_epoch == 0:
            return self.trainer.train_dataloader
        else:
            return self.create_dataset_from_last_clustering()

    def create_dataset_from_last_clustering(self):
        with Timer(
            name="dataset-clustering",
            text="Creating dataset from Clustering elapsed time: {:0.4f} seconds.",
        ) as t3:
            dataloader = self.trainer.train_dataloader
            dataset = dataloader.dataset
            cluster_labels = self.clustering_labels[-1]
            image_indices = dataloader.sampler.indices

            new_dataloader = torch.utils.data.DataLoader(
                ReassignedDataset(image_indices, cluster_labels, dataset),
                batch_size=dataloader.batch_size,
                num_workers=dataloader.num_workers,
            )
        self.timers["dataset-clustering"].append(t3.timers["dataset-clustering"])
        return new_dataloader

    def on_train_epoch_start(self) -> None:
        print(f"Epoch {self.current_epoch}")
        with Timer(
            name="feature-extraction",
            text="Feature extracting elapsed time: {:0.4f} seconds.",
        ) as t4:
            features, features_target = self.compute_features(self.train_dataloader())
        self.timers["feature-extraction"].append(t4.timers["feature-extraction"])

        if self.current_epoch == 0:
            self.original_train_targets = features_target

        cluster_loss, cluster_centers, cluster_labels = self.run_kmeans(
            features,
            n_clusters=self.num_classes,
            preprocess=self.preprocessing,
            pca=self.pca,
        )

        self.clustering_loss.append(cluster_loss)
        self.clustering_centers.append(cluster_centers)
        self.clustering_labels.append(cluster_labels)

        if self.current_epoch != 0:
            labels = [
                self.clustering_labels[self.current_epoch - 1],
                self.clustering_labels[self.current_epoch],
            ]
            metrics = {
                "clu_homogeneity_score": homogeneity_score(*labels),
                "clu_completeness_score": completeness_score(*labels),
                "clu_v_measure_score": v_measure_score(*labels),
                "clu_adjusted_rand_score": adjusted_rand_score(*labels),
                "clu_adjusted_mutual_info_score": adjusted_mutual_info_score(*labels),
            }
            print(f"Metrics: {metrics}")
            self.log_dict(metrics, logger=True)

        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)
        self.train_acc(y_pred, y)

        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True
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
        self.log(
            "test_acc",
            self.test_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)
        y_pred = torch.argmax(y_hat, dim=1).detach().numpy()
        y_truth = y.detach().numpy()
        return y_truth, y_pred
