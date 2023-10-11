import joblib
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
from models.cnn import ConvNet
from models.timer import Timer
from models.processors import pca_and_normalization
from models.base import BaseNet


class DConvNet(BaseNet):
    def __init__(
        self,
        dataset,
        num_classes,
        ngroups=None,
        lr=1e-4,
        dropout=0.0,
        preprocessing=False,
        pca=256,
        freeze=True,
        *args,
        **kwargs,
    ):
        super().__init__(num_classes=num_classes, dropout=dropout, lr=lr)

        self.model = ConvNet(
            num_classes=num_classes,
            lr=lr,
            dropout=dropout,
            freeze=freeze,
        )
        self.dataset = dataset
        self.preprocessing = preprocessing
        self.pca = pca
        if ngroups is None:
            self.ngroups = self.num_classes
        else:
            self.ngroups = ngroups
        self.features = None

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
            n_clusters=self.ngroups,
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
