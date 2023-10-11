import pandas as pd
import numpy as np


def metrics_to_dataframe(path, dataset):
    metrics = pd.read_csv(path)
    metrics["epoch"] = metrics["epoch"].fillna(method="ffill")

    n_btrain = len(dataset.train_dataloader)
    n_bval = len(dataset.val_dataloader)

    max_epoch = max(metrics["epoch"]) + 1

    train_metrics = []
    valid_metrics = []
    epoch_train_metrics = []
    epoch_valid_metrics = []

    for n, batch in enumerate(np.split(metrics, max_epoch)):
        train_metrics.append(
            batch[["epoch", "step", "train_loss_step", "train_acc_step"]][:n_btrain]
        )
        valid_metrics.append(
            batch[["epoch", "step", "val_loss_step", "val_acc_step"]][
                n_btrain : n_btrain + n_bval
            ]
        )
        epoch_valid_metrics.append(
            batch[["epoch", "val_loss_epoch", "val_acc_epoch"]][
                n_btrain + n_bval : n_btrain + n_bval + 1
            ]
        )
        epoch_train_metrics.append(
            batch[["epoch", "train_loss_epoch", "train_acc_epoch"]][
                n_btrain + n_bval + 1 : n_btrain + n_bval + 2
            ]
        )

    train_metrics = pd.concat(train_metrics, axis=0)
    valid_metrics = pd.concat(valid_metrics, axis=0)
    epoch_train_metrics = pd.concat(epoch_train_metrics, axis=0)
    epoch_valid_metrics = pd.concat(epoch_valid_metrics, axis=0)
    epoch_metrics = pd.merge(epoch_train_metrics, epoch_valid_metrics, on="epoch")
    epoch_metrics["train_acc_epoch"] = epoch_metrics["train_acc_epoch"] * 100
    epoch_metrics["val_acc_epoch"] = epoch_metrics["val_acc_epoch"] * 100
    return train_metrics, valid_metrics, epoch_metrics


def dmetrics_to_dataframe(path, dataset):
    metrics = pd.read_csv(path)
    metrics["epoch"] = metrics["epoch"].fillna(method="ffill")

    n_btrain = len(dataset.train_dataloader)
    n_bval = len(dataset.val_dataloader)

    max_epoch = max(metrics["epoch"]) + 1

    train_metrics = []
    valid_metrics = []
    epoch_train_metrics = []
    epoch_valid_metrics = []
    clustering_metrics = []

    for n, batch in enumerate(np.split(metrics, max_epoch)):
        train_metrics.append(
            batch[["epoch", "step", "train_loss_step", "train_acc_step"]][:n_btrain]
        )
        valid_metrics.append(
            batch[["epoch", "step", "val_loss_step", "val_acc_step"]][
                n_btrain : n_btrain + n_bval
            ]
        )
        epoch_valid_metrics.append(
            batch[["epoch", "val_loss_epoch", "val_acc_epoch"]][
                n_btrain + n_bval : n_btrain + n_bval + 1
            ]
        )
        epoch_train_metrics.append(
            batch[["epoch", "train_loss_epoch", "train_acc_epoch"]][
                n_btrain + n_bval + 1 : n_btrain + n_bval + 2
            ]
        )
        if n > 0:
            clustering_metrics.append(
                batch[
                    [
                        "clu_homogeneity_score",
                        "clu_completeness_score",
                        "clu_v_measure_score",
                        "clu_adjusted_rand_score",
                        "clu_adjusted_mutual_info_score",
                    ]
                ][n_btrain + n_bval + 1 : n_btrain + n_bval + 2]
            )

    train_metrics = pd.concat(train_metrics, axis=0)
    valid_metrics = pd.concat(valid_metrics, axis=0)
    epoch_train_metrics = pd.concat(epoch_train_metrics, axis=0)
    epoch_valid_metrics = pd.concat(epoch_valid_metrics, axis=0)
    clustering_metrics = pd.concat(clustering_metrics, axis=0)
    epoch_metrics = pd.merge(epoch_train_metrics, epoch_valid_metrics, on="epoch")
    epoch_metrics["train_acc_epoch"] = epoch_metrics["train_acc_epoch"] * 100
    epoch_metrics["val_acc_epoch"] = epoch_metrics["val_acc_epoch"] * 100
    return train_metrics, valid_metrics, epoch_metrics, clustering_metrics
