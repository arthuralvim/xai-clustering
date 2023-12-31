from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler
from .base import BaseDataset
import numpy as np


class BrainTumorMRI(BaseDataset):
    dataset_name = "brain-tumor-mri-dataset"

    config = {
        "paths": {
            "train_val": "Training",
            "train": "Training",
            "val": "Training",
            "test": "Testing",
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generate_samplers()

    @property
    def train_val_targets(self):
        if self.reduce_to is not None:
            self.train_val_dataset
            return self.reduce_train_val_targets
        return self.train_val_dataset.targets

    @property
    def test_targets(self):
        if self.reduce_to is not None:
            self.test_dataset
            return self.reduce_test_targets
        return self.test_dataset.targets

    @property
    def classes(self):
        return ["glioma", "meningioma", "notumor", "pituitary"]

    @property
    def labels(self):
        return self.classes

    @property
    def num_classes(self):
        return len(self.classes)

    def generate_samplers(self):
        """Generating samplers for a holdout test."""

        train_idx, val_idx, _, _ = train_test_split(
            np.arange(len(self.train_val_targets)),
            self.train_val_targets,
            test_size=self.val_size,
            stratify=self.train_val_targets,
            random_state=self.random_state,
        )

        self.train_sampler = SubsetRandomSampler(train_idx)
        self.val_sampler = SubsetRandomSampler(val_idx)
        self.train_val_sampler = SubsetRandomSampler(
            np.concatenate([train_idx, val_idx])
        )
        self.test_sampler = None
