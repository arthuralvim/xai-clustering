from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path
import numpy as np


class BaseDataset(object):
    dataset_name = "dataset-name"

    config = {
        "paths": {
            "train": "training-set",
            "val": "validation-set",
            "train_val": "training-set",
            "test": "testing-set",
        }
    }

    def __init__(
        self,
        dataset_root,
        random_state=333,
        batch_size=32,
        num_workers=4,
        train_size=None,
        val_size=None,
        test_size=None,
    ):
        self.dataset_root = (
            self.config.get("dataset_root") if not dataset_root else dataset_root
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_root = dataset_root
        self.random_state = random_state
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.test_sampler = None
        self.train_sampler = None
        self.val_sampler = None

    @property
    def classes(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def data_path(self):
        return Path(self.dataset_root).joinpath(self.dataset_name)

    def prepare_data(self):
        """Download images and prepare images datasets."""
        NotImplemented

    @property
    def targets(self):
        raise NotImplementedError

    def generate_samplers(self):
        """Generating samplers for a holdout test."""
        non_test_idx, test_idx, non_test_targets, test_targets = train_test_split(
            np.arange(len(self.targets)),
            self.targets,
            test_size=self.test_size,
            stratify=self.targets,
            random_state=self.random_state,
        )

        train_idx, val_idx, train_targets, val_targets = train_test_split(
            non_test_idx,
            non_test_targets,
            test_size=self.val_size,
            stratify=non_test_targets,
            random_state=self.random_state,
        )

        self.test_sampler = SubsetRandomSampler(test_idx)
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.val_sampler = SubsetRandomSampler(val_idx)

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((255, 255)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=mean, std=std)
            ]
        )

    @property
    def train_val_transform(self):
        return self.train_transform

    @property
    def val_transform(self):
        return self.train_transform

    @property
    def test_transform(self):
        return self.train_transform

    def get_sample_batch(self):
        data, labels = next(iter(self.test_dataloader))
        return data, labels.numpy()

    @property
    def train_dataset(self):
        return ImageFolder(
            root=self.data_path.joinpath(self.config.get("paths").get("train")),
            transform=self.train_transform,
        )

    @property
    def val_dataset(self):
        return ImageFolder(
            root=self.data_path.joinpath(self.config.get("paths").get("val")),
            transform=self.val_transform,
        )

    @property
    def train_val_dataset(self):
        return ImageFolder(
            root=self.data_path.joinpath(self.config.get("paths").get("train_val")),
            transform=self.train_val_transform,
        )

    @property
    def test_dataset(self):
        return ImageFolder(
            root=self.data_path.joinpath(self.config.get("paths").get("test")),
            transform=self.test_transform,
        )

    def generate_dataloader(self, dataset, sampler):
        extra = {}
        if not sampler is None:
            extra = {"sampler": sampler}
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers, **extra
        )

    @property
    def train_dataloader(self):
        return self.generate_dataloader(
            dataset=self.train_dataset, sampler=self.train_sampler
        )

    @property
    def val_dataloader(self):
        return self.generate_dataloader(
            dataset=self.val_dataset, sampler=self.val_sampler
        )

    @property
    def test_dataloader(self):
        return self.generate_dataloader(
            dataset=self.test_dataset, sampler=self.test_sampler
        )