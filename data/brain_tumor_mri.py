from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


class BrainTumorMRI(object):
    paths = {
        "train": "brain-tumor-mri-dataset/Training",
        "val": "brain-tumor-mri-dataset/Training",
        "test": "brain-tumor-mri-dataset/Testing",
    }

    transformations = transforms.Compose(
        [transforms.Resize((255, 255)), transforms.ToTensor()]
    )

    def prepare_data(self):
        pass

    def __init__(self, dataset_root, batch_size=32, num_workers=4):
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_sample_batch(self):
        data, labels = next(iter(self.test_dataloader))
        return data, labels.numpy()

    @property
    def train_ds(self):
        return ImageFolder(
            root=self.dataset_root + self.paths.get("train"),
            transform=self.transformations,
        )

    @property
    def val_ds(self):
        return ImageFolder(
            root=self.dataset_root + self.paths.get("val"),
            transform=self.transformations,
        )

    @property
    def test_ds(self):
        return ImageFolder(
            root=self.dataset_root + self.paths.get("test"),
            transform=self.transformations,
        )

    @property
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    @property
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @property
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
