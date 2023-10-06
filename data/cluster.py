from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler
from .base import BaseDataset
import numpy as np
from torch.utils.data import Dataset, Subset
from PIL import Image


def pil_loader(path):
    """Loads an image.
    Args:
        path (string): path to image file
    Returns:
        Image
    """
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ReassignedDataset(Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        if isinstance(dataset, Subset):
            self.transform = dataset.dataset.transform
        else:
            self.transform = dataset.transform

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):
            path = dataset[idx][0]
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((path, pseudolabel))
        return images

    def __getitem__(self, index):
        return self.imgs[index]

    def __len__(self):
        return len(self.imgs)
