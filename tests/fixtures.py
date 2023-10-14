import pytest
import numpy as np
import pandas as pd
from data.brain_tumor_mri import BrainTumorMRI
from helpers.prediction import get_labels_from_prediction
from interpret.lrp import LRP

import torchvision.models as models
from torchvision import transforms

from models.cnn import ConvNet
from models.cnn_deep import DConvNet
from models.vgg import VGG
from models.vgg_deep import DVGG


@pytest.fixture
def brain_tumor_mri_dataset():
    return BrainTumorMRI(
        val_size=0.3,
        batch_size=64,
        random_state=37,
        transformations=transforms.Compose(
            [
                transforms.Resize((255, 255)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )


@pytest.fixture
def test_sample(brain_tumor_mri_dataset):
    dataset = brain_tumor_mri_dataset.test_dataset
    dataloader = brain_tumor_mri_dataset.test_dataloader
    data, targets = get_labels_from_prediction(dataloader)
    samples = (
        pd.DataFrame(dataset.imgs, columns=["img", "target"])
        .groupby("target")
        .sample(1, random_state=6)
        .reset_index()
    )
    return (data, targets, samples)


@pytest.fixture
def lrp():
    return LRP()


@pytest.fixture
def vgg_():
    return models.vgg16(weights=models.VGG16_Weights.DEFAULT)


@pytest.fixture
def vgg():
    return VGG(num_classes=4)


@pytest.fixture
def convnet():
    return ConvNet(num_classes=4)


@pytest.fixture
def deep_convnet(brain_tumor_mri_dataset):
    return DConvNet(dataset=brain_tumor_mri_dataset, num_classes=4, freeze=False)


@pytest.fixture
def deep_vgg(brain_tumor_mri_dataset):
    return DVGG(dataset=brain_tumor_mri_dataset, num_classes=4, freeze=False)
