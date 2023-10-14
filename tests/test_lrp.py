import pytest
import torch


class TestExtractLayers:
    def test_lrp_should_extract_layers_from_vgg(self, lrp, vgg):
        layers, linear_layer_indices = lrp.extract_layers(vgg)
        assert len(layers) == 39
        assert linear_layer_indices == [32, 35, 38]

    def test_lrp_should_extract_layers_from_dvgg(self, lrp, deep_vgg):
        layers, linear_layer_indices = lrp.extract_layers(deep_vgg)
        assert len(layers) == 39
        assert linear_layer_indices == [32, 35, 38]

    def test_lrp_should_extract_layers_from_convnet(self, lrp, convnet):
        layers, linear_layer_indices = lrp.extract_layers(
            convnet, convert_callable=lrp.convert_convnet
        )
        assert len(layers) == 11
        assert linear_layer_indices == [8, 10]

    def test_lrp_should_extract_layers_from_dconvnet(self, lrp, deep_convnet):
        layers, linear_layer_indices = lrp.extract_layers(
            deep_convnet, convert_callable=lrp.convert_convnet
        )
        assert len(layers) == 11
        assert linear_layer_indices == [8, 10]


class TestLRP:
    def test_lrp_should_return_relevance_for_vgg(self, lrp, vgg, test_sample):
        data, _, _ = test_sample
        image_id = 5
        original_image = torch.from_numpy(data[image_id])
        image_with_relevance = lrp.apply(vgg, original_image)

    def test_lrp_should_return_relevance_for_convnet(self, lrp, convnet, test_sample):
        data, _, _ = test_sample
        image_id = 5
        original_image = torch.from_numpy(data[image_id])
        image_with_relevance = lrp.apply(convnet, original_image)

    def test_lrp_should_return_relevance_for_especific_target(
        self, lrp, convnet, test_sample
    ):
        data, _, _ = test_sample
        image_id = 5
        original_image = torch.from_numpy(data[image_id])
        image_with_relevance = lrp.apply(convnet, original_image, target=2)
