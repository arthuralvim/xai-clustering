import pytest
import torch
from interpret.lrp2 import InitLRP


class TestLRP2:
    def test_lrp_should_return_relevance_for_convnet(self, lrp, convnet, test_sample):
        data, _, _ = test_sample
        image_id = 5
        lrp = InitLRP(verbose=True)
        model, rules = lrp.set_rules(convnet)
        import ipdb

        ipdb.set_trace()

        attr_algo = LRP(vgg)
        attributions = attr_algo.attribute(sample_input.unsqueeze(0), target=0)

        # visualize image attributions
        original_im_mat = np.transpose(sample_input.cpu().detach().numpy(), (1, 2, 0))
        attributions_img = np.transpose(
            attributions.squeeze(0).cpu().detach().numpy(), (1, 2, 0)
        )

        fig, axes = visualization.visualize_image_attr_multiple(
            attr=attributions_img,
            original_image=original_im_mat,
            methods=["original_image", "heat_map"],
            signs=["all", "absolute"],
            titles=["Original Image", "Attribution Magnitude"],
            # cmap=default_cmap,
            cmap=plt.cm.RdBu,
            show_colorbar=True,
        )
        # original_image = torch.from_numpy(data[image_id])
        # image_with_relevance = lrp.apply(convnet, original_image)

    # def test_lrp_should_return_relevance_for_vgg(self, lrp, vgg, test_sample):
    #     data, _, _ = test_sample
    #     image_id = 5
    #     original_image = torch.from_numpy(data[image_id])
    #     image_with_relevance = lrp.apply(vgg, original_image)

    # def test_lrp_should_return_relevance_for_especific_target(
    #     self, lrp, convnet, test_sample
    # ):
    #     data, _, _ = test_sample
    #     image_id = 5
    #     original_image = torch.from_numpy(data[image_id])
    #     image_with_relevance = lrp.apply(convnet, original_image, target=2)
