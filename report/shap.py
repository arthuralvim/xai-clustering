import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Optional
from shap.plots import colors
from shap._explanation import Explanation
from shap.utils._legacy import kmeans
from matplotlib.colors import Colormap


def shap_plot(
    shap_values: Explanation or np.ndarray,
    pixel_values: Optional[np.ndarray] = None,
    labels: Optional[list or np.ndarray] = None,
    true_labels: Optional[list] = None,
    labelpad: Optional[float] = None,
    cmap: Optional[str or Colormap] = colors.red_transparent_blue,
    show: Optional[bool] = True,
    plot_shape=(3, 2),
    figsize=(9, 9),
    save_as: Optional[str] = None,
):
    """Plots SHAP values for image inputs.

    Copied and modified from:
    https://github.com/shap/shap/blob/master/shap/plots/_image.py

    Parameters
    ----------
    shap_values : [numpy.array]
        List of arrays of SHAP values. Each array has the shape
        (# samples x width x height x channels), and the
        length of the list is equal to the number of model outputs that are being
        explained.

    pixel_values : numpy.array
        Matrix of pixel values (# samples x width x height x channels) for each image.
        It should be the same
        shape as each array in the ``shap_values`` list of arrays.

    labels : list or np.ndarray
        List or ``np.ndarray`` (# samples x top_k classes) of names for each of the
        model outputs that are being explained.

    true_labels: list
        List of a true image labels to plot.

    width : float
        The width of the produced matplotlib plot.

    labelpad : float
        How much padding to use around the model output labels.

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot
        to be customized further after it has been created.

    Examples
    --------

    See `image plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/image.html>`_.

    """
    # support passing an explanation object
    if str(type(shap_values)).endswith("Explanation'>"):
        shap_exp = shap_values
        # feature_names = [shap_exp.feature_names]
        # ind = 0
        if len(shap_exp.output_dims) == 1:
            shap_values = [
                shap_exp.values[..., i] for i in range(shap_exp.values.shape[-1])
            ]
        elif len(shap_exp.output_dims) == 0:
            shap_values = shap_exp.values
        else:
            raise Exception(
                "Number of outputs needs to have support added!! (probably a simple fix)"
            )
        if pixel_values is None:
            pixel_values = shap_exp.data
        if labels is None:
            labels = shap_exp.output_names

    # multi_output = True
    if not isinstance(shap_values, list):
        # multi_output = False
        shap_values = [shap_values]

    if len(shap_values[0].shape) == 3:
        shap_values = [v.reshape(1, *v.shape) for v in shap_values]
        pixel_values = pixel_values.reshape(1, *pixel_values.shape)

    # labels: (rows (images) x columns (top_k classes) )
    if labels is not None:
        if isinstance(labels, list):
            labels = np.array(labels).reshape(1, -1)

    # if labels is not None:
    #     labels = np.array(labels)
    #     if labels.shape[0] != shap_values[0].shape[0] and labels.shape[0] == len(shap_values):
    #         labels = np.tile(np.array([labels]), shap_values[0].shape[0])
    #     assert labels.shape[0] == shap_values[0].shape[0], "Labels must have same row count as shap_values arrays!"
    #     if multi_output:
    #         assert labels.shape[1] == len(shap_values), "Labels must have a column for each output in shap_values!"
    #     else:
    #         assert len(labels[0].shape) == 1, "Labels must be a vector for single output shap_values."

    label_kwargs = {} if labelpad is None else {"pad": labelpad}

    # plot our explanations
    x = pixel_values
    # fig_size = np.array([3 * (len(shap_values) + 1), 2.5 * (x.shape[0] + 1)])
    # if fig_size[0] > width:
    #     fig_size *= width / fig_size[0]

    if plot_shape is None:
        nrows, ncols = x.shape[0], len(shap_values) + 1
    else:
        nrows, ncols = plot_shape

    fig = plt.figure(figsize=figsize)  # Notice the equal aspect ratio
    axes = [fig.add_subplot(nrows, ncols, i + 1) for i in range(nrows * ncols)]

    for row in range(x.shape[0]):
        x_curr = x[row].copy()

        # make sure we have a 2D array for grayscale
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
            x_curr = x_curr.reshape(x_curr.shape[:2])

        # if x_curr.max() > 1:
        #     x_curr /= 255.

        # get a grayscale version of the image
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
            x_curr_gray = (
                0.2989 * x_curr[:, :, 0]
                + 0.5870 * x_curr[:, :, 1]
                + 0.1140 * x_curr[:, :, 2]
            )  # rgb to gray
            x_curr_disp = x_curr
        elif len(x_curr.shape) == 3:
            x_curr_gray = x_curr.mean(2)

            # for non-RGB multi-channel data we show an RGB image where each of the three channels is a scaled k-mean center
            flat_vals = x_curr.reshape(
                [x_curr.shape[0] * x_curr.shape[1], x_curr.shape[2]]
            ).T
            flat_vals = (flat_vals.T - flat_vals.mean(1)).T
            means = kmeans(flat_vals, 3, round_values=False).data.T.reshape(
                [x_curr.shape[0], x_curr.shape[1], 3]
            )
            x_curr_disp = (means - np.percentile(means, 0.5, (0, 1))) / (
                np.percentile(means, 99.5, (0, 1)) - np.percentile(means, 1, (0, 1))
            )
            x_curr_disp[x_curr_disp > 1] = 1
            x_curr_disp[x_curr_disp < 0] = 0
        else:
            x_curr_gray = x_curr
            x_curr_disp = x_curr

        axes[0].imshow(x_curr_disp, cmap=plt.get_cmap("gray"))
        if true_labels:
            axes[0].set_title(true_labels[row], **label_kwargs)
        axes[0].axis("off")

        if len(shap_values[0][row].shape) == 2:
            abs_vals = np.stack(
                [np.abs(shap_values[i]) for i in range(len(shap_values))], 0
            ).flatten()
        else:
            abs_vals = np.stack(
                [np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0
            ).flatten()
        max_val = np.nanpercentile(abs_vals, 99.9)

        for i in range(len(shap_values)):
            if labels is not None:
                axes[i + 1].set_title(labels[row, i], **label_kwargs)

            sv = (
                shap_values[i][row]
                if len(shap_values[i][row].shape) == 2
                else shap_values[i][row].sum(-1)
            )
            axes[i + 1].imshow(
                x_curr_gray,
                cmap=plt.get_cmap("gray"),
                alpha=0.15,
                extent=(-1, sv.shape[1], sv.shape[0], -1),
            )
            im = axes[i + 1].imshow(sv, cmap=cmap, vmin=-max_val, vmax=max_val)
            axes[i + 1].axis("off")

    fig.delaxes(axes[5])

    fig.subplots_adjust(wspace=0.05, hspace=0.02)

    cb = plt.colorbar(
        im,
        ax=axes,
        orientation="horizontal",
        label="SHAP value",
        pad=0.01,
    )
    cb.outline.set_visible(False)

    if save_as is not None:
        plt.savefig(save_as, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
