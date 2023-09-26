from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def tsne_scatter(features, labels, dimensions=2, random_state=42, save_as=None):
    if dimensions not in (2, 3):
        raise ValueError("tsne_scatter can only plot in 2d or 3d.")

    features_embedded = TSNE(
        n_components=dimensions, random_state=random_state
    ).fit_transform(features)

    fig, ax = plt.subplots(figsize=(6, 6))

    if dimensions == 3:
        ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        *zip(*features_embedded[np.where(labels == 1)]),
        marker="o",
        color="r",
        s=2,
        alpha=0.7,
        label="Fraude"
    )
    ax.scatter(
        *zip(*features_embedded[np.where(labels == 0)]),
        marker="o",
        color="g",
        s=2,
        alpha=0.3,
        label="Normal"
    )
    plt.title("TSNE da Amostra", fontsize=18)

    plt.legend(loc="best")

    if save_as is not None:
        plt.savefig(save_as)

    plt.show()
