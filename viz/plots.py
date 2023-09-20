import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib import colors as mlt_colors
import seaborn as sns


train_test_colormap = np.array(["#1E6CA8", "#FFD441"])
classes_colormap = np.array(
    [
        "#2A9532",
        "#CF2528",
        "#885CB1",
        "#32346B",
        "#585858",
        "#783533",
        "#DE6DB7",
        "#83CDC0",
        "#1DB5C7",
        "#F9766A",
    ]
)


def plot_loss_curve(history, show=True, save_as=None):
    epochs = range(len(history["loss/train"]))

    fig = plt.figure(figsize=(6, 6))
    plt.plot(
        epochs,
        history["loss/train"],
        linestyle="-",
        marker="o",
        color="orange",
        label="Treinamento",
    )
    plt.plot(
        epochs,
        history["loss/val"],
        linestyle="--",
        marker="^",
        color="blue",
        label="Validação",
    )

    plt.ylabel("Custo", fontsize=14)
    plt.xlabel("Época", fontsize=14)
    plt.title("Curva de Decaimento do Erro", fontsize=18)

    plt.legend(loc="best")

    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close(fig)


def plot_acc_curve(history, show=True, save_as=None):
    epochs = range(len(history["acc/train"]))

    fig = plt.figure(figsize=(6, 6))
    plt.plot(
        epochs,
        history["acc/train"],
        linestyle="-",
        marker="o",
        color="orange",
        label="Treinamento",
    )
    plt.plot(
        epochs,
        history["acc/val"],
        linestyle="--",
        marker="^",
        color="blue",
        label="Validação",
    )

    plt.xlim([0, max(epochs)])
    plt.ylim([0, 100])

    plt.ylabel("Acurácia", fontsize=14)
    plt.xlabel("Época", fontsize=14)
    plt.title("Curva de Acurácia", fontsize=18)
    plt.legend(loc="best")

    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close(fig)


def plot_cv_indices(folds, targets, ax, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    n_splits = len(folds)
    n_targets = len(np.unique(targets))
    for ii, (tr, tt, tr_targets, tt_targets) in enumerate(folds):
        indices = np.array([np.nan] * len(targets))
        indices[tt] = 1
        indices[tr] = 0
        print()
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=train_test_colormap[indices.astype(int)],
            marker="_",
            lw=lw,
        )

    # Plot classes
    ax.scatter(
        range(len(targets)),
        [ii + 1.5] * len(targets),
        c=classes_colormap[targets],
        marker="_",
        lw=lw,
    )

    yticklabels = [f"Fold {n}" for n in list(range(1, n_splits + 1))] + ["class"]
    ax.set(
        yticks=np.arange(n_splits + 1) + 0.5,
        yticklabels=yticklabels,
        ylim=[n_splits + 2.2, -0.2],
    )
    ax.set_title("CV Eurosat", fontsize=15)
    ax.legend(
        [Patch(color=train_test_colormap[0]), Patch(color=train_test_colormap[1])]
        + [Patch(color=classes_colormap[n]) for n in list(range(n_targets))],
        ["Validation set", "Training set"]
        + [f"Classe {n}" for n in list(range(1, len(targets)))],
        loc=(1.02, 0.2),
    )
    return ax


def plot_cv_setup(folds, targets, show=True, save_as=None):
    fig, ax = plt.subplots()
    plot_cv_indices(folds, targets, ax, lw=10)
    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion_matrix(confusion, label_names, show=True, save_as=None):
    fig = plt.figure(figsize=(6, 6))
    fx = sns.heatmap(confusion, annot=True, fmt=".0f", cmap="RdBu")
    fx.set_title("Confusion Matrix \n")
    fx.set_xlabel("\n Predicted Values\n")
    fx.set_ylabel("Actual Values\n")
    fx.xaxis.set_ticklabels(label_names)
    fx.yaxis.set_ticklabels(label_names)
    plt.yticks(rotation=0)
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close(fig)
