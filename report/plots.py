import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib import colors as mlt_colors
import seaborn as sns
from dataclasses import dataclass
from matplotlib import markers
from enum import Enum

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


class LineStyle(Enum):
    SOLID = "-"  # solid
    DASHED = "--"  # dashed
    DASHDOT = "-."  # dashdot
    DOTTED = ":"  # dotted


class Markers(Enum):
    POINT = "."  # 'point',
    PIXEL = ","  # 'pixel',
    CIRCLE = "o"  # 'circle',
    TRIANGLE_DOWN = "v"  # 'triangle_down',
    TRIANGLE_UP = "^"  # 'triangle_up',
    TRIANGLE_LEFT = "<"  # 'triangle_left',
    TRIANGLE_RIGHT = ">"  # 'triangle_right',
    TRI_DOWN = "1"  # 'tri_down',
    TRI_UP = "2"  # 'tri_up',
    TRI_LEFT = "3"  # 'tri_left',
    TRI_RIGHT = "4"  # 'tri_right',
    OCTAGON = "8"  # 'octagon',
    SQUARE = "s"  # 'square',
    PENTAGON = "p"  # 'pentagon',
    STAR = "*"  # 'star',
    HEXAGON1 = "h"  # 'hexagon1',
    HEXAGON2 = "H"  # 'hexagon2',
    PLUS = "+"  # 'plus',
    X = "x"  # 'x',
    DIAMOND = "D"  # 'diamond',
    THIN_DIAMOND = "d"  # 'thin_diamond',
    VLINE = "|"  # 'vline',
    HLINE = "_"  # 'hline',
    PLUS_FILLED = "P"  # 'plus_filled',
    X_FILLED = "X"  # 'x_filled',
    TICKLEFT = "tickleft"
    TICKRIGHT = "tickright"
    TICKUP = "tickup"
    TICKDOWN = "tickdown"
    CARETLEFT = "caretleft"
    CARETRIGHT = "caretright"
    CARETUP = "caretup"
    CARETDOWN = "caretdown"
    CARETLEFTBASE = "caretleftbase"
    CARETRIGHTBASE = "caretrightbase"
    CARETUPBASE = "caretupbase"
    CARETDOWNBASE = "caretdownbase"


@dataclass
class PlotSettings(object):
    linestyle: LineStyle
    marker: Markers
    color: str = "blue"

    @property
    def kw(self):
        return {
            "linestyle": self.linestyle.value,
            "marker": self.marker.value,
            "color": self.color,
        }


training_set = PlotSettings(
    linestyle=LineStyle.SOLID, marker=Markers.CIRCLE, color="#1E6CA8"
)
validation_set = PlotSettings(
    linestyle=LineStyle.DASHED, marker=Markers.TRIANGLE_UP, color="#FFD441"
)

clustering_set = [
    PlotSettings(
        linestyle=LineStyle.DASHED, marker=Markers.CIRCLE, color=classes_colormap[0]
    ),
    PlotSettings(
        linestyle=LineStyle.DASHED,
        marker=Markers.TRIANGLE_UP,
        color=classes_colormap[1],
    ),
    PlotSettings(
        linestyle=LineStyle.DASHED, marker=Markers.SQUARE, color=classes_colormap[2]
    ),
    PlotSettings(
        linestyle=LineStyle.DASHED, marker=Markers.STAR, color=classes_colormap[3]
    ),
    PlotSettings(
        linestyle=LineStyle.DASHED,
        marker=Markers.PLUS_FILLED,
        color=classes_colormap[4],
    ),
]


def plot_loss_curve(df_metrics, show=True, save_as=None):
    epochs = range(len(df_metrics["train_loss_epoch"]))

    fig = plt.figure(figsize=(6, 6))
    plt.plot(
        epochs,
        df_metrics["train_loss_epoch"],
        label="Treinamento",
        **training_set.kw,
    )
    plt.plot(
        epochs, df_metrics["val_loss_epoch"], label="Validação", **validation_set.kw
    )

    plt.ylabel("Erro", fontsize=14)
    plt.xlabel("Época", fontsize=14)
    plt.title("Curva de Decaimento do Erro", fontsize=18)

    plt.legend(loc="best")

    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close(fig)


def plot_acc_curve(df_metrics, show=True, title="Curva de Acurácia", save_as=None):
    epochs = range(len(df_metrics["train_acc_epoch"]))

    fig = plt.figure(figsize=(6, 6))
    plt.plot(
        epochs,
        df_metrics["train_acc_epoch"],
        label="Treinamento",
        **training_set.kw,
    )
    plt.plot(
        epochs, df_metrics["val_acc_epoch"], label="Validação", **validation_set.kw
    )

    plt.xlim([0, max(epochs)])
    plt.ylim([0, 100])

    plt.ylabel("Acurácia", fontsize=14)
    plt.xlabel("Época", fontsize=14)
    plt.title(title, fontsize=18)
    plt.legend(loc="best")

    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close(fig)


def plot_clustering_metrics(
    df_metrics, show=True, title="Métricas de Agrupamento", save_as=None
):
    epochs = range(df_metrics.shape[0])

    fig = plt.figure(figsize=(6, 6))

    for n, metric in enumerate(
        [
            "clu_homogeneity_score",
            "clu_completeness_score",
            "clu_v_measure_score",
            "clu_adjusted_rand_score",
            "clu_adjusted_mutual_info_score",
        ]
    ):
        plt.plot(epochs, df_metrics[metric], label=f"{metric}", **clustering_set[n].kw)

    plt.xlim([0, max(epochs)])
    plt.ylim([df_metrics.to_numpy().min(), 1])

    plt.ylabel("Métrica", fontsize=14)
    plt.xlabel("Época", fontsize=14)
    plt.title(title, fontsize=18)
    plt.legend(loc="best")

    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close(fig)


def plot_analysis(
    df_metrics, show=True, title="Análise do Aprendizado", figsize=(19, 9), save_as=None
):
    epochs = range(len(df_metrics["train_acc_epoch"]))
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # loss
    axes[0].plot(
        epochs,
        df_metrics["train_loss_epoch"],
        label="Treinamento",
        **training_set.kw,
    )
    axes[0].plot(
        epochs, df_metrics["val_loss_epoch"], label="Validação", **validation_set.kw
    )
    axes[0].set_xlim([0, max(epochs)])
    axes[0].set_ylabel("Erro", fontsize=14)
    axes[0].set_xlabel("Época", fontsize=14)
    axes[0].set_title("Curva de Erro", fontsize=18)
    axes[0].legend(loc="best")

    # acc
    axes[1].plot(
        epochs,
        df_metrics["train_acc_epoch"],
        label="Treinamento",
        **training_set.kw,
    )
    axes[1].plot(
        epochs, df_metrics["val_acc_epoch"], label="Validação", **validation_set.kw
    )
    axes[1].set_xlim([0, max(epochs)])
    axes[1].set_ylim([0, 100])

    axes[1].set_ylabel("Acurácia", fontsize=14)
    axes[1].set_xlabel("Época", fontsize=14)
    axes[1].set_title("Curva de Acurácia", fontsize=18)
    axes[1].legend(loc="best")

    fig.suptitle(title, fontsize=18)

    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close(fig)
