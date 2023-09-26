# metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib import colors as mlt_colors
import pandas as pd
import seaborn as sns


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


def basic_report(truth, predictions, labels, label_names, show=True, save_as=None):
    cr = classification_report(
        truth,
        predictions,
        labels=labels,
        target_names=label_names,
        digits=3,
        zero_division=0,
    )
    confusion = pd.DataFrame(
        confusion_matrix(truth, predictions),
        index=label_names,
        columns=[s[:3] for s in label_names],
    )
    print("Classification report")
    print(cr)
    plot_confusion_matrix(confusion, label_names, show=show, save_as=save_as)
    return confusion
