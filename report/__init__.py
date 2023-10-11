# metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import pandas as pd


def plot_confusion_matrix(confusion, label_names, show=True, save_as=None):
    conf_plot = ConfusionMatrixDisplay(
        confusion_matrix=confusion, display_labels=label_names
    )
    conf_plot.plot(cmap=plt.cm.Grays)
    conf_plot.ax_.set_title("Matriz de Confusão")
    conf_plot.ax_.set_xlabel("\n Predição\n")
    conf_plot.ax_.set_ylabel("Rótulos\n")
    conf_plot.ax_.xaxis.set_ticklabels(label_names)
    conf_plot.ax_.yaxis.set_ticklabels(label_names)
    plt.yticks(rotation=0)
    plt.xticks(rotation=30)

    if show:
        plt.show()
    if save_as is not None:
        plt.savefig(save_as)

    plt.close()


def basic_report(truth, predictions, labels, label_names, show=True, save_as=None):
    cr = classification_report(
        truth,
        predictions,
        labels=labels,
        target_names=label_names,
        digits=3,
        zero_division=0,
    )

    print(cr)
    confusion = confusion_matrix(truth, predictions)
    confusion_pd = pd.DataFrame(
        confusion,
        index=label_names,
        columns=[s[:3] for s in label_names],
    )
    print(confusion_pd)

    plot_confusion_matrix(confusion, label_names, show=show, save_as=save_as)
