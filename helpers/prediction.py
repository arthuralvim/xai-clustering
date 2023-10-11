import numpy as np


def get_labels_from_prediction(predictions):
    truth_labels = []
    preds_labels = []
    for truth, pred in predictions:
        truth_labels.append(truth)
        preds_labels.append(pred)
    return np.concatenate(truth_labels), np.concatenate(preds_labels)
