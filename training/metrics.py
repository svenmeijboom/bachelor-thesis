import numpy as np

from transformers.data.metrics.squad_metrics import compute_f1, compute_exact


def f1_metric(y_true, y_pred):
    return np.mean([compute_f1(a_true, a_pred) for a_true, a_pred in zip(y_true, y_pred)])


def em_metric(y_true, y_pred):
    return np.mean([compute_exact(a_true, a_pred) for a_true, a_pred in zip(y_true, y_pred)])
