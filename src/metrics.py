"""Classification metrics implemented with NumPy."""
from __future__ import annotations

import numpy as np


def accuracy(y_true, y_pred) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def confusion_matrix(y_true, y_pred, n_classes: int = 10) -> np.ndarray:
    """cm[i, j] = number of samples with true class i predicted as class j."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_report(cm: np.ndarray):
    """Precision, recall, F1 and support for every class, plus the macro average."""
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    rows = [dict(cls=int(k), precision=float(precision[k]), recall=float(recall[k]),
                 f1=float(f1[k]), support=int(support[k])) for k in range(len(tp))]
    macro = dict(precision=float(precision.mean()), recall=float(recall.mean()),
                 f1=float(f1.mean()))
    return rows, macro
