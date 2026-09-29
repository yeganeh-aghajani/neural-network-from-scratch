"""Analysis of what the first hidden layer has learned.

Extends the neuron analysis of the original project:
  * weight maps of every hidden unit (kept from the original);
  * top-activating samples, ranked by the *pre-activation* z so that ties caused
    by saturated sigmoid outputs (a = 1.000) cannot occur;
  * class-conditional mean activation and a simple class-selectivity index;
  * ablation importance: drop in validation accuracy when one unit is silenced;
  * a check of the original heuristic "high weight variance = more meaningful neuron".
"""
from __future__ import annotations

import numpy as np


def hidden_layer(model, X):
    """Return (pre-activations Z, activations A) of the first hidden layer."""
    model.forward(X)
    return model.pre_acts[0], model.hidden[0]


def weight_variance_scores(model) -> np.ndarray:
    """Heuristic from the original report: variance of each unit's 784 input weights."""
    return model.dense[0].W.var(axis=0)


def top_activating(model, X, unit: int, k: int = 8):
    """Indices and pre-activations of the k samples that drive `unit` the most."""
    Z, _ = hidden_layer(model, X)
    z = Z[:, unit]
    idx = np.argsort(-z, kind="stable")[:k]
    return idx, z[idx]


def count_saturated(model, X, unit: int, threshold: float = 0.9995) -> int:
    """How many samples push a (sigmoid) unit to a displayed value of 1.000."""
    _, A = hidden_layer(model, X)
    return int((A[:, unit] >= threshold).sum())


def class_mean_activation(model, X, y, n_classes: int = 10) -> np.ndarray:
    """M[j, c] = mean activation of hidden unit j over samples of class c."""
    _, A = hidden_layer(model, X)
    return np.stack([A[y == c].mean(axis=0) for c in range(n_classes)], axis=1)


def selectivity_index(M: np.ndarray) -> np.ndarray:
    """(max_c - mean_other_c) / (max_c + mean_other_c), in [0, 1] for non-negative activations.

    0 = responds equally to all digits, 1 = responds to a single digit only.
    """
    top = M.max(axis=1)
    others = (M.sum(axis=1) - top) / (M.shape[1] - 1)
    return np.divide(top - others, top + others, out=np.zeros_like(top), where=(top + others) > 0)


def ablation_importance(model, X, y) -> np.ndarray:
    """Drop in accuracy when each first-layer hidden unit is set to zero.

    Only defined for networks with exactly one hidden layer (the case studied here).
    """
    assert len(model.dense) == 2, "ablation_importance expects a single hidden layer"
    _, A = hidden_layer(model, X)
    W2, b2 = model.dense[1].W, model.dense[1].b
    logits = A @ W2 + b2
    base = (logits.argmax(axis=1) == y).mean()
    drops = np.empty(A.shape[1])
    for j in range(A.shape[1]):
        ablated = logits - np.outer(A[:, j], W2[j])     # remove unit j's contribution
        drops[j] = base - (ablated.argmax(axis=1) == y).mean()
    return drops


def spearman(a, b) -> float:
    """Spearman rank correlation (no tie correction; ties are negligible here)."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])
