"""Numerical gradient checking for the hand-written backpropagation.

For every parameter entry theta_i the analytic gradient from `model.backward()`
is compared with the central difference

    (L(theta + eps e_i) - L(theta - eps e_i)) / (2 eps)

using the relative error |g_a - g_n| / max(|g_a| + |g_n|, tiny).
Run in float64 on a small network, errors around 1e-7 or below indicate
a correct implementation.
"""
from __future__ import annotations

import numpy as np


def gradient_check(model, X, Y_onehot, eps: float = 1e-5, max_entries: int | None = None,
                   seed: int = 0):
    """Return {param_name: max relative error} for all (or a random subset of) entries."""
    model.compute_loss(X, Y_onehot)
    model.backward()
    analytic = {name: g.copy() for name, _, g in model.parameters()}

    rng = np.random.default_rng(seed)
    results = {}
    for name, param, _ in model.parameters():
        flat = param.reshape(-1)                       # view: edits modify the parameter
        idx = np.arange(flat.size)
        if max_entries is not None and flat.size > max_entries:
            idx = rng.choice(flat.size, max_entries, replace=False)
        num = np.zeros(len(idx))
        for j, i in enumerate(idx):
            old = flat[i]
            flat[i] = old + eps
            lp = model.compute_loss(X, Y_onehot)
            flat[i] = old - eps
            lm = model.compute_loss(X, Y_onehot)
            flat[i] = old
            num[j] = (lp - lm) / (2 * eps)
        ana = analytic[name].reshape(-1)[idx]
        rel = np.abs(ana - num) / np.maximum(np.abs(ana) + np.abs(num), 1e-12)
        results[name] = float(rel.max())
    return results
