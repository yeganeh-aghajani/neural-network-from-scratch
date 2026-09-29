"""Output layer + loss pairs. Each takes the logits Z of the last dense layer.

Both losses are averaged over the mini-batch, so the gradient returned by
`backward` is dL/dZ for the batch-mean loss.
"""
from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax, stabilised by subtracting the row maximum."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    """p = softmax(z),  L = -(1/N) sum_n sum_k y_nk log p_nk,  dL/dz = (p - y) / N."""

    name = "softmax_ce"

    def output(self, z: np.ndarray) -> np.ndarray:
        return softmax(z)

    def forward(self, z: np.ndarray, y_onehot: np.ndarray) -> float:
        z_shift = z - z.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.exp(z_shift).sum(axis=1, keepdims=True))
        log_p = z_shift - log_sum_exp                      # log-softmax, numerically stable
        self.p = np.exp(log_p)
        self.y = y_onehot
        return float(-(y_onehot * log_p).sum(axis=1).mean())

    def backward(self) -> np.ndarray:
        return (self.p - self.y) / self.y.shape[0]


class SigmoidMSE:
    """Formulation of the original course project.

    a = sigmoid(z),  L = (1/N) sum_n 0.5 * ||a_n - y_n||^2,
    dL/dz = (a - y) * a * (1 - a) / N.
    With N = 1 this is exactly the per-sample update of `original/mlp.py`.
    """

    name = "sigmoid_mse"

    def output(self, z: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + np.tanh(0.5 * z))

    def forward(self, z: np.ndarray, y_onehot: np.ndarray) -> float:
        self.a = self.output(z)
        self.y = y_onehot
        return float(0.5 * ((self.a - y_onehot) ** 2).sum(axis=1).mean())

    def backward(self) -> np.ndarray:
        return (self.a - self.y) * self.a * (1.0 - self.a) / self.y.shape[0]


LOSSES = {"softmax_ce": SoftmaxCrossEntropy, "sigmoid_mse": SigmoidMSE}
