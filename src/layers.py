"""Layers with hand-written forward and backward passes (NumPy only).

Convention: a mini-batch is a matrix X of shape (N, D) with one sample per row.
A dense layer computes Z = X W + b with W of shape (D_in, D_out).
"""
from __future__ import annotations

import numpy as np


def init_weights(kind: str, fan_in: int, fan_out: int, rng: np.random.Generator, dtype):
    """Weight initialisation schemes.

    - "normal_0.01": N(0, 0.01^2), the scheme used in the original course code.
    - "xavier":      N(0, 2 / (fan_in + fan_out))  (Glorot normal; suited to sigmoid/tanh).
    - "he":          N(0, 2 / fan_in)               (suited to ReLU).
    """
    if kind == "normal_0.01":
        std = 0.01
    elif kind == "xavier":
        std = np.sqrt(2.0 / (fan_in + fan_out))
    elif kind == "he":
        std = np.sqrt(2.0 / fan_in)
    else:
        raise ValueError(f"Unknown init '{kind}'")
    return (rng.standard_normal((fan_in, fan_out)) * std).astype(dtype)


class Dense:
    """Fully connected layer: Z = X W + b."""

    def __init__(self, in_dim: int, out_dim: int, init: str, rng, dtype=np.float32):
        self.W = init_weights(init, in_dim, out_dim, rng, dtype)
        self.b = np.zeros(out_dim, dtype=dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W + self.b

    def backward(self, dz: np.ndarray) -> np.ndarray:
        # dL/dW = X^T dL/dZ,  dL/db = sum_n dL/dZ_n,  dL/dX = dL/dZ W^T
        self.dW = self.x.T @ dz
        self.db = dz.sum(axis=0)
        return dz @ self.W.T


class Sigmoid:
    def forward(self, z: np.ndarray) -> np.ndarray:
        # 0.5 * (1 + tanh(z/2)) is mathematically identical to 1/(1+e^-z)
        # but never overflows for large |z|.
        self.a = 0.5 * (1.0 + np.tanh(0.5 * z))
        return self.a

    def backward(self, da: np.ndarray) -> np.ndarray:
        return da * self.a * (1.0 - self.a)


class ReLU:
    def forward(self, z: np.ndarray) -> np.ndarray:
        self.mask = z > 0
        return z * self.mask

    def backward(self, da: np.ndarray) -> np.ndarray:
        return da * self.mask


ACTIVATIONS = {"sigmoid": Sigmoid, "relu": ReLU}
