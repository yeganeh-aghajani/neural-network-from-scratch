"""Multilayer perceptron assembled from the hand-written layers in `layers.py`."""
from __future__ import annotations

import json

import numpy as np

from .layers import ACTIVATIONS, Dense
from .losses import LOSSES


class MLP:
    """MLP with configurable hidden layers, hidden activation, output/loss and initialisation.

    Example: MLP([784, 64, 10], hidden_activation="relu", loss="softmax_ce", init="he")
    """

    def __init__(self, layer_sizes, hidden_activation="relu", loss="softmax_ce",
                 init="he", seed=0, dtype=np.float32):
        self.config = dict(layer_sizes=list(layer_sizes), hidden_activation=hidden_activation,
                           loss=loss, init=init, seed=seed, dtype=np.dtype(dtype).name)
        rng = np.random.default_rng(seed)
        self.dense = []
        self.acts = []
        for i, (d_in, d_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.dense.append(Dense(d_in, d_out, init, rng, dtype))
            if i < len(layer_sizes) - 2:                  # no activation on the logits
                self.acts.append(ACTIVATIONS[hidden_activation]())
        self.loss_fn = LOSSES[loss]()

    # ----- forward / backward -------------------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Return the logits. Pre-activations and activations are cached for analysis."""
        self.pre_acts, self.hidden = [], []
        h = X
        for i, layer in enumerate(self.dense):
            z = layer.forward(h)
            if i < len(self.acts):
                self.pre_acts.append(z)
                h = self.acts[i].forward(z)
                self.hidden.append(h)
            else:
                h = z
        return h

    def compute_loss(self, X: np.ndarray, Y_onehot: np.ndarray) -> float:
        return self.loss_fn.forward(self.forward(X), Y_onehot)

    def backward(self) -> None:
        """Backpropagate from the loss; fills dW, db of every Dense layer."""
        grad = self.loss_fn.backward()                    # dL/d(logits)
        for i in reversed(range(len(self.dense))):
            grad = self.dense[i].backward(grad)
            if i > 0:
                grad = self.acts[i - 1].backward(grad)

    # ----- inference ----------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.loss_fn.output(self.forward(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(axis=1)            # argmax is the same for logits and outputs

    # ----- parameters ---------------------------------------------------------
    def parameters(self):
        """List of (name, param, grad) triples."""
        out = []
        for i, layer in enumerate(self.dense):
            out.append((f"W{i + 1}", layer.W, layer.dW))
            out.append((f"b{i + 1}", layer.b, layer.db))
        return out

    def num_parameters(self) -> int:
        return sum(p.size for _, p, _ in self.parameters())

    def save(self, path: str) -> None:
        arrays = {name: p for name, p, _ in self.parameters()}
        np.savez(path, config=json.dumps(self.config), **arrays)

    @classmethod
    def load(cls, path: str) -> "MLP":
        data = np.load(path)
        cfg = json.loads(str(data["config"]))
        model = cls(cfg["layer_sizes"], cfg["hidden_activation"], cfg["loss"], cfg["init"],
                    cfg["seed"], np.dtype(cfg["dtype"]))
        for i, layer in enumerate(model.dense):
            layer.W = data[f"W{i + 1}"]
            layer.b = data[f"b{i + 1}"]
        return model
