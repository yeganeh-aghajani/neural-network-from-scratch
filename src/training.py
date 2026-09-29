"""Mini-batch gradient descent, written out explicitly."""
from __future__ import annotations

import time

import numpy as np

from .data import one_hot


def sgd_step(model, lr: float) -> None:
    """Plain gradient descent update: theta <- theta - lr * dL/dtheta."""
    for layer in model.dense:
        layer.W -= lr * layer.dW
        layer.b -= lr * layer.db


def evaluate(model, X, y, batch_size: int = 10_000):
    """Return (mean loss, accuracy) over the whole set."""
    total_loss, correct = 0.0, 0
    for start in range(0, len(X), batch_size):
        xb, yb = X[start:start + batch_size], y[start:start + batch_size]
        logits = model.forward(xb)
        total_loss += model.loss_fn.forward(logits, one_hot(yb, dtype=xb.dtype)) * len(xb)
        correct += int((logits.argmax(axis=1) == yb).sum())
    return total_loss / len(X), correct / len(X)


def train(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=64, lr=0.1,
          seed=0, verbose=True):
    """Train with shuffled mini-batches; evaluate on train and validation after every epoch.

    Returns a history dict. If the loss becomes non-finite, training stops and
    history["diverged"] is set to True.
    """
    rng = np.random.default_rng(seed)
    Y_train = one_hot(y_train, dtype=X_train.dtype)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
            "epoch_time": [], "diverged": False}
    n = len(X_train)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        perm = rng.permutation(n)                          # reshuffle every epoch
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            model.compute_loss(X_train[idx], Y_train[idx])  # forward pass
            model.backward()                               # backpropagation
            sgd_step(model, lr)                            # parameter update
        tr_loss, tr_acc = evaluate(model, X_train, y_train)
        va_loss, va_acc = evaluate(model, X_val, y_val)
        for k, v in zip(["train_loss", "train_acc", "val_loss", "val_acc", "epoch_time"],
                        [tr_loss, tr_acc, va_loss, va_acc, time.time() - t0]):
            hist[k].append(float(v))
        if verbose:
            print(f"  epoch {epoch:2d}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f}"
                  f" | val loss {va_loss:.4f} acc {va_acc:.4f}")
        if not np.isfinite(tr_loss):
            hist["diverged"] = True
            break
    return hist
