"""Shared settings and helpers for all experiments.

Protocol
--------
* Official 60k training set -> 50k train / 10k validation (fixed split, seed 0).
* All model / hyper-parameter choices use the validation set only.
* The official 10k test set is used once, by final_evaluation.py, for the selected model.
* Learning rate: chosen per configuration from LR_GRID by final-epoch validation accuracy
  (seed 0); the chosen setting is then re-trained with SEEDS and reported as mean +- std.
"""
from __future__ import annotations

import argparse

import numpy as np

from src.data import load_splits
from src.model import MLP
from src.training import train
from src.utils import DATA

EPOCHS = 20
BATCH_SIZE = 64
SEEDS = [0, 1, 2]
LR_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

# Named model configurations: (hidden activation, output/loss, initialisation)
CONFIGS = {
    "original":       dict(hidden_activation="sigmoid", loss="sigmoid_mse", init="normal_0.01"),
    "sigmoid_ce":     dict(hidden_activation="sigmoid", loss="softmax_ce", init="normal_0.01"),
    "sigmoid_ce_xav": dict(hidden_activation="sigmoid", loss="softmax_ce", init="xavier"),
    "relu_ce_small":  dict(hidden_activation="relu", loss="softmax_ce", init="normal_0.01"),
    "relu_ce_he":     dict(hidden_activation="relu", loss="softmax_ce", init="he"),
}

LABELS = {
    "original":       "Sigmoid + MSE, N(0,0.01²) init (original)",
    "sigmoid_ce":     "Sigmoid + softmax/CE, N(0,0.01²) init",
    "sigmoid_ce_xav": "Sigmoid + softmax/CE, Xavier init",
    "relu_ce_small":  "ReLU + softmax/CE, N(0,0.01²) init",
    "relu_ce_he":     "ReLU + softmax/CE, He init",
}


def parse_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--quick", action="store_true",
                   help="smoke test: 2 epochs, 1 seed, 3 learning rates (numbers not meaningful)")
    p.add_argument("--data-dir", default=DATA)
    p.add_argument("--from-log", action="store_true",
                   help="(ablation only) rebuild tables/figures from results/logs without training")
    return p.parse_args()


def settings(args):
    if args.quick:
        return dict(epochs=2, seeds=[0], lr_grid=[0.03, 0.1, 0.3])
    return dict(epochs=EPOCHS, seeds=SEEDS, lr_grid=LR_GRID)


def get_data(args):
    d = load_splits(args.data_dir)
    return d["X_train"], d["y_train"], d["X_val"], d["y_val"]


def build(config_name, hidden, seed):
    return MLP([784, hidden, 10], seed=seed, **CONFIGS[config_name])


def run(config_name, hidden, lr, seed, data, epochs, batch_size=BATCH_SIZE, verbose=False):
    X_tr, y_tr, X_val, y_val = data
    model = build(config_name, hidden, seed)
    hist = train(model, X_tr, y_tr, X_val, y_val, epochs=epochs, batch_size=batch_size,
                 lr=lr, seed=seed, verbose=verbose)
    return model, hist


def tune_lr(config_name, hidden, data, epochs, lr_grid):
    """Pick the learning rate with the best final-epoch validation accuracy (seed 0)."""
    scores = {}
    for lr in lr_grid:
        _, h = run(config_name, hidden, lr, 0, data, epochs)
        acc = h["val_acc"][-1] if not h["diverged"] else 0.0
        scores[lr] = acc
        print(f"    lr={lr:<5} -> val acc {acc:.4f}{'  (diverged)' if h['diverged'] else ''}")
    best = max(scores, key=scores.get)
    edge = best in (lr_grid[0], lr_grid[-1])
    return best, scores, edge


def summarise(histories):
    fin = lambda k: np.array([h[k][-1] for h in histories])  # noqa: E731
    return {k: (float(fin(k).mean()), float(fin(k).std(ddof=1)) if len(histories) > 1 else 0.0)
            for k in ["train_acc", "val_acc", "train_loss", "val_loss"]}
