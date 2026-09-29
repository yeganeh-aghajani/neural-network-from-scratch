"""Small I/O helpers shared by the experiment scripts."""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
TABLES = os.path.join(RESULTS, "tables")
LOGS = os.path.join(RESULTS, "logs")
MODELS = os.path.join(RESULTS, "models")
DATA = os.path.join(ROOT, "data")

for _d in (FIGURES, TABLES, LOGS, MODELS):
    os.makedirs(_d, exist_ok=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def mean_curves(histories):
    """Average per-epoch curves over seeds."""
    keys = ["train_loss", "train_acc", "val_loss", "val_acc"]
    return {k: np.mean([h[k] for h in histories], axis=0).tolist() for k in keys}
