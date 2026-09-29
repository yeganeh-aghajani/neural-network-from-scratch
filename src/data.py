"""MNIST loading, IDX parsing and a reproducible train/validation split.

The IDX parser follows the same logic as the original course code
(`original/mlp.py`), but returns whole arrays instead of a Python list of
(x, y) tuples so that training can be vectorised over mini-batches.
"""
from __future__ import annotations

import gzip
import os
import urllib.request

import numpy as np

# Public mirrors of the original LeCun files (tried in order).
MIRRORS = [
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]

FILES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images": "t10k-images-idx3-ubyte",
    "test_labels": "t10k-labels-idx1-ubyte",
}


def _find_local(data_dir: str, stem: str) -> str | None:
    """Return a path to `stem` or `stem.gz` inside data_dir (searched recursively)."""
    for root, _, files in os.walk(data_dir):
        for name in (stem, stem + ".gz"):
            if name in files:
                return os.path.join(root, name)
    return None


def download_mnist(data_dir: str = "data") -> None:
    """Download the four MNIST files into data_dir if they are not present."""
    os.makedirs(data_dir, exist_ok=True)
    for stem in FILES.values():
        if _find_local(data_dir, stem) is not None:
            continue
        target = os.path.join(data_dir, stem + ".gz")
        last_err = None
        for mirror in MIRRORS:
            try:
                print(f"Downloading {stem}.gz from {mirror}")
                urllib.request.urlretrieve(mirror + stem + ".gz", target)
                last_err = None
                break
            except Exception as err:  # try next mirror
                last_err = err
        if last_err is not None:
            raise RuntimeError(
                f"Could not download {stem}. Place the MNIST files in '{data_dir}' manually."
            ) from last_err


def read_idx(path: str) -> np.ndarray:
    """Parse an IDX file (optionally gzipped) into a uint8 array."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        n_dims = magic & 0xFF
        shape = tuple(int.from_bytes(f.read(4), "big") for _ in range(n_dims))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(shape)


def load_mnist(data_dir: str = "data", dtype=np.float32):
    """Return (X_train, y_train, X_test, y_test).

    Images are flattened to 784-vectors and scaled to [0, 1];
    labels are integer class indices (0-9).
    """
    download_mnist(data_dir)
    arrays = {k: read_idx(_find_local(data_dir, v)) for k, v in FILES.items()}
    X_train = arrays["train_images"].reshape(-1, 784).astype(dtype) / 255.0
    X_test = arrays["test_images"].reshape(-1, 784).astype(dtype) / 255.0
    y_train = arrays["train_labels"].astype(np.int64)
    y_test = arrays["test_labels"].astype(np.int64)
    return X_train, y_train, X_test, y_test


def train_val_split(X, y, n_val: int = 10_000, seed: int = 0):
    """Shuffle once with a fixed seed and hold out n_val samples for validation."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def load_splits(data_dir: str = "data", n_val: int = 10_000, split_seed: int = 0):
    """Train (50k) / validation (10k) from the official training set; test = official 10k.

    The test split is returned for completeness but must only be used by
    `experiments/final_evaluation.py`.
    """
    X_train_full, y_train_full, X_test, y_test = load_mnist(data_dir)
    X_tr, y_tr, X_val, y_val = train_val_split(X_train_full, y_train_full, n_val, split_seed)
    return {
        "X_train": X_tr, "y_train": y_tr,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def one_hot(y: np.ndarray, n_classes: int = 10, dtype=np.float32) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=dtype)
    out[np.arange(len(y)), y] = 1.0
    return out
