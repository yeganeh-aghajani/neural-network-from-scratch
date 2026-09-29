"""Matplotlib figures used by the experiment scripts."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_curves(curves: dict, path: str, title: str = "", show_loss: bool = True,
                acc_ylim=None):
    """curves: {label: {"train_loss", "val_loss", "train_acc", "val_acc"} (lists or mean arrays)}.

    Set show_loss=False when the curves use different loss functions (e.g. MSE vs
    cross-entropy), whose values are not comparable on one axis.
    """
    fig, axes = plt.subplots(1, 2 if show_loss else 1, figsize=(12 if show_loss else 7, 4.5))
    axes = np.atleast_1d(axes)
    acc_ax = axes[-1]
    for (label, h), c in zip(curves.items(), COLORS * 3):
        ep = np.arange(1, len(h["val_acc"]) + 1)
        if show_loss:
            axes[0].plot(ep, h["train_loss"], "--", color=c, alpha=0.7)
            axes[0].plot(ep, h["val_loss"], "-", color=c, label=label)
        acc_ax.plot(ep, np.array(h["train_acc"]) * 100, "--", color=c, alpha=0.7)
        acc_ax.plot(ep, np.array(h["val_acc"]) * 100, "-", color=c, label=label)
    if show_loss:
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss (solid: validation, dashed: train)")
    acc_ax.set(xlabel="Epoch", ylabel="Accuracy (%)",
               title="Accuracy (solid: validation, dashed: train)")
    if acc_ylim:
        acc_ax.set_ylim(*acc_ylim)
    for ax in axes:
        ax.grid(alpha=0.3)
    acc_ax.legend(fontsize=9)
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_bars_with_error(labels, means, stds, path, ylabel, title, ylim=None):
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(labels)), 4.2))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5, color=COLORS[: len(labels)])
    for xi, m in zip(x, means):
        ax.text(xi, m, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set(ylabel=ylabel, title=title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, path)


def plot_confusion(cm: np.ndarray, path: str, title: str = "Confusion matrix"):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted", ylabel="True", title=title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised")
    fig.tight_layout()
    _save(fig, path)


def plot_weight_maps(W1: np.ndarray, units, path: str, title: str, cols: int = 8,
                     subtitles=None):
    """Input-weight map of each hidden unit as a 28x28 image.

    Uses a diverging colormap centred at zero (red = positive, blue = negative) so the
    sign of every weight stays visible; the original min-max grey scaling hid it.
    """
    units = list(units)
    rows = int(np.ceil(len(units) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.55 * cols, 1.75 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for k, u in enumerate(units):
        w = W1[:, u].reshape(28, 28)
        lim = np.abs(w).max()
        axes[k].imshow(w, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[k].set_title(subtitles[k] if subtitles else f"n{u}", fontsize=8)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_image_grid(images, titles, path, title, cols=8):
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.75 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, img, t in zip(axes, images, titles):
        ax.imshow(img.reshape(28, 28), cmap="gray")
        ax.set_title(t, fontsize=8)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_top_activating(W1, X, units, top_idx, top_vals, y, path, title):
    """One row per unit: its weight map followed by its top-activating samples."""
    k = top_idx.shape[1]
    fig, axes = plt.subplots(len(units), k + 1, figsize=(1.45 * (k + 1), 1.65 * len(units)))
    for r, u in enumerate(units):
        w = W1[:, u].reshape(28, 28)
        lim = np.abs(w).max()
        axes[r, 0].imshow(w, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[r, 0].set_title(f"n{u} weights", fontsize=8)
        for c in range(k):
            i = top_idx[r, c]
            axes[r, c + 1].imshow(X[i].reshape(28, 28), cmap="gray")
            axes[r, c + 1].set_title(f"z={top_vals[r, c]:.1f} y={y[i]}", fontsize=7)
        for ax in axes[r]:
            ax.axis("off")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_class_heatmap(M, units, path, title):
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(units) + 1.5))
    im = ax.imshow(M[units], aspect="auto", cmap="viridis")
    ax.set(xticks=range(10), yticks=range(len(units)), xlabel="Digit class",
           title=title)
    ax.set_yticklabels([f"n{u}" for u in units], fontsize=7)
    fig.colorbar(im, ax=ax, label="Mean activation")
    fig.tight_layout()
    _save(fig, path)


def plot_scatter(x, y, path, xlabel, ylabel, title, annotate=None):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(x, y, s=22, color=COLORS[0], alpha=0.8)
    if annotate is not None:
        for i in annotate:
            ax.annotate(f"n{i}", (x[i], y[i]), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, path)


def plot_lr_sensitivity(scores: dict, path: str, title: str):
    """scores: {label: {lr: val_acc}} -> accuracy vs learning rate on a log axis."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (label, sc), c in zip(scores.items(), COLORS):
        lrs = sorted(sc, key=float)
        ax.plot([float(l) for l in lrs], [sc[l] * 100 for l in lrs], "o-", color=c, label=label)
    ax.set_xscale("log")
    ax.set(xlabel="Learning rate", ylabel="Validation accuracy (%)", title=title)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    _save(fig, path)
