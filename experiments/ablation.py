"""Ablation study at a fixed architecture (784-64-10), mini-batch SGD (batch 64), 20 epochs.

Each row changes one ingredient relative to the row above it, so the effect of the
output/loss formulation, the initialisation and the hidden activation can be seen separately.
"""
import sys

import numpy as np

from experiments.common import BATCH_SIZE, LABELS, get_data, parse_args, run, settings, summarise, tune_lr
from src.utils import FIGURES, LOGS, TABLES, load_json, markdown_table, mean_curves, save_json
from src.visualize import plot_bars_with_error, plot_curves, plot_lr_sensitivity

ORDER = ["original", "sigmoid_ce", "sigmoid_ce_xav", "relu_ce_small", "relu_ce_he"]
HIDDEN = 64

def report(results):
    """Write the table and figures from a results dict (also usable with --from-log)."""
    rows = []
    base = results["original"]["summary"]["val_acc"][0]
    for name in ORDER:
        r = results[name]
        s = r["summary"]
        ep1 = np.array([h["val_acc"][0] for h in r["histories"]]) * 100
        rows.append([LABELS[name], f"{r['lr']}" + (" *" if r["lr_at_grid_edge"] else ""),
                     f"{ep1.mean():.2f}",
                     f"{s['train_acc'][0]*100:.2f} ± {s['train_acc'][1]*100:.2f}",
                     f"{s['val_acc'][0]*100:.2f} ± {s['val_acc'][1]*100:.2f}",
                     f"{(s['val_acc'][0]-base)*100:+.2f}"])
    table = markdown_table(["Configuration", "Best LR", "Val acc after epoch 1 (%)",
                            "Final train acc (%)", "Final val acc (%)", "Δ val vs original (pp)"],
                           rows)
    open(f"{TABLES}/ablation.md", "w").write(table + "\n")
    print("\n" + table)

    plot_curves({LABELS[n]: mean_curves(results[n]["histories"]) for n in ORDER},
                f"{FIGURES}/ablation_curves.png", "Ablation (784-64-10, mean over seeds)",
                show_loss=False, acc_ylim=(85, 100))
    plot_lr_sensitivity({LABELS[n]: results[n]["lr_scores"] for n in ORDER},
                        f"{FIGURES}/ablation_lr_sensitivity.png",
                        "Validation accuracy after 20 epochs vs learning rate (seed 0)")
    plot_bars_with_error(
        ["Sigmoid+MSE\n(original)", "Sigmoid+CE", "Sigmoid+CE\nXavier", "ReLU+CE\nsmall init",
         "ReLU+CE\nHe"],
        [results[n]["summary"]["val_acc"][0] * 100 for n in ORDER],
        [results[n]["summary"]["val_acc"][1] * 100 for n in ORDER],
        f"{FIGURES}/ablation_bars.png", "Validation accuracy (%)",
        "Ablation: final validation accuracy (mean ± std over seeds)",
        ylim=(min(results[n]["summary"]["val_acc"][0] for n in ORDER) * 100 - 2, 100))


if __name__ == "__main__":
    if "--from-log" in sys.argv:          # regenerate table/figures without retraining
        report(load_json(f"{LOGS}/ablation.json")["results"])
        sys.exit(0)
    args = parse_args(__doc__)
    cfg = settings(args)
    data = get_data(args)
    results = {}
    for name in ORDER:
        print(f"\n== {LABELS[name]} ==")
        lr, scores, edge = tune_lr(name, HIDDEN, data, cfg["epochs"], cfg["lr_grid"])
        hists = []
        for seed in cfg["seeds"]:
            _, h = run(name, HIDDEN, lr, seed, data, cfg["epochs"])
            hists.append(h)
        results[name] = dict(lr=lr, lr_scores=scores, lr_at_grid_edge=edge,
                             histories=hists, summary=summarise(hists))
        s = results[name]["summary"]
        print(f"  selected lr={lr}: val acc {s['val_acc'][0]*100:.2f} ± {s['val_acc'][1]*100:.2f}")
    save_json(dict(epochs=cfg["epochs"], batch_size=BATCH_SIZE, hidden=HIDDEN, seeds=cfg["seeds"],
                   results=results), f"{LOGS}/ablation.json")

    report(results)
