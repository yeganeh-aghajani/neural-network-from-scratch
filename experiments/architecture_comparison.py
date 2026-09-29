"""Hidden-layer size comparison (16 / 64 / 128 units) for two formulations, on validation data.

  * original formulation: sigmoid + MSE, N(0, 0.01^2) init   (the original Question 3, redone)
  * improved formulation: ReLU + softmax/cross-entropy, He init

The configuration with the best mean validation accuracy is written to
results/selected_config.json and is the only model evaluated on the test set.
"""
from experiments.common import BATCH_SIZE, LABELS, get_data, parse_args, run, settings, summarise, tune_lr
from src.utils import FIGURES, LOGS, RESULTS, TABLES, markdown_table, mean_curves, save_json
from src.visualize import plot_curves

FAMILIES = ["original", "relu_ce_he"]
SIZES = [16, 64, 128]

if __name__ == "__main__":
    args = parse_args(__doc__)
    cfg = settings(args)
    data = get_data(args)
    results = {}
    for fam in FAMILIES:
        for h in SIZES:
            key = f"{fam}_h{h}"
            print(f"\n== {LABELS[fam]}, hidden = {h} ==")
            lr, scores, edge = tune_lr(fam, h, data, cfg["epochs"], cfg["lr_grid"])
            hists = [run(fam, h, lr, s, data, cfg["epochs"])[1] for s in cfg["seeds"]]
            n_params = 784 * h + h + h * 10 + 10
            results[key] = dict(family=fam, hidden=h, lr=lr, lr_scores=scores,
                                lr_at_grid_edge=edge, params=n_params,
                                histories=hists, summary=summarise(hists))
            s = results[key]["summary"]
            print(f"  lr={lr}: train {s['train_acc'][0]*100:.2f}  val {s['val_acc'][0]*100:.2f}")
    save_json(dict(epochs=cfg["epochs"], batch_size=BATCH_SIZE, seeds=cfg["seeds"],
                   results=results), f"{LOGS}/architecture_comparison.json")

    rows = []
    for key, r in results.items():
        s = r["summary"]
        gap = (s["train_acc"][0] - s["val_acc"][0]) * 100
        rows.append([LABELS[r["family"]].split(",")[0], r["hidden"], f"{r['params']:,}",
                     f"{r['lr']}" + (" *" if r["lr_at_grid_edge"] else ""),
                     f"{s['train_acc'][0]*100:.2f} ± {s['train_acc'][1]*100:.2f}",
                     f"{s['val_acc'][0]*100:.2f} ± {s['val_acc'][1]*100:.2f}", f"{gap:.2f}"])
    table = markdown_table(["Formulation", "Hidden", "Params", "LR", "Train acc (%)",
                            "Val acc (%)", "Train−val gap (pp)"], rows)
    open(f"{TABLES}/architecture_comparison.md", "w").write(table + "\n")
    print("\n" + table)

    for fam in FAMILIES:
        plot_curves({f"h = {h}": mean_curves(results[f"{fam}_h{h}"]["histories"]) for h in SIZES},
                    f"{FIGURES}/architecture_{fam}.png",
                    f"Hidden-layer size — {LABELS[fam]} (mean over seeds)")

    best = max(results, key=lambda k: results[k]["summary"]["val_acc"][0])
    b = results[best]
    selected = dict(key=best, family=b["family"], hidden=b["hidden"], lr=b["lr"],
                    epochs=cfg["epochs"], batch_size=BATCH_SIZE,
                    val_acc_mean=b["summary"]["val_acc"][0], val_acc_std=b["summary"]["val_acc"][1],
                    selected_by="highest mean final-epoch validation accuracy")
    save_json(selected, f"{RESULTS}/selected_config.json")
    print(f"\nSelected for final test evaluation: {selected}")
