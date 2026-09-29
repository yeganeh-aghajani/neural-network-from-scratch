"""Train the selected configuration and evaluate it ONCE on the official 10k test set.

Reads results/selected_config.json written by architecture_comparison.py.
"""
import numpy as np

from experiments.common import LABELS, parse_args, run
from src.data import load_splits
from src.metrics import confusion_matrix, per_class_report
from src.training import evaluate
from src.utils import FIGURES, LOGS, MODELS, RESULTS, TABLES, load_json, markdown_table, mean_curves, save_json
from src.visualize import plot_confusion, plot_curves, plot_image_grid

if __name__ == "__main__":
    args = parse_args(__doc__)
    sel = load_json(f"{RESULTS}/selected_config.json")
    epochs, seeds = (2, [0]) if args.quick else (sel["epochs"], [0, 1, 2])
    d = load_splits(args.data_dir)
    data = (d["X_train"], d["y_train"], d["X_val"], d["y_val"])
    print(f"Selected: {LABELS[sel['family']]}, hidden={sel['hidden']}, lr={sel['lr']}")

    runs = []
    for seed in seeds:
        model, hist = run(sel["family"], sel["hidden"], sel["lr"], seed, data, epochs)
        test_loss, test_acc = evaluate(model, d["X_test"], d["y_test"])
        print(f"  seed {seed}: val acc {hist['val_acc'][-1]:.4f} | test acc {test_acc:.4f}")
        runs.append(dict(seed=seed, history=hist, test_loss=test_loss, test_acc=test_acc))
        if seed == 0:
            model.save(f"{MODELS}/final_model_seed0.npz")
            final_model, final_hist = model, hist

    accs = np.array([r["test_acc"] for r in runs])
    y_pred = final_model.predict(d["X_test"])
    cm = confusion_matrix(d["y_test"], y_pred)
    per_class, macro = per_class_report(cm)
    summary = dict(selected=sel, test_acc_mean=float(accs.mean()),
                   test_acc_std=float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
                   test_acc_per_seed=accs.tolist(), seed0_macro=macro,
                   seed0_errors=int((y_pred != d["y_test"]).sum()),
                   seed0_confusion=cm.tolist(), seed0_per_class=per_class)
    save_json(dict(summary=summary, runs=runs), f"{LOGS}/final_evaluation.json")
    save_json(summary, f"{TABLES}/final_test_results.json")

    rows = [[r["cls"], f"{r['precision']*100:.2f}", f"{r['recall']*100:.2f}", f"{r['f1']*100:.2f}",
             r["support"]] for r in per_class]
    rows.append(["macro avg", f"{macro['precision']*100:.2f}", f"{macro['recall']*100:.2f}",
                 f"{macro['f1']*100:.2f}", int(cm.sum())])
    table = markdown_table(["Class", "Precision (%)", "Recall (%)", "F1 (%)", "Support"], rows)
    open(f"{TABLES}/per_class_test.md", "w").write(table + "\n")

    off = cm.copy(); np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:5], cm.shape))[0]
    confusions = [dict(true=int(i), pred=int(j), count=int(cm[i, j])) for i, j in top]
    save_json(confusions, f"{TABLES}/top_confusions.json")

    print(f"\nTest accuracy: {accs.mean()*100:.2f} ± {summary['test_acc_std']*100:.2f} % "
          f"(per seed: {', '.join(f'{a*100:.2f}' for a in accs)})")
    print(table)
    print("Most frequent confusions (true -> predicted):", confusions)

    title = f"{LABELS[sel['family']]}, 784-{sel['hidden']}-10"
    plot_confusion(cm, f"{FIGURES}/confusion_matrix_test.png", f"Test confusion matrix — {title}")
    plot_curves({"selected model": mean_curves([r["history"] for r in runs])},
                f"{FIGURES}/final_model_curves.png", f"Training curves — {title}")
    wrong = np.where(y_pred != d["y_test"])[0][:24]
    probs = final_model.predict_proba(d["X_test"][wrong])
    plot_image_grid(d["X_test"][wrong],
                    [f"T{d['y_test'][i]} P{y_pred[i]} ({p.max():.2f})" for i, p in zip(wrong, probs)],
                    f"{FIGURES}/misclassified_test.png",
                    "First 24 misclassified test images (T = true, P = predicted, confidence)")
    first = np.arange(16)
    plot_image_grid(d["X_test"][first], [f"T{d['y_test'][i]} P{y_pred[i]}" for i in first],
                    f"{FIGURES}/test_predictions.png", "Test samples: true (T) vs predicted (P)")
