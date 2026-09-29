"""Faithful re-run of the original course configuration under the corrected evaluation.

Original setup (kept exactly): 784-64-10, sigmoid hidden + sigmoid output, MSE loss,
N(0, 0.01^2) init, per-sample SGD (batch size 1), lr = 0.1, 10 epochs.
What changes: fixed seeds, per-epoch shuffling, a held-out validation split, and
accuracy on the full 10k validation set instead of the first 1000 test images.
"""
from experiments.common import get_data, parse_args, run, summarise
from src.utils import FIGURES, LOGS, MODELS, TABLES, markdown_table, mean_curves, save_json
from src.visualize import plot_curves

if __name__ == "__main__":
    args = parse_args(__doc__)
    epochs, seeds = (2, [0]) if args.quick else (10, [0, 1, 2])
    data = get_data(args)
    hists = []
    for seed in seeds:
        print(f"seed {seed}")
        model, h = run("original", 64, 0.1, seed, data, epochs, batch_size=1, verbose=True)
        hists.append(h)
        if seed == 0:
            model.save(f"{MODELS}/sigmoid_mse_h64_seed0.npz")
    s = summarise(hists)
    per_seed = [round(h["val_acc"][-1] * 100, 2) for h in hists]
    save_json(dict(histories=hists, summary=s, per_seed_val_acc=per_seed),
              f"{LOGS}/reproduce_original.json")
    plot_curves({"Original config (batch 1, lr 0.1)": mean_curves(hists)},
                f"{FIGURES}/original_rerun_curves.png",
                "Original sigmoid+MSE configuration, re-run with corrected evaluation")
    table = markdown_table(
        ["Setting", "Train acc (%)", "Val acc (%)", "Val acc per seed (%)"],
        [["784-64-10, sigmoid+MSE, batch 1, lr 0.1, 10 epochs",
          f"{s['train_acc'][0]*100:.2f} ± {s['train_acc'][1]*100:.2f}",
          f"{s['val_acc'][0]*100:.2f} ± {s['val_acc'][1]*100:.2f}",
          ", ".join(map(str, per_seed))]])
    open(f"{TABLES}/original_rerun.md", "w").write(table + "\n")
    print(table)
