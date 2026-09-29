"""What do the hidden units learn? (extends Question 2 of the original project)

Analysed model: improved formulation (ReLU + softmax/CE, He init), 784-64-10 — the same
hidden size as the original analysis — trained with its validation-selected learning rate.
All statistics are computed on the validation set (the test set is not used here).
The faithful re-run of the original sigmoid model (reproduce_original.py) is used to
quantify the saturation problem of the original top-activation ranking.
"""
import os

import numpy as np

from experiments.common import get_data, parse_args, run
from src.analysis import (ablation_importance, class_mean_activation, count_saturated,
                          selectivity_index, spearman, top_activating, weight_variance_scores)
from src.model import MLP
from src.utils import FIGURES, LOGS, MODELS, TABLES, load_json, save_json
from src.visualize import (plot_class_heatmap, plot_scatter, plot_top_activating,
                           plot_weight_maps)

if __name__ == "__main__":
    args = parse_args(__doc__)
    data = get_data(args)
    X_val, y_val = data[2], data[3]
    arch = load_json(f"{LOGS}/architecture_comparison.json")
    lr = arch["results"]["relu_ce_he_h64"]["lr"]
    epochs = 2 if args.quick else arch["epochs"]
    model, hist = run("relu_ce_he", 64, lr, 0, data, epochs)
    model.save(f"{MODELS}/analysis_relu_h64_seed0.npz")
    W1 = model.dense[0].W
    print(f"Analysed model val acc: {hist['val_acc'][-1]:.4f}")

    # 1. weight maps of all 64 units
    plot_weight_maps(W1, range(64), f"{FIGURES}/weight_maps_all.png",
                     "Input-weight maps of all 64 hidden units (red +, blue −)")

    # 2. importance by ablation, and the original variance heuristic
    imp = ablation_importance(model, X_val, y_val)
    var = weight_variance_scores(model)
    rho = spearman(var, imp)
    order = np.argsort(-imp)
    top_units = order[:8]
    plot_weight_maps(W1, top_units, f"{FIGURES}/weight_maps_most_important.png",
                     "8 most important hidden units (ablation Δ val. acc.)", cols=8,
                     subtitles=[f"n{u}: −{imp[u]*100:.2f}pp" for u in top_units])
    plot_scatter(var, imp * 100, f"{FIGURES}/variance_vs_importance.png",
                 "Variance of input weights (original heuristic)",
                 "Accuracy drop when unit is removed (pp)",
                 f"Is weight variance a good proxy for importance? Spearman ρ = {rho:.2f}",
                 annotate=order[:5])

    # 3. top-activating validation images, ranked by pre-activation
    idx = np.zeros((8, 8), dtype=int); vals = np.zeros((8, 8))
    for r, u in enumerate(top_units):
        idx[r], vals[r] = top_activating(model, X_val, u, k=8)
    plot_top_activating(W1, X_val, top_units, idx, vals, y_val,
                        f"{FIGURES}/top_activating_samples.png",
                        "Most important units: weights and top-8 validation images (ranked by z)")

    # 4. class selectivity
    M = class_mean_activation(model, X_val, y_val)
    sel = selectivity_index(M)
    plot_class_heatmap(M, order, f"{FIGURES}/class_selectivity.png",
                       "Mean activation per digit class (units sorted by importance)")
    dead = int((M.max(axis=1) == 0).sum())

    # 5. saturation in the original sigmoid model (why ranking by a was unreliable)
    sat = None
    path = f"{MODELS}/sigmoid_mse_h64_seed0.npz"
    if os.path.exists(path):
        orig = MLP.load(path)
        sat_counts = [count_saturated(orig, X_val, u) for u in range(6)]
        rho_orig = spearman(weight_variance_scores(orig), ablation_importance(orig, X_val, y_val))
        sat = dict(units_0_to_5_samples_displayed_as_a_1000=sat_counts,
                   spearman_variance_vs_importance=rho_orig)
        print("Original model - samples with a >= 0.9995 for units 0-5:", sat_counts)
        print(f"Original model - Spearman(variance, importance) = {rho_orig:.2f}")

    summary = dict(
        analysed_model=dict(config="relu_ce_he", hidden=64, lr=lr, val_acc=hist["val_acc"][-1]),
        spearman_variance_vs_importance=rho,
        importance_pp=dict(max=float(imp.max() * 100), median=float(np.median(imp) * 100),
                           min=float(imp.min() * 100)),
        most_important_units=[dict(unit=int(u), drop_pp=float(imp[u] * 100),
                                   preferred_class=int(M[u].argmax()),
                                   selectivity=float(sel[u])) for u in top_units],
        selectivity=dict(mean=float(sel.mean()), median=float(np.median(sel))),
        dead_units_on_validation=dead,
        original_model_saturation=sat)
    save_json(summary, f"{TABLES}/feature_analysis.json")
    print(summary)
