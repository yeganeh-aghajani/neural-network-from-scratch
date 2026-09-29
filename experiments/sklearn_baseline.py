"""OPTIONAL, SECONDARY sanity check: the same architecture trained with scikit-learn.

This is NOT part of the from-scratch implementation. It uses the selected hidden size,
learning rate, batch size and epoch count, plain SGD without momentum or L2, and reports
VALIDATION accuracy only. Initialisation differs (scikit-learn uses Glorot-uniform).
Requires: pip install scikit-learn
"""
import warnings

import numpy as np

from experiments.common import get_data, parse_args
from src.utils import RESULTS, TABLES, load_json, save_json

if __name__ == "__main__":
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier

    args = parse_args(__doc__)
    sel = load_json(f"{RESULTS}/selected_config.json")
    X_tr, y_tr, X_val, y_val = get_data(args)
    epochs = 2 if args.quick else sel["epochs"]
    accs = []
    for seed in ([0] if args.quick else [0, 1, 2]):
        clf = MLPClassifier(hidden_layer_sizes=(sel["hidden"],), activation="relu", solver="sgd",
                            learning_rate_init=sel["lr"], batch_size=sel["batch_size"],
                            momentum=0.0, nesterovs_momentum=False, alpha=0.0, max_iter=epochs,
                            shuffle=True, early_stopping=False, random_state=seed, tol=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(X_tr, y_tr)
        accs.append(float(clf.score(X_val, y_val)))
        print(f"seed {seed}: val acc {accs[-1]:.4f}")
    res = dict(hidden=sel["hidden"], lr=sel["lr"], epochs=epochs, val_acc_per_seed=accs,
               val_acc_mean=float(np.mean(accs)),
               val_acc_std=float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0)
    save_json(res, f"{TABLES}/sklearn_baseline.json")
    print(res)
