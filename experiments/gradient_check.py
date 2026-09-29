"""Verify backpropagation against central finite differences (float64, small networks)."""
import numpy as np

from src.data import one_hot
from src.gradcheck import gradient_check
from src.model import MLP
from src.utils import TABLES, markdown_table, save_json

CASES = [
    ("sigmoid", "sigmoid_mse", "normal_0.01", [20, 10, 10]),
    ("sigmoid", "softmax_ce", "xavier", [20, 10, 10]),
    ("relu", "softmax_ce", "he", [20, 10, 10]),
    ("relu", "softmax_ce", "he", [20, 12, 8, 10]),
]

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows, out = [], []
    for act, loss, init, sizes in CASES:
        X = rng.random((8, sizes[0]))
        Y = one_hot(rng.integers(0, 10, 8), 10, dtype=np.float64)
        model = MLP(sizes, act, loss, init, seed=0, dtype=np.float64)
        err = gradient_check(model, X, Y, eps=1e-5)
        worst = max(err.values())
        arch = "-".join(map(str, sizes))
        print(f"{act:8s} {loss:12s} {init:12s} {arch:12s} max rel. error = {worst:.2e}")
        rows.append([act, loss, init, arch, f"{worst:.1e}", "pass" if worst < 1e-5 else "FAIL"])
        out.append(dict(activation=act, loss=loss, init=init, sizes=sizes, errors=err))
    table = markdown_table(["Hidden act.", "Loss", "Init", "Layers", "Max rel. error", "Result"], rows)
    open(f"{TABLES}/gradient_check.md", "w").write(table + "\n")
    save_json(out, f"{TABLES}/gradient_check.json")
    print(table)
