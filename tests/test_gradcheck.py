import numpy as np
import pytest

from src.data import one_hot
from src.gradcheck import gradient_check
from src.model import MLP

CASES = [
    # (activation, loss, init, layer sizes)
    ("sigmoid", "sigmoid_mse", "normal_0.01", [12, 7, 5]),   # original formulation / architecture
    ("sigmoid", "sigmoid_mse", "xavier", [12, 8, 6, 5]),
    ("sigmoid", "softmax_ce", "xavier", [12, 7, 5]),
    ("sigmoid", "softmax_ce", "xavier", [12, 8, 6, 5]),
    ("relu", "softmax_ce", "he", [12, 7, 5]),
    ("relu", "softmax_ce", "he", [12, 8, 6, 5]),
]
# Note: a *two*-hidden-layer sigmoid net with N(0, 0.01^2) init is not tested because its
# first-layer gradients vanish (~1e-10), so the relative error is dominated by
# floating-point noise in the finite difference rather than by any backprop error.


@pytest.mark.parametrize("act,loss,init,sizes", CASES)
def test_backprop_matches_numerical_gradient(act, loss, init, sizes):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((6, sizes[0]))
    Y = one_hot(rng.integers(0, sizes[-1], 6), sizes[-1], dtype=np.float64)
    model = MLP(sizes, act, loss, init, seed=0, dtype=np.float64)
    errors = gradient_check(model, X, Y)
    assert max(errors.values()) < 1e-5, errors
