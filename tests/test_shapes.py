import numpy as np
import pytest

from src.data import one_hot
from src.model import MLP


@pytest.mark.parametrize("sizes", [[784, 16, 10], [784, 64, 10], [784, 128, 10], [20, 8, 6, 5]])
@pytest.mark.parametrize("act,loss", [("sigmoid", "sigmoid_mse"), ("relu", "softmax_ce")])
def test_forward_backward_shapes(sizes, act, loss):
    rng = np.random.default_rng(0)
    X = rng.random((32, sizes[0]), dtype=np.float32)
    y = rng.integers(0, sizes[-1], 32)
    model = MLP(sizes, act, loss, "he", seed=0)
    logits = model.forward(X)
    assert logits.shape == (32, sizes[-1])
    assert [h.shape for h in model.hidden] == [(32, s) for s in sizes[1:-1]]
    model.compute_loss(X, one_hot(y, sizes[-1]))
    model.backward()
    for (name, p, g) in model.parameters():
        assert p.shape == g.shape, name
    assert model.predict(X).shape == (32,)
    assert model.predict_proba(X).shape == (32, sizes[-1])


def test_parameter_count():
    model = MLP([784, 64, 10])
    assert model.num_parameters() == 784 * 64 + 64 + 64 * 10 + 10


def test_single_sample_batch():
    """Batch size 1 must work: the original project trained sample by sample."""
    model = MLP([784, 64, 10], "sigmoid", "sigmoid_mse", "normal_0.01")
    x = np.random.default_rng(0).random((1, 784), dtype=np.float32)
    model.compute_loss(x, one_hot(np.array([3])))
    model.backward()
    assert model.dense[0].dW.shape == (784, 64)


def test_seed_reproducibility():
    a, b = MLP([784, 64, 10], seed=7), MLP([784, 64, 10], seed=7)
    assert all(np.array_equal(p, q) for (_, p, _), (_, q, _) in zip(a.parameters(), b.parameters()))
