import numpy as np

from src.losses import softmax


def test_rows_sum_to_one_and_positive():
    z = np.random.default_rng(0).standard_normal((50, 10)) * 5
    p = softmax(z)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p > 0).all()


def test_numerically_stable_for_large_logits():
    z = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1000.0, -1000.0]])
    p = softmax(z)
    assert np.isfinite(p).all()
    assert np.allclose(p[1], 1 / 3)
    assert np.allclose(p[0], softmax(np.array([[0.0, 1.0, 2.0]]))[0])


def test_shift_invariance():
    z = np.random.default_rng(1).standard_normal((4, 10))
    assert np.allclose(softmax(z), softmax(z + 123.4))
