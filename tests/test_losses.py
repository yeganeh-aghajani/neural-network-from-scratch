import numpy as np

from src.data import one_hot
from src.losses import SigmoidMSE, SoftmaxCrossEntropy
from src.model import MLP
from src.training import sgd_step


def test_cross_entropy_uniform_prediction_equals_log_k():
    loss = SoftmaxCrossEntropy().forward(np.zeros((5, 10)), one_hot(np.arange(5), 10))
    assert np.isclose(loss, np.log(10))


def test_cross_entropy_confident_correct_is_near_zero_and_wrong_is_large():
    Y = one_hot(np.array([2]), 10)
    good = np.full((1, 10), -20.0); good[0, 2] = 20.0
    bad = np.full((1, 10), -20.0); bad[0, 5] = 20.0
    ce = SoftmaxCrossEntropy()
    assert ce.forward(good, Y) < 1e-10
    assert ce.forward(bad, Y) > 30


def test_mse_zero_when_output_matches_target():
    Y = one_hot(np.array([1, 4]), 10)
    z = np.where(Y == 1, 50.0, -50.0)          # sigmoid(z) ~ one-hot
    assert SigmoidMSE().forward(z, Y) < 1e-12


def test_losses_non_negative():
    rng = np.random.default_rng(0)
    z, Y = rng.standard_normal((20, 10)), one_hot(rng.integers(0, 10, 20), 10)
    assert SoftmaxCrossEntropy().forward(z, Y) >= 0
    assert SigmoidMSE().forward(z, Y) >= 0


def test_gradient_step_decreases_loss():
    rng = np.random.default_rng(0)
    X = rng.random((64, 784)).astype(np.float32)
    Y = one_hot(rng.integers(0, 10, 64))
    for act, loss, init in [("relu", "softmax_ce", "he"), ("sigmoid", "sigmoid_mse", "xavier")]:
        model = MLP([784, 32, 10], act, loss, init, seed=0)
        before = model.compute_loss(X, Y)
        model.backward()
        sgd_step(model, lr=0.05)
        assert model.compute_loss(X, Y) < before
