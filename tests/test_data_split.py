import numpy as np

from src.data import one_hot, train_val_split


def test_split_is_disjoint_complete_and_reproducible():
    X = np.arange(100).reshape(100, 1)
    y = np.arange(100) % 10
    Xtr, ytr, Xva, yva = train_val_split(X, y, n_val=20, seed=0)
    assert len(Xva) == 20 and len(Xtr) == 80
    assert set(Xtr[:, 0]).isdisjoint(Xva[:, 0])
    assert set(Xtr[:, 0]) | set(Xva[:, 0]) == set(range(100))
    assert np.array_equal(Xva, train_val_split(X, y, n_val=20, seed=0)[2])


def test_one_hot():
    Y = one_hot(np.array([0, 3, 9]), 10)
    assert Y.shape == (3, 10) and np.array_equal(Y.argmax(1), [0, 3, 9]) and Y.sum() == 3
