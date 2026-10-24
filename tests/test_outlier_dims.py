import numpy as np

from gbleed import activations as acts
from gbleed import outlier_dims as od


def test_zeroes_only_dominant_dims(monkeypatch):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2, 20))
    X[:, 1, 3] *= 30  # layer 1: one dimension with most of the variance
    monkeypatch.setattr(acts, "load", lambda m, n: (X, {"words": []}))
    real = acts.load
    dims = od.outlier_dims("google/gemma-x")
    assert list(dims[0]) == [] and list(dims[1]) == [3]
    with od.dropped("google/gemma-x"):
        Y, _ = acts.load("google/gemma-x", "anything")
    assert np.all(Y[:, 1, 3] == 0) and np.array_equal(Y[:, 0], X[:, 0])
    assert np.array_equal(np.delete(Y[:, 1], 3, 1), np.delete(X[:, 1], 3, 1))
    assert acts.load is real  # restored after the block
