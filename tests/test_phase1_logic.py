"""Guards against the Phase 1 circularity bug: a probe trained on the eraser's own fit data
scores chance by construction (LEACE equalises class means -> optimal logistic weights are 0),
so erasure must be tested with probes trained on data the eraser never saw."""

import numpy as np

from gbleed.phase1 import _apply, _cv, _fit


def _two_types(rng, n=200, d=20):
    """Type A encodes the label on axis 0; type B on a mix of axes 0 and 1."""
    y = np.r_[np.zeros(n), np.ones(n)]
    a = rng.standard_normal((2 * n, d))
    a[:, 0] += 3 * (y - 0.5)
    b = rng.standard_normal((2 * n, d))
    b[:, 0] += 2 * (y - 0.5)
    b[:, 1] += 2 * (y - 0.5)
    return a, b, y


def test_probe_on_fit_data_is_chance_by_construction():
    """Probe trained on the eraser's fit data learns nothing: constant scores."""
    from gbleed.phase1 import _probe

    rng = np.random.default_rng(0)
    a, _, y = _two_types(rng)
    xa = _apply(_fit(a, y.astype(np.int64)), a)
    s = _probe().fit(xa, y).decision_function(xa)
    assert np.ptp(s) < 1e-6


def test_crossvalidating_insample_erasure_goes_below_chance():
    """The artifact behind the original 'verbs after 0.15-0.20': erase on all data, then
    cross-validate -> each training fold's mean difference opposes its test fold's."""
    rng = np.random.default_rng(0)
    a, _, y = _two_types(rng)
    _, auc = _cv(_apply(_fit(a, y.astype(np.int64)), a), y, seed=0)
    assert auc < 0.4


def test_unseen_data_reveals_leftover_signal():
    rng = np.random.default_rng(0)
    a, b, y = _two_types(rng)
    e = _fit(a, y.astype(np.int64))  # erases axis-0 signal of type A only
    _, auc = _cv(_apply(e, b), y, seed=0)  # type B's axis-1 signal survives
    assert auc > 0.8


def test_adjusted_difference_of_means_removes_confound():
    """Gender on axis 0; concreteness on axis 1, and correlated with gender. Plain difference of
    means leaks onto axis 1; the adjusted version doesn't."""
    from gbleed.phase2 import _adom, _dom

    rng = np.random.default_rng(1)
    n = 400
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    conc = rng.normal(0, 1, 2 * n) - 0.8 * y  # feminine nouns less concrete
    X = rng.normal(0, 0.1, (2 * n, 5))
    X[:, 0] += 1.0 * y
    X[:, 1] += 1.0 * conc
    plain, adj = _dom(X, y), _adom(X, y, [conc])
    assert abs(plain[1]) > 0.5  # confounded
    assert abs(adj[1]) < 0.1 and abs(adj[0] - 1.0) < 0.1  # recovers the gender effect only


def test_readout_identity_retrieval():
    """Same word in two contexts = same vector plus noise -> retrieval succeeds; shuffled -> chance."""
    from gbleed.readout import _identity

    rng = np.random.default_rng(0)
    A = rng.standard_normal((30, 16))
    B = A + 0.05 * rng.standard_normal((30, 16))
    groups = [np.arange(0, 10), np.arange(10, 20), np.arange(20, 30)]
    assert _identity(A, B, groups) == 1.0
    assert _identity(A, rng.standard_normal((30, 16)), groups) < 0.4
