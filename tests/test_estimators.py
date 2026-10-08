"""Synthetic checks for the stratified / split-half estimators (known ground truth)."""

import numpy as np

from gbleed.estimators import (
    boot_within,
    nuisance,
    plain_geometry,
    probe_strat,
    residualise,
    shuffle_within,
    split_half_geometry,
)


def _data(rng, n_m, n_f, n_n, d=400, len_m=1.0, len_f=1.0, cos=-0.2, noise=1.0, cells=3):
    """Class means at chosen lengths/cosine from neuter, isotropic noise, cell offsets."""
    u = rng.normal(size=d)
    u /= np.linalg.norm(u)
    w = rng.normal(size=d)
    w -= (w @ u) * u
    w /= np.linalg.norm(w)
    vm = len_m * u
    vf = len_f * (cos * u + np.sqrt(1 - cos**2) * w)
    labels = np.array(["m"] * n_m + ["f"] * n_f + ["n"] * n_n)
    cell = rng.integers(0, cells, len(labels))
    offsets = rng.normal(scale=5.0, size=(cells, d))  # big cell effects, shared by genders
    X = rng.normal(scale=noise, size=(len(labels), d)) + offsets[cell]
    X[labels == "m"] += vm
    X[labels == "f"] += vf
    return X, labels, cell


def test_split_half_length_unbiased_where_plain_is_not():
    rng = np.random.default_rng(0)
    plain_m, plain_f, sh_m, sh_f = [], [], [], []
    for _ in range(20):
        X, lab, cell = _data(rng, n_m=100, n_f=25, n_n=30, len_m=1.0, len_f=1.0)
        p = plain_geometry(X, lab, cell, [], ["m", "f"])
        s = split_half_geometry(X, lab, cell, [], ["m", "f"], "n", rng, n_splits=10)
        plain_m.append(p["len2_m"]), plain_f.append(p["len2_f"])
        sh_m.append(s["len2_m"]), sh_f.append(s["len2_f"])
    # equal true lengths (1.0): plain makes the smaller group look longer; split-half doesn't
    assert np.mean(plain_f) - np.mean(plain_m) > 3.0
    assert abs(np.mean(sh_f) - 1.0) < 0.5 and abs(np.mean(sh_m) - 1.0) < 0.5
    assert abs(np.mean(sh_f) - np.mean(sh_m)) < 0.5


def test_split_half_cosine_recovers_truth_and_plain_is_pushed_up():
    rng = np.random.default_rng(1)
    X, lab, cell = _data(rng, n_m=200, n_f=200, n_n=200, len_m=2.0, len_f=2.0, cos=-0.5)
    s = split_half_geometry(X, lab, cell, [], ["m", "f"], "n", rng, n_splits=20)
    p = plain_geometry(X, lab, cell, [], ["m", "f"])
    assert abs(s["cos"] - (-0.5)) < 0.15
    assert p["cos"] > s["cos"]  # shared neuter noise pushes the plug-in cosine up
    # reliability = cos of two noisy half-estimates: here true len2 4 vs ~8 noise per half, so ~0.33
    assert 0.2 < s["rel_m"] < 0.5 and 0.2 < s["rel_f"] < 0.5


def test_shuffle_keeps_cell_counts_and_null_lengths_near_zero():
    rng = np.random.default_rng(2)
    null_f, real_f = [], []
    for _ in range(30):
        X, lab, cell = _data(rng, n_m=60, n_f=30, n_n=30, len_m=1.5, len_f=1.5)
        sh = shuffle_within(lab, cell, rng)
        for c in np.unique(cell):
            assert sorted(sh[cell == c]) == sorted(lab[cell == c])
        null_f.append(split_half_geometry(X, sh, cell, [], ["m", "f"], "n", rng, 5)["len2_f"])
        real_f.append(split_half_geometry(X, lab, cell, [], ["m", "f"], "n", rng, 5)["len2_f"])
    # unbiased: the shuffled (no real difference) lengths average ~0, the real ones ~1.5^2
    assert abs(np.mean(null_f)) < 0.8
    assert abs(np.mean(real_f) - 2.25) < 0.8


def test_residualise_removes_cell_effects_and_probe_ignores_cells():
    rng = np.random.default_rng(3)
    n, d = 300, 50
    cell = rng.integers(0, 4, n)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, d))
    X[:, 0] += 10 * cell  # a cell feature (e.g. the ending)
    X[:, 1] += 1.0 * y  # the real class feature
    R = residualise(X, nuisance(cell))
    for c in range(4):
        assert abs(R[cell == c, 0].mean()) < 1e-8
    w = probe_strat(X, y, nuisance(cell))
    assert abs(w[1]) > 5 * abs(w[0])


def test_boot_within_keeps_strata_sizes():
    rng = np.random.default_rng(4)
    cell = np.array([0, 0, 0, 1, 1, 2])
    lab = np.array(["m", "f", "m", "f", "f", "m"])
    idx = boot_within(cell, lab, rng)
    assert len(idx) == len(cell)
    for c, g in {(0, "m"), (0, "f"), (1, "f"), (2, "m")}:
        assert ((cell[idx] == c) & (lab[idx] == g)).sum() == ((cell == c) & (lab == g)).sum()


def test_gram_shortcut_matches_direct_computation():
    """The Gram-matrix shortcut must equal fitting each half explicitly in activation space."""
    from gbleed.estimators import _halves, class_betas

    rng = np.random.default_rng(5)
    X, lab, cell = _data(rng, n_m=40, n_f=20, n_n=25, d=60, len_m=3.0, len_f=3.0)
    cov = [rng.normal(size=len(lab))]
    s = split_half_geometry(X, lab, cell, cov, ["m", "f"], "n", np.random.default_rng(9), 1)
    a = _halves(cell, lab, np.random.default_rng(9))
    G = np.column_stack([(lab == c).astype(float) for c in ("m", "f")])
    vA = class_betas(X[a], G[a], nuisance(cell[a], [cov[0][a]]))
    vB = class_betas(X[~a], G[~a], nuisance(cell[~a], [cov[0][~a]]))
    assert np.isclose(s["len2_m"], vA[0] @ vB[0]) and np.isclose(s["len2_f"], vA[1] @ vB[1])
    assert np.isclose(
        s["cos"], (vA[0] @ vB[1] + vB[0] @ vA[1]) / 2 / np.sqrt(s["len2_m"] * s["len2_f"])
    )
