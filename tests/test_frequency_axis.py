from pathlib import Path

import numpy as np
import pytest

from gbleed import frequency_axis as fa
from gbleed import paths


@pytest.mark.skipif(
    not Path(paths.kaikki("kaikki.org-dictionary-German.jsonl")).exists(),
    reason="needs the Wiktionary German dump (not bundled to pods)",
)
def test_complexity_rule_examples():
    why = fa.complexity_rule()
    assert "numeral+er" in why("Vierer")
    assert "noun+verb+er" in why("Wasserkocher")
    assert why("Gabel") == ""
    assert why("Luft") == ""


def test_matched_pairs_respect_caliper_cells_and_tokens():
    rng = np.random.default_rng(0)
    n = 60
    d = {"y": rng.integers(0, 2, n), "cells": rng.choice(["a", "b"], n),
         "covs": [np.zeros(n), np.zeros(n), rng.normal(3, 1, n), rng.integers(1, 3, n)]}  # fmt: skip
    pr = fa.matched_pairs(d)
    used = [i for p in pr for i in p]
    assert len(used) == len(set(used))
    for i, j in pr:
        assert d["y"][i] == 1 and d["y"][j] == 0 and d["cells"][i] == d["cells"][j]
        assert d["covs"][3][i] == d["covs"][3][j]
        assert abs(d["covs"][2][i] - d["covs"][2][j]) <= fa.CALIPER


def test_projection_removes_axis():
    from gbleed import activations as acts

    X = np.random.default_rng(1).normal(size=(5, 3, 8)).astype(np.float32)
    Q = np.linalg.qr(np.random.default_rng(2).normal(size=(8, 3)))[0].T[None].repeat(3, 0)[:, :2]
    orig = acts.load
    acts.load = lambda m, n: (X.copy(), {})
    try:
        with fa.projected(Q):
            Y, _ = acts.load("m", "x")
    finally:
        acts.load = orig
    assert np.abs(np.einsum("nld,lrd->nlr", Y, Q)).max() < 1e-5
