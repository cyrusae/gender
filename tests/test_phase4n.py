import numpy as np

from gbleed import phase4, phase4n


def _d(n, extra, rng, prefix):
    return {"y": rng.integers(0, 2, n), "cells": np.array([f"{prefix}{i % 3}" for i in range(n)]),
            "covs": [rng.normal(size=n) for _ in range(4 + extra)]}  # fmt: skip


def test_pooled_n_matches_two_language_pooled():
    rng = np.random.default_rng(0)
    es, de = _d(20, 3, rng, "e"), _d(15, 0, rng, "d")
    a, b = phase4.pooled(es, de), phase4n.pooled_n({"es": es, "de": de})
    assert (a["y"] == b["y"]).all() and (a["cells"] == b["cells"]).all()
    assert len(a["covs"]) == len(b["covs"])
    assert all(np.allclose(x, y) for x, y in zip(a["covs"], b["covs"], strict=True))


def test_pooled_n_three_languages_pads_extras():
    rng = np.random.default_rng(1)
    ds = {"es": _d(10, 2, rng, "e"), "de": _d(8, 0, rng, "d"), "ru": _d(6, 1, rng, "r")}
    p = phase4n.pooled_n(ds)
    assert len(p["y"]) == 24 and len(p["covs"]) == 4 + 2 + 1
    assert (p["covs"][-1][:18] == 0).all()  # Russian-only covariate is zero for es, de
    assert list(p["offsets"]) == [0, 10, 18]
