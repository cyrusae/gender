import numpy as np
import pandas as pd

from gbleed import rarity_heldout as rh


def test_partial_spearman_controls_ending_and_tokens():
    rng = np.random.default_rng(0)
    n = 300
    ending = rng.choice(["a", "b", "c"], n)
    ntok = rng.integers(1, 4, n)
    z = rng.normal(size=n)
    shift = {"a": 0.0, "b": 5.0, "c": -5.0}
    p = -0.5 * z + np.array([shift[e] for e in ending]) + 3 * ntok + rng.normal(size=n)
    assert rh.partial_spearman(p, z, ntok, ending) < -0.3
    p0 = np.array([shift[e] for e in ending]) + 3 * ntok + rng.normal(size=n)  # no z effect
    assert abs(rh.partial_spearman(p0, z, ntok, ending)) < 0.15


def test_rho_test_pass_needs_null_and_floor():
    rng = np.random.default_rng(1)
    n = 200
    ending, ntok, z = np.array(["a"] * n), np.ones(n), rng.normal(size=n)
    strong = rh.rho_test(-z + 0.5 * rng.normal(size=n), z, ntok, ending, rng, n_perm=200)
    assert strong["passes"] and strong["rho"] < -0.5
    wrong_sign = rh.rho_test(z + 0.5 * rng.normal(size=n), z, ntok, ending, rng, n_perm=200)
    assert not wrong_sign["passes"]


def _table(m_pass, f_pass, fnoie_pass, num_pass, auc=0.9, n_m=190):
    rows = []
    for i, k in enumerate([9, 14, 20, 25]):
        for d, passes in (("gender", {"m": m_pass[i], "f": f_pass[i], "f_no_ie": fnoie_pass[i],
                                      "n": False}),
                          ("number", {"m": num_pass[i], "f": num_pass[i], "f_no_ie": False,
                                      "n": False})):  # fmt: skip
            for g, p in passes.items():
                rows.append({"layer": k, "direction": d, "group": g, "passes": p,
                             "n": n_m if g == "m" else 500, "auc_mf": auc})  # fmt: skip
    return pd.DataFrame(rows)


def test_readings():
    T, F = [True] * 4, [False] * 4
    assert rh.reading(_table(T, T, T, F))["reading"] == "lean generalises"
    assert rh.reading(_table(T, T, F, F))["reading"] == "mixed"  # only with -ie
    assert rh.reading(_table(T, T, T, T))["reading"] == "mixed"  # number control passes too
    assert rh.reading(_table(F, F, F, F))["reading"] == "training-set property"
    assert rh.reading(_table(T, F, F, F))["reading"] == "mixed"
    assert rh.reading(_table(T, T, T, F, auc=0.55))["reading"] == "no valid test"
    assert rh.reading(_table(T, T, T, F, n_m=90))["reading"] == "no valid test"
