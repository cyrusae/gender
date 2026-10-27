import numpy as np
import pandas as pd

from gbleed import russian_dir as rd


def test_matched_pairs_within_cell_and_caliper():
    c = np.array([1.0, 1.1, 3.0, 3.05, 2.0, 4.9, np.nan, 2.0])
    d = {"conc_raw": c, "y": np.array([1, 0, 1, 0, 1, 0, 1, 0]),
         "cells": np.array(["a", "a", "b", "b", "a", "a", "b", "b"])}  # fmt: skip
    pr = rd.matched_pairs(d)
    cal = rd.CALIPER_SD * np.nanstd(c)
    for i, j in pr:
        assert d["cells"][i] == d["cells"][j] and d["y"][i] == 1 and d["y"][j] == 0
        assert abs(c[i] - c[j]) <= cal
    assert len({j for _, j in pr}) == len(pr)  # 1:1


def test_readings():
    def res(passes):
        rows = [{f"{k}_lo": 0.6 if passes[k][i] else 0.4 for k in rd.R_A} for i in range(5)]
        return pd.DataFrame(rows)

    T, F = [True] * 5, [False] * 5
    assert rd.r_a_reading(res(dict.fromkeys(rd.R_A, T)))["reading"] == "aligns"
    one = {"ru_on_es": F, "ru_on_de": F, "es_on_ru": T, "de_on_ru": T}
    assert rd.r_a_reading(res(one))["reading"] == "one-way"
    part = {"ru_on_es": T, "ru_on_de": F, "es_on_ru": F, "de_on_ru": F}
    assert rd.r_a_reading(res(part))["reading"] == "partial"
    assert rd.r_a_reading(res(dict.fromkeys(rd.R_A, F)))["reading"] == "no alignment"
    b = pd.DataFrame({"valid": [True] * 4, "overlap": [True, True, True, False]})
    assert rd.r_b_reading(b) == "overlap"
    b2 = pd.DataFrame({"valid": [False, False, True, True], "overlap": [False] * 4})
    assert rd.r_b_reading(b2) == "no valid test"


def test_random_subsets_keep_composition():
    cells = np.array(["a"] * 6 + ["b"] * 6)
    y = np.array([1, 1, 1, 0, 0, 0] * 2)
    ru = {"cells": cells, "y": y}
    subs = rd.random_subsets(ru, [(0, 3), (6, 9)], 5)
    for s in subs:
        assert sorted(zip(cells[s], y[s], strict=True)) == [("a", 0), ("a", 1), ("b", 0), ("b", 1)]


def test_cosine_label_clips():
    from gbleed import cosine_boot as cb

    assert cb.label(1.34) == "> 1 (estimator limit)"
    assert cb.label(-1.2) == "< -1 (estimator limit)"
    assert cb.label(0.5) == "+0.50"
