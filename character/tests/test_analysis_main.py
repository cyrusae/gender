"""Synthetic-data checks for analysis_main (no real outputs used)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis_main as am


def fake_writer(rng, effect, n_noun=120, draws=10, tau=0.10):
    """Coded-summary-like rows for writer arms with a planted f-minus-m shift difference."""
    gen, rows = {}, []
    for i in range(n_noun):
        noun, base = f"n{i}", rng.uniform(0.3, 0.8)
        for lg in am.GLANGS:
            g = rng.choice(["f", "m"])
            gen[(noun, lg)] = g
            u = rng.normal(0, tau)
            for fr in ("writer", "consider"):
                p_en = base
                p_fo = np.clip(base + (effect / 2) * (1 if g == "f" else -1) + u, 0.01, 0.99)
                for arm, p in ((am.foreign_arm(fr, lg), p_fo), (am.english_arm(fr, lg), p_en)):
                    for s in rng.random(draws) < p:
                        rows.append({"noun": noun, "modifier": arm, "valid": True,
                                     "pron": "she" if s else "he"})  # fmt: skip
    return pd.DataFrame(rows), gen


def test_q1_recovers_planted_effect():
    """Unbiased and ~95% coverage over 40 datasets with a planted 10-point effect."""
    rng = np.random.default_rng(1)
    rs = [
        am.q1_fit(am.writer_shifts(*fake_writer(rng, 0.10, n_noun=80, draws=6))) for _ in range(40)
    ]
    assert abs(np.mean([r["est"] for r in rs]) - 10) < 1.5
    assert np.mean([r["lo95"] < 10 < r["hi95"] for r in rs]) >= 0.85
    assert np.mean([r["p"] < 0.05 for r in rs]) > 0.9


def test_q1_null_false_alarm_rate():
    rng = np.random.default_rng(2)
    ps = [
        am.q1_fit(am.writer_shifts(*fake_writer(rng, 0.0, n_noun=60, draws=4)))["p"]
        for _ in range(200)
    ]
    assert np.mean(np.array(ps) < 0.05) < 0.09


def test_permutation_agrees():
    rng = np.random.default_rng(3)
    d, g = fake_writer(rng, 0.10, n_noun=80)
    r = am.q1_fit(am.writer_shifts(d, g), perm=300)
    assert r["p_perm"] < 0.05


def test_decide_and_holm():
    assert am.holm([0.001, 0.02, 0.04, 0.3]) == [True, False, False, False]
    assert am.holm([0.001, 0.012, 0.02, 0.3]) == [True, True, True, False]
    r = am.summarise(0.5, 1.0, 100)
    assert am.decide(r, False) == "no effect of interest"
    assert am.decide(am.summarise(3.0, 4.0, 100), False) == "inconclusive"


def test_gatekeeping_blocks_cells_when_headline_fails():
    rng = np.random.default_rng(4)
    cells = {c: pd.Series(rng.normal(0, 0.2, 150)) for c in "abcd"}
    _head, res = am.gatekeep(cells, am.mean_fit, headline_ok=False)
    assert all(not r["decision"].startswith("effect") for r in res.values())


def test_q2_modifier_and_frame_diffs():
    rng = np.random.default_rng(5)
    rows = []
    for i in range(150):
        b = rng.uniform(0.3, 0.7)
        for fr in ("writer", "consider"):
            for m, p in (
                ("plain_aan", b),
                ("a rough", b - 0.10),
                ("the best", b + 0.0),
                ("plain_the", b),
            ):
                p = p + (0.06 if fr == "writer" else 0.0)
                for s in rng.random(10) < p:
                    rows.append({"noun": f"n{i}", "modifier": f"{fr}|{m}", "valid": True,
                                 "pron": "she" if s else "he"})  # fmt: skip
    ss = am.she_table(pd.DataFrame(rows))
    assert am.base_of("the best") == "plain_the" and am.base_of("a rough") == "plain_aan"
    r = am.mean_fit(am.noun_diffs(ss, "a rough", "plain_aan"))
    assert r["lo95"] < -10 < r["hi95"]
    f = am.mean_fit(am.frame_diffs(ss, ["plain_aan", "a rough"]))
    assert f["lo95"] < 6 < f["hi95"]
