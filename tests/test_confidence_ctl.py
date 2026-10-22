import numpy as np
import pandas as pd

from gbleed import confidence_ctl as cc
from gbleed import phase5_analysis as pa
from gbleed.p29 import _contrast


def _rat(n=30, seed=0):
    r = np.random.default_rng(seed).normal(size=(n, 5))
    return pd.DataFrame(r, columns=["GEND", "VAL", "AROU", "SIZE", "zipf"])


def test_graded_mid_without_mid_is_graded():
    rat, y = _rat(), np.random.default_rng(1).normal(size=(6, 30))
    for a, b in zip(pa.graded(y, rat), cc.graded_mid(y, rat), strict=True):
        assert np.allclose(a, b)


def test_graded_mid_matches_lstsq_with_extra_column():
    rat = _rat()
    rng = np.random.default_rng(2)
    y, m = rng.normal(size=(5, 30)), rng.normal(size=(5, 30))
    g, _ = cc.graded_mid(cc.Shift(y, m), rat)
    X = np.column_stack([np.ones(30), ((rat - rat.mean()) / rat.std(ddof=0)).to_numpy()])
    bf = [-np.linalg.lstsq(np.column_stack([X, m[r]]), y[r], rcond=None)[0][1] for r in range(5)]
    assert np.allclose(g, bf)


def test_shift_slicing_keeps_mid():
    s = cc.Shift(np.arange(12.0).reshape(3, 4), np.arange(12.0).reshape(3, 4) * 10)
    t = s[:, [0, 2]]
    assert np.allclose(t.mid, np.asarray(t) * 10)
    assert not isinstance(s[0], cc.Shift)


def test_pivot_avg_matches_contrast_and_midpoint():
    keys = pd.DataFrame({"word": ["a", "a", "a", "a", "b", "b"],
                         "wording": ["W1", "W1", "W2", "W2", "W1", "W1"],
                         "gender": ["m", "f", "m", "f", "m", "f"], "stem": "s", "ending": "",
                         "primary": True})  # fmt: skip
    keys["row"] = np.arange(len(keys))
    X = np.random.default_rng(3).normal(size=(6, 4))
    _, d = _contrast(keys, X)
    _, s = cc._contrast_mid(keys, X)
    assert np.allclose(np.asarray(s), d)
    assert np.allclose(s.mid[1], (X[4] + X[5]) / 2)
    assert np.allclose(s.mid[0], (X[0] + X[1] + X[2] + X[3]) / 4)


def test_midpoint_context_restores():
    from gbleed import p29, p30, p31

    before = (pa.graded, pa.shifts, p29._contrast, p30._contrast, p31._pair)
    with cc.midpoint():
        assert pa.graded is cc.graded_mid
    assert before == (pa.graded, pa.shifts, p29._contrast, p30._contrast, p31._pair)
