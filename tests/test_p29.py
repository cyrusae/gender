import numpy as np
import pandas as pd

from gbleed import p29


def _keys():
    rows = []
    for stem in ("sia", "capi"):
        for e in ("e", "iz"):
            for w in ("W1", "W2"):
                for g in ("m", "f"):
                    rows.append({"word": stem + e, "stem": stem, "ending": e, "primary": True,
                                 "wording": w, "gender": g})  # fmt: skip
    k = pd.DataFrame(rows)
    k["row"] = np.arange(len(k))
    return k


def test_contrast_is_feminine_minus_masculine_averaged_over_wordings():
    k = _keys()
    X = np.zeros((len(k), 3))
    # feminine rows +1 in W1 and +3 in W2 on column 0; column 1 identical across articles
    X[(k.gender == "f") & (k.wording == "W1"), 0] = 1.0
    X[(k.gender == "f") & (k.wording == "W2"), 0] = 3.0
    X[:, 1] = k.row.to_numpy()  # differs by prompt, but m/f rows are adjacent: shift 1
    meta, d = p29._contrast(k, X)
    assert sorted(meta.word) == sorted(set(k.word)) and len(meta) == len(d)  # rows aligned
    assert np.allclose(d[:, 0], 2.0) and np.allclose(d[:, 1], 1.0) and np.allclose(d[:, 2], 0.0)


def test_boot_keeps_stems_together_and_brackets_the_mean():
    score = np.array([1.0, 1.0, 3.0, 3.0])
    lo, hi = p29._boot(score, np.array(["a", "a", "b", "b"]), np.random.default_rng(0), 500)
    assert lo == 1.0 and hi == 3.0  # only whole stems are resampled: means 1, 2 or 3
