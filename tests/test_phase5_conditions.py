import numpy as np
import pandas as pd

from gbleed import phase5
from gbleed.phase5_analysis import graded, window, working_test


def test_conditions_counts_and_shared_baseline():
    c = phase5.conditions(2, 100, 20, extra_vecs=2, in_subset=[True, False])
    per = {i: [x for x in c if x[0] == i] for i in (0, 1)}
    # v: 9 doses; extras: 2 x 8 nonzero; random sweep: 20 x 8 (subset) or 20 x 1; rest 80 x 1
    assert len(per[0]) == 9 + 16 + 160 + 80
    assert len(per[1]) == 9 + 16 + 20 + 80
    zero = [x for x in c if x[2] == 0.0]
    assert all(x[1] == 0 for x in zero) and len(zero) == 2  # dose 0 once per prompt, with v
    assert {x[1] for x in per[1] if x[2] == 1.0} == set(range(103))


def test_every_position_has_no_random_dose_curve():
    c = phase5.conditions(1, 100, 0, extra_vecs=2)
    assert {m for _, vi, m in c if 1 <= vi <= 100} == {1.0}


def test_working_conditions_extras_index():
    c = phase5.working_conditions(1, 20, extra_vecs=2, n_vec_random=100)
    assert {vi for _, vi, _ in c} == {0, *range(1, 21), 101, 102}


def test_dose_subset_is_stratified_and_seeded():
    n = pd.DataFrame({"lang": "es", "set": ["a"] * 8 + ["b"] * 4, "gender": ["m", "f"] * 6,
                      "lemma": [f"w{i}" for i in range(12)],
                      "concept_en": [f"c{i}" for i in range(12)]})  # fmt: skip
    s1, s2 = phase5.dose_subset(n), phase5.dose_subset(n)
    assert s1 == s2
    picked = n[n.lemma.isin(s1)]
    assert picked.groupby(["set", "gender"]).size().min() >= 1


def test_graded_recovers_feminine_gender_effect():
    rng = np.random.default_rng(0)
    rat = pd.DataFrame({c: rng.normal(size=200) for c in ("GEND", "VAL", "AROU", "SIZE", "zipf")})
    # feminine shift: probability moves toward low GEND (feminine-rated), confounded with valence
    shift = -0.5 * rat.GEND.to_numpy() + 0.8 * rat.VAL.to_numpy()
    g, size = graded(np.vstack([shift, -shift]), rat)
    assert g[0] > 0 and g[1] < 0 and abs(size[0]) < 1e-8


def test_working_test_rank_p():
    rows = []
    for lem in range(30):
        rows.append({"vec": "v", "mult": 1.0, "lemma": lem, "score": 1.0})
        for r in range(99):
            rows.append({"vec": f"rand{r}", "mult": 1.0, "lemma": lem, "score": 0.0})
    out = working_test(pd.DataFrame(rows), np.random.default_rng(0), n_boot=50)
    assert out["p"] == 1 / 100 and out["diff_lo_95"] > 0


def test_window():
    d = pd.DataFrame({"vec": ["rand0"] * 3 + ["v"] * 4, "mult": [1.0] * 3 + [-1.0, 0.5, 1.0, 2.0],
                      "kl": [1.0, 1.0, 1.0, 0.5, 0.2, 0.9, 3.0]})  # fmt: skip
    assert window(d) == [-1.0, 0.0, 0.5, 1.0]
