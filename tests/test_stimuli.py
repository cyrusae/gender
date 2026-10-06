from pathlib import Path

import pytest

from gbleed.phase0 import LANG_CONFIG, check_shots_disjoint, meta_prompt, shot_orderings
from gbleed.stimuli import flipped_pairs, load_stimuli

STIM = sorted(Path("data/stimuli").glob("*.csv"))


@pytest.mark.parametrize("path", STIM, ids=lambda p: p.name)
def test_stimulus_files_valid(path):
    df = load_stimuli(path)
    check_shots_disjoint(df)
    assert len(df) > 0


def test_seed_has_flipped_pairs():
    df = load_stimuli("data/stimuli/phase0_seed.csv")
    pairs = flipped_pairs(df)
    assert len(pairs) >= 10
    assert (pairs.de_gender != pairs.es_gender).all()


def test_shots_balanced_and_orderings_end_differently():
    for cfg in LANG_CONFIG.values():
        arts = [a for _, a in cfg["shots"]]
        assert arts.count(cfg["meta"]["m"].strip()) == arts.count(cfg["meta"]["f"].strip())
        a, b = shot_orderings(cfg["shots"])
        assert a[-1][1] != b[-1][1]


def test_meta_prompt_ends_before_article():
    cfg = LANG_CONFIG["de"]
    p = meta_prompt(cfg, cfg["shots"], "Brücke")
    assert p.endswith("\nBrücke:")


def test_classify_statuses():
    import pandas as pd

    from gbleed.phase0 import classify

    df = pd.DataFrame(
        {
            "gender": ["m", "m", "f", "f", "m"],
            "ctx1_margin": [3.0, 3.0, -0.2, 2.0, -4.0],
            "ctx2_margin": [2.0, -3.0, -5.0, 2.0, -0.5],
            "meta_margin": [0.1] * 5,
        }
    )
    out = classify(df, min_margin=1.0)
    assert out.status.tolist() == ["known", "conflict", "unsure", "wrong", "wrong"]
    assert out.passed.tolist() == [True, False, False, False, False]
