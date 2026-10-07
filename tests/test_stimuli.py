from pathlib import Path

import pandas as pd
import pytest

from gbleed.phase0 import LANG_CONFIG, check_shots_disjoint, meta_prompt, shot_orderings
from gbleed.stimuli import flipped_pairs, load_stimuli

# Phase 0-format stimulus lists (lang, lemma, gender, ...); other phases have their own schemas.
STIM = sorted(
    [
        *Path("data/stimuli").glob("phase0_*.csv"),
        Path("data/stimuli/classics.csv"),
        Path("data/stimuli/phase2_pool_v2.csv"),
    ]
)


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


def test_phase1_stimuli_disjoint_and_clean():
    import pandas as pd

    n = pd.read_csv("data/stimuli/phase1_nonce_v1.csv")
    assert set(n.split) == {"train", "test"}
    assert not set(n[n.split == "train"].stem) & set(n[n.split == "test"].stem)
    v = pd.read_csv("data/stimuli/phase1_verbs_v1.csv")
    assert (v.form_o == v.stem + "o").all() and (v.form_a == v.stem + "a").all()
    assert not set(n.form_a) & set(v.form_a)


def test_batched_request_bookkeeping():
    """_requests/_assemble must map flat batched results back to the right quantities."""
    from gbleed.phase0 import _assemble, _requests

    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2]}

        def decode(self, ids):
            return "x"

    for lang in ("de", "es"):
        reqs, nc = _requests(lang, "Brücke" if lang == "de" else "puente")
        n_orders = 2
        assert len(reqs) == n_orders * nc + 4  # quiz candidates per ordering + 2 frames x (m, f)
        vals = [0.0] * (n_orders * nc) + [-1.0, -3.0, -2.0, -2.5]
        out = _assemble(lang, "x", vals, nc, Tok())
        assert out["ctx1_margin"] == 2.0 and out["ctx2_margin"] == 0.5


def test_homograph_verb_frame_same_for_both_genders():
    """The verb-frame context must not differ by gender (v2's yo/usted gave it away)."""
    df = pd.read_csv("data/stimuli/phase2_final_v3.csv", keep_default_na=False)
    h = df[df.set == "homograph"]
    assert h.verb_frame.nunique() == 1
    assert set(h.gender) == {"m", "f"}


def test_phase3_known_frames_separate_all_three_genders():
    from gbleed.phase3_known import FRAMES

    for _, arts in FRAMES:
        assert len(set(arts.values())) == 3


def test_phase3_pool_excludes_held_out_and_morphological_neuters():
    from gbleed.phase3_stimuli import POOL, held_out_phase45

    p = pd.read_csv(POOL, keep_default_na=False)
    w = p[p.set != "multi"].lemma
    assert not set(w) & held_out_phase45()
    assert not w.str.contains(r"(?:chen|lein)$").any()
    assert not w.str.startswith("Ge").any()


def test_split_compound_head_first_part_and_linkers():
    from gbleed.phase3_stimuli import split_compound

    g = {w.lower(): (w, x) for w, x in [("Haus", "n"), ("Herr", "m"), ("Bett", "n"), ("Lei", "f"),
         ("Weisung", "f"), ("Sonne", "f"), ("Schein", "m"), ("Date", "n"), ("Schutz", "m")]}  # fmt: skip
    assert split_compound("Herrenhaus", g) == ("Haus", "n", "Herr", "m")  # -en- after m: ok
    assert split_compound("Sonnenschein", g) == ("Schein", "m", "Sonne", "f")  # -n- after f in -e
    assert split_compound("Bettelei", g) is None  # heads need >= 4 letters
    assert split_compound("Ausweisung", g) is None  # particle prefix: a derivation
    assert split_compound("Datenschutz", g)[3] == ""  # -n- after neuter: first part unresolved
