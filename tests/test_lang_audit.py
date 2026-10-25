import numpy as np
import pandas as pd

from gbleed import lang_audit as la
from gbleed.languages import spec


def test_loo_ending_accuracy_perfect_and_chance():
    w = ["casa", "mesa", "rosa", "gato", "pato", "dato"]
    g = ["f", "f", "f", "m", "m", "m"]
    assert la.loo_ending_accuracy(w, g, 1) == 1.0
    w2 = ["ab", "cb", "db", "eb"]
    g2 = ["f", "m", "f", "m"]  # same ending, half each: leave-one-out picks the other gender
    assert la.loo_ending_accuracy(w2, g2, 1) == 0.0


def test_loo_cell_majority_excludes_self():
    assert la.loo_cell_majority(["a", "a", "a"], ["f", "f", "m"]) == ["f", "f", "f"]


def test_within_cell_d_detects_shift():
    rng = np.random.default_rng(0)
    g = np.array(["m"] * 50 + ["f"] * 50)
    cells = np.array(["x", "y"] * 50)
    x = rng.normal(size=100) + (g == "f") * 1.0
    d = la.within_cell_d(x, g, cells, "m")["f"]
    assert 0.5 < d < 1.5


def test_initial_letters_flags_skew():
    df = pd.DataFrame({"lemma": ["ax"] * 10 + ["bx"] * 10 + ["cx"] * 10,
                       "gender": ["m"] * 10 + ["f"] * 5 + ["m"] * 5 + ["f"] * 10})  # fmt: skip
    t = la.initial_letters(df).set_index("initial")
    assert t.loc["a", "flag"] and not t.loc["b", "flag"]


def test_specs_have_cells():
    assert spec("de").cell("Gabel") == "el"
    assert spec("ru").cell("кость") == "soft"


def test_initial_letters_flags_missing_gender_at_low_base_rate():
    # 17 masculine a-initial, no feminine, with 20% feminine overall (the Spanish case)
    df = pd.DataFrame({"lemma": ["ax"] * 17 + ["bx"] * 83 + ["cx"] * 25,
                       "gender": ["m"] * 17 + ["m"] * 58 + ["f"] * 25 + ["m"] * 25})  # fmt: skip
    t = la.initial_letters(df).set_index("initial")
    assert t.loc["a", "flag"] and t.loc["a", "missing"] == "f"


def test_specs_point_to_frames():
    assert spec("es").frames()["frames"][0][1]["f"] == "la"
    ru = spec("ru").frames()
    assert ru[0][1]["f"] == "наша" and not spec("ru").has_articles


def test_semantic_class_from_gloss():
    from gbleed.lang_audit import semantic_class

    assert semantic_class("March") == "month"
    assert semantic_class("the Monday") == "weekday"
    assert semantic_class("winter") == "season"
    assert semantic_class("the number four") == "number"
    assert semantic_class("bridge") == ""
    assert semantic_class("marching band") == ""
