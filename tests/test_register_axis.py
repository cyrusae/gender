import numpy as np

from gbleed import register_axis as ra


def test_gloss_aoa_exact_and_head(monkeypatch, tmp_path):
    import pandas as pd

    lex = pd.DataFrame({"lemma": ["A", "B", "C", "D"],
                        "concept_en": ["juice", "operating lever", "pair of compasses", ""]})  # fmt: skip
    lex.to_csv(tmp_path / "xx_nouns.csv", index=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "lexicon").mkdir(parents=True)
    lex.to_csv(tmp_path / "data" / "lexicon" / "xx_nouns.csv", index=False)
    aoa, how = ra.gloss_aoa(["A", "B", "C", "D"], "xx", {"juice": 3.0, "lever": 9.0})
    assert aoa[0] == 3.0 and aoa[1] == 9.0 and np.isnan(aoa[2]) and np.isnan(aoa[3])
    assert how == ["exact", "head", "", ""]
