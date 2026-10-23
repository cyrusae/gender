import pandas as pd

from gbleed import known_filter as kf
from gbleed.models import model_slug


def test_qwen_is_identity():
    assert kf.known_lemmas("Qwen/Qwen3-8B-Base", "de") is None
    assert kf.loss_by_gender("Qwen/Qwen3-8B-Base", "es", ["a", "b"], ["m", "f"]) == {
        "f": 0.0,
        "m": 0.0,
    }


def test_gemma_filter_and_portability(tmp_path, monkeypatch):
    m = "google/gemma-3-4b-pt"
    es, de = tmp_path / "es" / model_slug(m), tmp_path / "de" / model_slug(m)
    es.mkdir(parents=True)
    de.mkdir(parents=True)
    pd.DataFrame({"lang": ["es", "es", "de"], "lemma": ["casa", "perro", "casa2"],
                  "passed": [True, False, True]}).to_csv(es / "items.csv", index=False)  # fmt: skip
    pd.DataFrame({"lemma": ["Hund", "Katze"], "status": ["known", "unsure"]}).to_csv(
        de / "scores.csv", index=False
    )
    monkeypatch.setattr(kf, "ROOTS", {"es": str(tmp_path / "es"), "de": str(tmp_path / "de")})
    assert kf.known_lemmas(m, "es") == {"casa"}
    assert kf.known_lemmas(m, "de") == {"Hund"}
    loss = kf.loss_by_gender(m, "de", ["Hund", "Katze", "Tisch", "Uhr"], ["m", "f", "m", "f"])
    assert loss == {"f": 1.0, "m": 0.5}
    assert not kf.portable(loss)
    assert kf.portable({"m": 0.3, "f": 0.1})
