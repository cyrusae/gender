"""Noun-sense adjective sensitivity check (gemma-slim prereg, "Noun-sense sensitivity check";
PI 2026-10-22).

Some Phase 5 English readout adjectives are mostly nouns (expert, assistant, consonant), so
their Glasgow gender rating may describe the noun (who an assistant is) rather than an
adjective. Rule, fixed before any Gemma result: drop a word when its dominant part of speech in
SUBTLEX-US (Brysbaert, New & Keuleers 2012, `Dom_PoS_SUBTLEX`) is Noun; words SUBTLEX lacks fall
back to WordNet's SemCor sense counts (noun uses > adjective uses). The same rule on every
family's list. The check reruns the Phase 5 analysis with those words removed from the English
graded readouts (R1, R1-EN, RN-R1: the regression drops their columns) and compares verdicts with
the published ones. Qwen3 Phase 5 results already exist, so on Qwen this is exploratory.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

SUBTLEX = "data/raw/subtlex/subtlex_us_pos.csv"
OUT = Path("results/noun_sense")
ENGLISH_GRADED = {"R1", "R1-EN", "RN-R1"}
FAMILIES = ["qwen3", "eurollm", "gemma3"]


def _wordnet_counts(w: str) -> tuple[int, int]:
    import nltk

    nltk.data.path.insert(0, "data/raw/nltk_data")
    from nltk.corpus import wordnet as wn

    def c(pos):
        return sum(lm.count() for s in wn.synsets(w, pos) for lm in s.lemmas()
                   if lm.name().lower() == w)  # fmt: skip

    return c("n"), c("a") + c("s")


def rule(families=FAMILIES) -> pd.DataFrame:
    """One row per English adjective on any family's list: SUBTLEX dominant PoS (and its share),
    WordNet fallback counts, and `drop`."""
    s = pd.read_csv(SUBTLEX, keep_default_na=False, dtype={"Word": str})
    s = s.assign(w=s.Word.str.lower()).drop_duplicates("w").set_index("w")
    words = sorted({w.lower() for f in families
                    for w in pd.read_csv(f"data/stimuli/phase5_adjectives_{f}_v1.csv",
                                         keep_default_na=False, dtype={"word": str}).word})  # fmt: skip
    rows = []
    for w in words:
        if w in s.index:
            r = {"word": w, "source": "subtlex", "dom_pos": s.at[w, "Dom_PoS_SUBTLEX"],
                 "dom_share": float(s.at[w, "Percentage_dom_PoS"] or np.nan)}  # fmt: skip
            r["drop"] = r["dom_pos"] == "Noun"
        else:
            n, a = _wordnet_counts(w)
            r = {"word": w, "source": "wordnet", "wn_noun": n, "wn_adj": a, "drop": n > a}
        rows.append(r)
    return pd.DataFrame(rows)


def drop_set() -> set[str]:
    d = rule()
    return set(d.word[d["drop"]])


@contextmanager
def dropping(words: set[str]):
    """Rerun the analysis with `words` removed from the English graded readouts: ratings_for
    tags the rating table with its readout, graded keeps only the kept adjectives' columns."""
    from . import phase5_analysis as pa

    rf0, gr0 = pa.ratings_for, pa.graded

    def ratings_for(name, fam, read):
        t = rf0(name, fam, read)
        t.attrs["readout"] = name
        return t

    def graded(shift, rat):
        if rat.attrs.get("readout") in ENGLISH_GRADED:
            keep = ~rat.word.astype(str).str.lower().isin(words).to_numpy()
            return gr0(shift[:, keep], rat[keep].reset_index(drop=True))
        return gr0(shift, rat)

    pa.ratings_for, pa.graded = ratings_for, graded
    try:
        yield
    finally:
        pa.ratings_for, pa.graded = rf0, gr0


def run_model(model_id: str, root: str | None = None, rc: str | None = None) -> pd.DataFrame:
    """Phase 5 verdicts for one model, as published and with the noun-sense words dropped."""
    from transformers import AutoTokenizer

    from .models import model_slug
    from .phase5 import family
    from .phase5_analysis import analyze_run, failed_wordings, verdicts

    fam = family(model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    rc = rc or "results/phase5_readoutcheck"
    failed = failed_wordings(model_id, rc) if (Path(rc) / model_slug(model_id)).exists() else set()
    d = Path(root or paths.p5_dir()) / model_slug(model_id)
    dirs = [r for r in paths.visible(sorted(d.iterdir())) if (r / "vectors.json").exists()]
    words = drop_set()
    out = []
    for tag in ("orig", "noun_sense_dropped"):
        if tag == "orig":
            runs = [analyze_run(r, fam, tok, failed=failed) for r in dirs]
        else:
            with dropping(words):
                runs = [analyze_run(r, fam, tok, failed=failed) for r in dirs]
        v = verdicts(runs)
        out.append(v[v.readout.isin(ENGLISH_GRADED)].assign(version=tag, model=model_id))
    return pd.concat(out)


def compare(v: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, lang, positions, readout): verdict (majority) and effect, both ways."""
    k = ["model", "lang", "positions", "readout"]
    a = v[v.version == "orig"].set_index(k)[["passed", "layers", "majority", "effect"]]
    b = v[v.version != "orig"].set_index(k)[["passed", "majority", "effect"]]
    c = a.join(b, rsuffix="_dropped").reset_index()
    c["verdict_changed"] = c.majority != c.majority_dropped
    return c


def main(models: list[str], root: str | None = None, rc: str | None = None) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    r = rule()
    r.to_csv(OUT / "rule.csv", index=False)
    for f in FAMILIES:
        lst = pd.read_csv(f"data/stimuli/phase5_adjectives_{f}_v1.csv", keep_default_na=False,
                          dtype={"word": str})  # fmt: skip
        n = lst.word.str.lower().isin(set(r.word[r["drop"]])).sum()
        print(f"{f}: {n} of {len(lst)} adjectives dropped by the noun-sense rule")
    v = pd.concat([run_model(m, root, rc) for m in models])
    v.to_csv(OUT / "verdicts.csv", index=False)
    c = compare(v)
    c.to_csv(OUT / "compare.csv", index=False)
    print(c.to_string(index=False))
    return c
