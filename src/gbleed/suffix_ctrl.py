"""Suffix follow-up test set `suffix_ctrl` (pre-registration adopted 2026-10-08, D1-D4).

Derived German nouns of every gender, so the single-root gender direction can be tested on
whether it *ranks* suffix nouns by gender (S1), not only where feminine-suffix nouns sit.
  masculine  -ismus (borrowed)                     (-ling dropped, D2)
  neuter     -tum (native), -ment (borrowed)       (some are masculine: der Reichtum, der Moment;
                                                    genders always come from Wiktionary)
  feminine   top-ups for -ung, -keit (native; the v3 suffix set kept only 1 known noun each)

Same filters as the Phase 3 pool (phase3_stimuli.exclusions), with one fix: the compound check
there treats the suffix itself as a head (Kapital-ismus: "Ismus" is a Wiktionary noun), which is
why the v3 suffix set has no -ismus nouns. Here an ending only marks a compound if it is a lexicon
noun *longer than the suffix* (Bundes-regierung still is one). Nouns in any Phase 3 training set
are excluded (asserted). Up to 20 per suffix, Zipf >= 2.5, seeded.
"""

from __future__ import annotations

import pandas as pd

from .lexicon import load_lexicon, source_tag
from .phase2_stimuli import flags
from .phase3_stimuli import SEED, _base, etymology, exclusions, german_verbs

POOL = "data/stimuli/phase3_suffixctrl_pool_v1.csv"
SUFFIXES = {"ismus": "m", "tum": "n", "ment": "n", "ung": "f", "keit": "f"}
N_PER = 20
ZMIN = 2.5
SOCIAL_GENDER = {"Feminismus", "Chauvinismus"}  # PI review 2026-10-08
TRAIN_SETS = ["strat3", "matched3", "matched3_end", "matched2", "compound_train"]


def _suffix(w: str) -> str:
    for s in SUFFIXES:
        if w.lower().endswith(s) and len(w) >= len(s) + 3:
            return s
    return ""


def _real_compound(w: str, nouns: set[str], suffix: str) -> bool:
    lw = w.lower()
    return any(lw[i:] in nouns and len(lw) - i > len(suffix) for i in range(3, len(lw) - 2))


def _latin_sourced(words: set[str]) -> set[str]:
    """Nouns whose etymology templates name a Latin source (la, la-new, ML.): Latin -um words
    (Ultimatum, Faktum, Rektum) as opposed to the native suffix -tum (Irrtum < MHG irretuom)."""
    import json

    from .lexicon import dump_path

    out = set()
    with open(dump_path("de"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("lang_code") != "de" or r.get("pos") != "noun" or r.get("word") not in words:
                continue
            for t in r.get("etymology_templates", []):
                vals = [str(v) for k, v in t.get("args", {}).items() if k != "1"]
                if any(
                    v.split(":")[0] in ("la", "la-new", "ML.") or v.startswith(":bor")
                    for v in vals[:2]
                ):
                    out.add(r["word"])
    return out


def build() -> pd.DataFrame:
    lex = load_lexicon("de")
    nouns = set(lex.lemma.str.lower())
    base = _base(lex).copy()
    base["suffix"] = base.lemma.map(_suffix)
    base = base[(base.suffix != "") & (base.zipf >= ZMIN)].copy()
    why = exclusions(base, nouns, german_verbs())
    # Replace the generic compound verdict with the suffix-aware one.
    real = [_real_compound(w, nouns, s) for w, s in zip(base.lemma, base.suffix, strict=True)]
    why = [
        "; ".join([r for r in x.split("; ") if r and r != "compound"] + (["compound"] if c else []))
        for x, c in zip(why, real, strict=True)
    ]
    base["excluded"] = why
    v3 = pd.read_csv("data/stimuli/phase3_final_v3.csv", keep_default_na=False)
    train = set(v3[v3.set.isin(TRAIN_SETS)].lemma)
    existing = set(v3[v3.set == "suffix"].lemma)
    base.loc[base.lemma.isin(train), "excluded"] = "Phase 3 training noun"
    # PI review 2026-10-08: (1) -tum must be the native German suffix (Wachstum, Eigentum), not a
    # Latin -um word that happens to end in -tum (Ultimatum, Faktum, Praeteritum, Rektum);
    # (2) concepts carrying social gender in their meaning (as the sex-associated rule).
    latin_tum = (base.suffix == "tum") & base.lemma.isin(_latin_sourced(set(base.lemma)))
    base.loc[latin_tum, "excluded"] = "Latin -um word, not the suffix -tum"
    base.loc[base.lemma.isin(SOCIAL_GENDER), "excluded"] = "social gender in meaning"
    keep = base[base.excluded == ""]
    keep = keep[~keep.lemma.isin(existing)]  # already in the v3 suffix test set
    parts = [g.sample(min(N_PER, len(g)), random_state=SEED) for _, g in keep.groupby("suffix")]
    out = pd.concat(parts)
    assert not set(out.lemma) & train, "a training noun in suffix_ctrl"
    out = flags(out)
    out["loan"] = out.lemma.map(etymology(set(out.lemma))).fillna("unknown")
    out["set"] = "suffix_ctrl"
    out["lang"] = "de"
    out["source"] = source_tag("de")
    cols = ["lang", "lemma", "gender", "concept_en", "gloss", "set", "suffix", "loan", "zipf",
            "en_overlap", "sex_assoc", "source"]  # fmt: skip
    out = out[cols].sort_values(["suffix", "gender", "zipf"], ascending=[True, True, False])
    out.to_csv(POOL, index=False)
    base[["lemma", "gender", "suffix", "zipf", "excluded"]].to_csv(
        POOL.replace(".csv", "_screening.csv"), index=False
    )
    return out
