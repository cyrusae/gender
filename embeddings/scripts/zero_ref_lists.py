"""Candidate reference sets for the Zhou zero-point study (docs/zhou-zero-point-prereg.md, DRAFT).

Writes CANDIDATE lists (v0, for PI review; not frozen) by rule from the Wiktionary lexicons:
  epicene   person nouns with one grammatical gender used for anyone (la persona, la víctima);
  common    person nouns with one spelling for both sexes (el/la estudiante);
  inanimate the frozen C1 nouns, frequency-matched masculine vs feminine.
Every list excludes Zhou et al.'s occupation words and definitional/attribute words. The epicene and
common pools are deliberately broad: the PI marks pi_keep (rule-based pools are noisy: printer-type
senses, proper names, plurals).

    uv run --no-project --python 3.12 --with pandas python embeddings/scripts/zero_ref_lists.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gbleed.lexicon import _wordnet  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
Z = ROOT / "embeddings/external/Bias_in_Gendered_Languages"
OUT = ROOT / "embeddings/data"
# Glosses that denote a sex (hombre, madre, rey) or a sexual role: not sex-neutral, so not epicene.
SEXED = re.compile(
    r"\b(?:man|men|woman|women|male|female|boy|girl|husband|wife|mother|father|son|daughter|brother|"
    r"sister|uncle|aunt|nephew|niece|grand(?:father|mother|son|daughter)|king|queen|prince|princess|"
    r"lord|lady|gentleman|nun|monk|bride|groom|widow|widower|whore|prostitute|harlot|guy|bloke|lad|"
    r"lass|mistress|god|goddess|virgin|maid|maiden|matron|fiancée?|bachelor|spinster|duke|duchess|"
    r"count|countess|emperor|empress|baron|baroness|actress|heroine|hero|pope|christ|messiah)\b",
    re.IGNORECASE,
)


def excluded(lang):
    # Zhou's noun lists are NOT excluded: in the original space nothing is shifted, so membership
    # doesn't matter there; post-Shift analyses report the overlap instead (in_zhou_lists column).
    ex = set()
    ex |= {w for p in json.load(open(Z / f"{lang.upper()}_definitional_pairs.json")) for w in p}
    for line in open(Z / f"{lang.upper()}_occupation_words_with_EN_translations", encoding="utf-8"):
        ex |= set(line.rstrip("\n").split("\t")[:2])
    if lang == "es":
        ex |= set("hombre niño padre masculino abuelo esposo hijo tio niña femenina tía hija esposa mujer madre abuela".split())
    return ex


def person_share(wn, concept):
    """Share of WordNet (SemCor-tagged) usage of the English gloss that is noun.person; when no
    sense has a usage count, the share of senses that are noun.person. The lexicon's own animacy
    flag deliberately errs toward 'animate' (safe for inanimate lists, wrong for person lists):
    'licuado' (smoothie) is animate there because WordNet's first sense of smoothie is a person."""
    for c in (str(concept).replace(" ", "_"), str(concept).split(" of ")[0].split()[-1] if str(concept).strip() else ""):
        lem = wn.lemmas(c, pos="n") if c else []
        if lem:
            cnt = [x.count() for x in lem]
            per = [x.synset().lexname() == "noun.person" for x in lem]
            if sum(cnt):
                return sum(n for n, p in zip(cnt, per) if p) / sum(cnt)
            return sum(per) / len(per)
    return np.nan


def build(lang):
    d = pd.read_csv(ROOT / f"data/lexicon/{lang}_nouns.csv", low_memory=False)
    wn = _wordnet()
    d["person_share"] = d["concept_en"].map(lambda c: person_share(wn, c))
    # persons: most (> half) WordNet usage of the gloss is a person, or the gloss itself says person (no WordNet entry)
    personish = (d["person_share"] > 0.5) | (d["person_share"].isna() & d["animacy_reason"].astype(str).str.contains("gloss"))
    ex = excluded(lang)
    reason = d["animacy_reason"].astype(str)
    base = (d["zipf"] >= 3) & d["marked"].isna() & d["also_pos"].isna() & ~d["lemma"].isin(ex)
    gloss_ok = ~d["gloss"].astype(str).str.contains(SEXED)
    # animacy == animate: the lexicon's overall judgement (WordNet first-sense lexname alone made
    # "licuado" (smoothie) a person, from WordNet's first sense "someone with an ingratiating
    # manner"); persons only, so animal-category nouns are dropped.
    epi = d[base & gloss_ok & personish & d["gender"].isin(["m", "f"]) & (d["animacy"] == "animate")
            & ~reason.str.contains("animal") & (reason.str.contains("noun.person")
            | ((d["animacy"] == "animate") & reason.str.contains("gloss|people-category") & ~reason.str.contains("animal")))
            & ~reason.str.contains("counterpart|personal-gender")].assign(set="epicene")  # fmt: skip
    com = d[base & gloss_ok & personish & (d["gender"] == "multi") & (d["animacy"] == "animate")
            & reason.str.contains("by-personal-gender")
            & (d["multi_type"] == "free_variation")].assign(set="common")  # fmt: skip
    c1 = pd.read_csv(OUT / f"c1_nouns_{lang}_v1.csv")
    c1 = c1[c1["in_50k"]]
    # frequency-match: within 0.25-Zipf bins, keep min(n_m, n_f) of each gender (seed 0)
    rng = np.random.default_rng(0)
    c1["bin"] = (c1["zipf"] / 0.25).round()
    keep = []
    for _, g in c1.groupby("bin"):
        k = min((g.gender == "m").sum(), (g.gender == "f").sum())
        for sex in "mf":
            sub = g[g.gender == sex]
            keep += list(sub.iloc[rng.permutation(len(sub))[:k]].index)
    ina = c1.loc[keep].assign(set="inanimate")
    zl = set()
    for name in (f"masculine_nouns_{lang}", f"feminine_nouns_{lang}"):
        zl |= set(open(Z / name, encoding="utf-8").read().split())
    for x in (epi, com):
        x["in_zhou_lists"] = x["lemma"].isin(zl)
        x["pi_keep"] = ""  # PI review: y / n
    epi.sort_values("zipf", ascending=False, inplace=True)
    com.sort_values("zipf", ascending=False, inplace=True)
    cols = ["set", "lemma", "gender", "concept_en", "gloss", "zipf", "animacy_reason", "person_share", "in_zhou_lists", "pi_keep"]
    out = pd.concat([epi.reindex(columns=cols), com.reindex(columns=cols), ina.reindex(columns=cols)])
    path = OUT / f"zero_ref_candidates_{lang}_v0.csv"
    out.to_csv(path, index=False)
    print(lang, out.groupby(["set", "gender"]).size().to_dict(), "->", path.name)


if __name__ == "__main__":
    for lang in ("es", "fr"):
        build(lang)
