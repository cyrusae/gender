"""Phase 3 stimulus pool (German): masculine, feminine and neuter inanimate nouns.

Sets (all genders from Wiktionary via the lexicon):
  matched3   TRAIN (primary): simplex m/f/n nouns with equal numbers of each gender for every final
             two letters (der Mantel / die Gabel / das Kabel), so the ending carries no gender
             information. Neuter is the *reference* class for separate masculine and feminine
             vectors (masc = m - n, fem = f - n).
  matched2   TRAIN (single m-f axis, for comparison with the literature): the same, m/f only, so it
             can use endings that have no neuter (a superset of matched3's m and f).
  suffix     TEST ONLY: nouns with a gender-predicting suffix (-ung, -heit, -schaft, -ismus...).
  multi      TEST ONLY: der/die See-type items (multi_gender_spec.csv), spelling constant.

Filters on every set, beyond `eligible()`'s (inanimate, unmarked, no other part of speech or
inflected-form homograph, not more frequent in English):
  - simplex only: compounds take their head's gender (das Herrenhaus <- das Haus), so the last
    token would give the gender away. A noun counts as a compound if any ending of >= 3 letters,
    after >= 3 letters, is itself a lexicon noun. Deliberately over-inclusive.
  - no place or other proper names (German place names are neuter): gloss check.
  - no diminutives (-chen, -lein; neuter for a morphological reason) and no Ge- nouns (Ge-...-e
    collectives are neuter by pattern); applied to every gender, so the filter is symmetric.
  - no chemical elements/compounds (a neuter-heavy semantic class): gloss check.
  - no nominalised infinitives: das Essen is already out (also a verb); compounds ending in an
    infinitive (das Auswendiglernen, das Nichtstun) are caught by matching the end of the noun
    against every German verb in Wiktionary.
  - chemical elements and compounds also by WordNet (Uran "uranium" has no telling gloss).
  - a second animacy check by WordNet (person/organism), for person nouns the lexicon misses.
  - not in Phase 4-5 held-out sets: German nouns of flipped pairs and the classics.
  - training sets also drop English-overlapping and sex-associated nouns (as in Phase 2).
Each candidate then has to pass the Phase 3 "known" check (three-way article frames,
`phase3_known.py`) before `finalize()` samples the matched sets.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from .lexicon import _wordnet, dump_path, load_lexicon, source_tag
from .phase0 import LANG_CONFIG
from .phase2_stimuli import _clean, flags

POOL = "data/stimuli/phase3_pool_v1.csv"
SEED = 0
ZMIN = 2.0
N_SUFFIX = 20
PROPER = re.compile(
    r"\b(city|town|municipality|village|district|county|province|region|state|country|river|"
    r"lake|mountain|island|capital|surname|given name|placename|place name|football club|"
    r"in (germany|austria|switzerland|poland|france|italy))\b",
    re.IGNORECASE,
)
CHEMICAL = re.compile(
    r"\b(chemical element|element with|amino acid|alkaloid|enzyme|hormone|compound of|"
    r"chemistry\)|mineral\b|isotope)",
    re.IGNORECASE,
)
DIMINUTIVE = re.compile(r"(?:chen|lein)$")


CHEM_ROOTS = ["chemical_element.n.01", "compound.n.02", "alloy.n.01"]
# Second animacy check: the lexicon tag misses some person nouns (Individuum, Bajazzo).
ANIMATE_ROOTS = ["person.n.01", "organism.n.01", "people.n.01"]


def german_verbs() -> set[str]:
    out = set()
    with open(dump_path("de"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            w = r.get("word", "")
            if r.get("lang_code") == "de" and r.get("pos") == "verb" and " " not in w:
                out.add(w.lower())
    return out


def _under(concept: str, wn, roots) -> bool:
    for syn in wn.synsets(str(concept).replace(" ", "_"), pos="n")[:1]:
        if roots & {h for path in syn.hypernym_paths() for h in path}:
            return True
    return False


def _base(lex: pd.DataFrame) -> pd.DataFrame:
    return lex[
        lex.gender.isin(["m", "f", "n"]) & (lex.animacy == "inanimate") & (lex.marked == "")
        & (lex.regions == "") & (lex.also_pos == "") & (lex.also_form == "")
        & (lex.zipf_en < lex.zipf) & (lex.zipf >= ZMIN)
    ]  # fmt: skip


def held_out_phase45() -> set[str]:
    pairs = pd.read_csv("data/lexicon/pairs_de_es.csv", keep_default_na=False)
    flipped = pairs.flipped.astype(str).eq("True")
    classics = pd.read_csv("data/stimuli/classics.csv", keep_default_na=False)
    return set(pairs[flipped].de_lemma) | set(classics[classics.lang == "de"].lemma)


def exclusions(df: pd.DataFrame, nouns: set[str], verbs: set[str]) -> pd.Series:
    """Why each noun is excluded from every Phase 3 set ('' = kept)."""
    wn = _wordnet()
    roots = {wn.synset(r) for r in CHEM_ROOTS}
    animate = {wn.synset(r) for r in ANIMATE_ROOTS}

    def nominalised(w: str) -> bool:
        w = w.lower()
        return w.endswith("n") and any(w[i:] in verbs for i in range(len(w) - 2))

    def compound(w: str) -> bool:
        w = w.lower()
        return any(w[i:] in nouns for i in range(3, len(w) - 2))

    held = held_out_phase45()
    shots = {n for n, _ in LANG_CONFIG["de"]["shots"]}
    gloss = df.gloss.astype(str) + " " + df.concept_en.astype(str)
    out = []
    for w, g, c in zip(df.lemma, gloss, df.concept_en, strict=True):
        why = [
            r for r, c in [
                ("compound", compound(w)), ("proper name", bool(PROPER.search(g))),
                ("diminutive", bool(DIMINUTIVE.search(w))), ("Ge-", w.startswith("Ge")),
                ("chemical", bool(CHEMICAL.search(g)) or _under(c, wn, roots)),
                ("animate (wordnet)", _under(c, wn, animate)),
                ("nominalised verb", nominalised(w)), ("phase 4-5 held out", w in held),
                ("phase 0 shot", w in shots),
            ] if c
        ]  # fmt: skip
        out.append("; ".join(why))
    return pd.Series(out, index=df.index)


def multi() -> pd.DataFrame:
    s = pd.read_csv("data/stimuli/multi_gender_spec.csv", keep_default_na=False)
    s = s[(s.decision == "keep") & (s.lang == "de")]
    return s[["lemma", "type", "gloss_m", "gloss_f"]].assign(set="multi", gender="multi")


def build() -> pd.DataFrame:
    lex = load_lexicon("de")
    nouns = set(lex.lemma.str.lower())
    base = _base(lex).copy()
    base["exclude"] = exclusions(base, nouns, german_verbs())
    dropped = base[base.exclude != ""]
    base = flags(base[base.exclude == ""])
    base["set"] = ["suffix" if s else "simplex" for s in base.de_suffix]
    # Suffix test group: up to N_SUFFIX frequent nouns per suffix (it's a test set; the full
    # ~1,700 would only slow the known check).
    suf = base[(base.set == "suffix") & (base.zipf >= 3.0)]
    suf = pd.concat(
        g.sample(min(N_SUFFIX, len(g)), random_state=SEED) for _, g in suf.groupby("de_suffix")
    )
    base = pd.concat([base[base.set == "simplex"], suf])
    base["ending"] = base.lemma.str[-2:]
    pool = pd.concat([base, multi()], ignore_index=True)
    pool["lang"] = "de"
    pool["source"] = source_tag("de")
    cols = ["lang", "lemma", "gender", "concept_en", "gloss", "set", "de_suffix", "ending", "zipf",
            "n_tokens_qwen3", "en_overlap", "sex_assoc", "type", "gloss_m", "gloss_f", "source"]  # fmt: skip
    pool = pool.reindex(columns=cols)
    pool.to_csv(POOL, index=False)
    print(pool.groupby(["set", "gender"]).size().unstack(fill_value=0).to_string())
    print("clean simplex (training-eligible):")
    print(_clean(pool[pool.set == "simplex"]).groupby("gender").size().to_string())
    print("excluded by reason (first reason):")
    print(dropped.exclude.str.split("; ").str[0].value_counts().to_string())
    return pool
