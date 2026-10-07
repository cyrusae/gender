"""Phase 3 stimulus pool (German): masculine, feminine and neuter inanimate nouns.

Sets (all genders from Wiktionary via the lexicon; final sampling in `phase3_known.finalize`):
  simplex     candidates for the TRAIN sets (single-root nouns):
                matched3     primary: per final two letters x loan status, equal m, f and n
                             (der Mantel / die Gabel / das Kabel). Neuter is the *reference* class
                             for separate vectors (masc = m - n, fem = f - n).
                matched3_end secondary: per final two letters only (more nouns); loan status
                             goes in as a covariate instead.
                matched2     single m-f axis (comparison with the literature): per final two
                             letters x loan status, equal m and f; contains matched3's m/f nouns.
  compound    compounds take their head's gender (das Herrenhaus <- das Haus). One compound per
              head. Heads are split once (seeded) into:
                compound_train   secondary TRAIN set (matched like matched3, by the head's ending)
                compound_test    TEST: does a simplex-trained direction predict compound gender?
                                 Includes *conflict* compounds whose first part has a different
                                 gender (das Herrenhaus, der Herr): a gender representation should
                                 follow the head.
  suffix      TEST: nouns with a gender-predicting suffix (-ung, -heit, -schaft, -ismus...).
  multi       TEST: der/die See-type items (multi_gender_spec.csv), spelling constant.

Filters on every set, beyond `eligible()`'s (inanimate, unmarked, no other part of speech or
inflected-form homograph, not more frequent in English):
  - no place or other proper names (German place names are neuter): gloss check.
  - no diminutives (-chen, -lein; neuter for a morphological reason) and no Ge- nouns (Ge-...-e
    collectives are neuter by pattern); applied to every gender, so the filter is symmetric.
    For compounds the same checks apply to the head (das Kaffeekännchen is out).
  - no chemical elements/compounds (a neuter-heavy semantic class): gloss + WordNet.
  - no nominalised infinitives: das Essen is already out (also a verb); compounds ending in an
    infinitive (das Auswendiglernen, das Nichtstun) are caught by matching the end of the noun
    against every German verb in Wiktionary.
  - a second animacy check by WordNet (person/organism), for person nouns the lexicon misses.
  - not in Phase 4-5 held-out sets: German nouns of flipped pairs and the classics.
  - training sets also drop English-overlapping and sex-associated nouns (as in Phase 2).
Compound detection: the head is the longest ending (>= 4 letters, after >= 3 letters) that is a
lexicon noun with a single gender; kept only if the head's gender equals the compound's
(Wiktionary), which also validates the split. The first part is the remaining prefix, with a
linking element (-s-, -es-, -n-, -en-, -er-, -e-) stripped if needed, matched to a lexicon noun;
a linker counts only after first parts of a gender that takes it (else the first part's gender
is left blank and the compound is not a conflict item). Particle prefixes (aus-, vor-) are
derivations, not compounds, and are dropped.
Loan status: from Wiktionary etymology templates (borrowed vs inherited/derived from earlier
German or Germanic; word formations count as native; no etymology = unknown).
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .lexicon import _wordnet, dump_path, load_lexicon, source_tag
from .phase0 import LANG_CONFIG
from .phase2_stimuli import _clean, flags

POOL = "data/stimuli/phase3_pool_v2.csv"
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
# Linking elements, each allowed only after first parts of the genders that take it
# (Sonne-n-schein: f in -e; Arbeit-s-platz / Hund-e-hütte / Kind-er-garten: m/n).
LINKERS = {
    "": "mfn",
    "s": "mn",
    "es": "mn",
    "n": "f",
    "en": "mf",
    "er": "mn",
    "e": "mn",
    "ns": "mn",
}
# Verb particles / prepositions: Ausweisung, Fortbildung are prefix derivations, not compounds.
PARTICLES = {"ab", "an", "auf", "aus", "bei", "durch", "ein", "fort", "gegen", "hinter", "los",
             "mit", "nach", "über", "um", "unter", "vor", "weg", "wider", "zu", "zurück", "zusammen",
             "miss", "hin", "her", "voran", "vorbei", "entgegen", "nieder", "wieder", "ur", "un"}  # fmt: skip
BORROWED = {"bor", "bor+", "lbor", "obor", "slbor", "ubor"}
INHERITED = {"inh", "inh+"}
FORMATION = {"compound", "af", "affix", "suffix", "prefix", "confix"}
GERMANIC = {"gmh", "goh", "gem-pro", "gmw-pro", "gml", "odt", "osx", "gem", "ine-pro", "de"}


def german_verbs() -> set[str]:
    out = set()
    with open(dump_path("de"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            w = r.get("word", "")
            if r.get("lang_code") == "de" and r.get("pos") == "verb" and " " not in w:
                out.add(w.lower())
    return out


def etymology(words: set[str]) -> dict[str, str]:
    """lemma -> loan / native / unknown, from the first decisive etymology template."""
    out: dict[str, str] = {}
    with open(dump_path("de"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            w = r.get("word", "")
            if r.get("lang_code") != "de" or r.get("pos") != "noun" or w not in words:
                continue
            if out.get(w, "unknown") != "unknown":
                continue
            lab = "unknown"
            for t in r.get("etymology_templates", []):
                n, src = t.get("name", ""), t.get("args", {}).get("2", "")
                if n in BORROWED:
                    lab = "loan"
                    break
                if n in INHERITED:
                    lab = "native"
                    break
                if n in ("der", "der+") and lab == "unknown":
                    lab = "native" if src in GERMANIC else "loan"
                if n in FORMATION and lab == "unknown":
                    lab = "native"
            out[w] = lab
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


def gender_of(lex: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """lowercased lemma -> (lemma, gender) for nouns with a single m/f/n gender."""
    g = lex[lex.gender.isin(["m", "f", "n"])]
    amb = g.groupby(g.lemma.str.lower()).gender.nunique()
    amb = set(amb[amb > 1].index)
    return {
        w.lower(): (w, x) for w, x in zip(g.lemma, g.gender, strict=True) if w.lower() not in amb
    }


def split_compound(w: str, genders: dict) -> tuple[str, str, str, str] | None:
    """(head, head gender, first part, first-part gender or '') for the longest lexicon-noun head."""
    lw = w.lower()
    for i in range(3, len(lw) - 3):  # head >= 4 letters (Bettelei is not Bett + Lei)
        if lw[i:] in genders:
            head, hg = genders[lw[i:]]
            pre = lw[:i]
            if pre in PARTICLES:
                return None
            for link, allowed in LINKERS.items():
                stem = pre[: len(pre) - len(link)] if link else pre
                if (not link or pre.endswith(link)) and len(stem) >= 3 and stem in genders:
                    first, fg = genders[stem]
                    if fg in allowed and (link != "n" or stem.endswith("e")):
                        return head, hg, first, fg
            return head, hg, pre, ""
    return None


def exclusions(df: pd.DataFrame, nouns: set[str], verbs: set[str]) -> pd.Series:
    """Why each noun is excluded ('' = kept; 'compound' alone = goes to the compound sets)."""
    wn = _wordnet()
    chem = {wn.synset(r) for r in CHEM_ROOTS}
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
    for w, g, concept in zip(df.lemma, gloss, df.concept_en, strict=True):
        why = [
            r for r, cond in [
                ("compound", compound(w)), ("proper name", bool(PROPER.search(g))),
                ("diminutive", bool(DIMINUTIVE.search(w))), ("Ge-", w.startswith("Ge")),
                ("chemical", bool(CHEMICAL.search(g)) or _under(concept, wn, chem)),
                ("animate (wordnet)", _under(concept, wn, animate)),
                ("nominalised verb", nominalised(w)), ("phase 4-5 held out", w in held),
                ("phase 0 shot", w in shots),
            ] if cond
        ]  # fmt: skip
        out.append("; ".join(why))
    return pd.Series(out, index=df.index)


def multi() -> pd.DataFrame:
    s = pd.read_csv("data/stimuli/multi_gender_spec.csv", keep_default_na=False)
    s = s[(s.decision == "keep") & (s.lang == "de")]
    return s[["lemma", "type", "gloss_m", "gloss_f"]].assign(set="multi", gender="multi")


def compounds(base: pd.DataFrame, lex: pd.DataFrame) -> pd.DataFrame:
    """One compound per head, head gender = compound gender, head passes the simplex filters."""
    genders = gender_of(lex)
    rows = []
    for r in base.itertuples():
        sp = split_compound(r.lemma, genders)
        if sp and sp[1] == r.gender:
            rows.append({"lemma": r.lemma, "head": sp[0], "first": sp[2], "first_gender": sp[3]})
    c = base.merge(pd.DataFrame(rows), on="lemma")
    head_bad = c["head"].str.contains(DIMINUTIVE) | c["head"].str.startswith("Ge")
    c = c[~head_bad].copy()
    c["conflict"] = (c.first_gender != "") & (c.first_gender != c.gender)
    # One per head: prefer a conflict compound (the sharpest test), then the most frequent.
    c = c.sort_values(["conflict", "zipf"], ascending=False).drop_duplicates("head")
    heads = sorted(c["head"].unique())
    rng = np.random.default_rng(SEED)
    test_heads = set(rng.choice(heads, len(heads) // 2, replace=False))
    c["set"] = ["compound_test" if h in test_heads else "compound_train" for h in c["head"]]
    c["ending"] = c["head"].str.lower().str[-2:]
    return c


def build() -> pd.DataFrame:
    lex = load_lexicon("de")
    nouns = set(lex.lemma.str.lower())
    base = _base(lex).copy()
    base["exclude"] = exclusions(base, nouns, german_verbs())
    dropped = base[~base.exclude.isin(["", "compound"])]
    comp = compounds(base[base.exclude == "compound"], lex)
    base = base[base.exclude == ""].copy()
    base["set"] = ["suffix" if s else "simplex" for s in base.de_suffix]
    # Suffix test group: up to N_SUFFIX frequent nouns per suffix (it's a test set; the full
    # ~1,700 would only slow the known check).
    suf = base[(base.set == "suffix") & (base.zipf >= 3.0)]
    suf = pd.concat(
        g.sample(min(N_SUFFIX, len(g)), random_state=SEED) for _, g in suf.groupby("de_suffix")
    )
    base = pd.concat([base[base.set == "simplex"], suf])
    base["ending"] = base.lemma.str[-2:]
    pool = flags(pd.concat([base, comp], ignore_index=True))
    ety = etymology(set(pool.lemma))
    pool["loan"] = pool.lemma.map(ety).fillna("unknown")
    pool = pd.concat([pool, multi()], ignore_index=True)
    pool["lang"] = "de"
    pool["source"] = source_tag("de")
    cols = ["lang", "lemma", "gender", "concept_en", "gloss", "set", "de_suffix", "ending", "loan",
            "zipf", "n_tokens_qwen3", "en_overlap", "sex_assoc", "head", "first", "first_gender",
            "conflict", "type", "gloss_m", "gloss_f", "source"]  # fmt: skip
    pool = pool.reindex(columns=cols)
    pool.to_csv(POOL, index=False)
    print(pool.groupby(["set", "gender"]).size().unstack(fill_value=0).to_string())
    s = _clean(pool[pool.set == "simplex"])
    print("clean simplex by loan status:")
    print(pd.crosstab(s.gender, s.loan).to_string())
    c = pool[pool.set.str.startswith("compound")]
    print("compound conflicts:", c.groupby("set").conflict.sum().to_dict())
    print("excluded by reason (first reason):")
    print(dropped.exclude.str.split("; ").str[0].value_counts().to_string())
    return pool
