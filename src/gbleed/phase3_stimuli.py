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
  - a second animacy check by WordNet (person/animal, not plants; first two senses, since the
    first can be the meat or fur: quail, sable) and by gloss ("person who", species names, "pack of"), and
    a group check on the first sense (social/animal/military groups: das Heer).
  - no -er agent/instrument nouns (Bohrer <- bohren): masculine by derivation and often
    person-or-tool ambiguous.
  - particle + noun formations (Abwasser, Zuspiel, Unverständnis) count as compounds, and are then
    dropped as derivations (their gender comes from the embedded noun).
  - extra sex-typed garments (bodice, corset...) beyond the Phase 2 filter.
  - stratification cells: a shared two-gender suffix (-nis, -sal, -tum, -ment) where present, else
    the last two letters (cell_ending).
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

POOL = "data/stimuli/phase3_pool_v4.csv"
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
# Suffixes shared by two genders (die Erlaubnis / das Ergebnis; das Schicksal / die Trübsal; der
# Irrtum / das Eigentum; das Dokument / der Moment): a noun ending in one is put in that suffix's
# cell rather than its last-two-letter cell, so e.g. -nis neuters aren't compared with -eis
# masculines (der Kreis) as if spelling were matched.
# -sel (das Anhängsel, das Überbleibsel) is a neuter derivational suffix inside the -el cell
# (der Mantel, die Gabel), so it also gets its own cell.
MIXED_SUFFIXES = ["nis", "sal", "tum", "ment", "sel"]
# Sex-typed garments WordNet doesn't place under the Phase 2 roots (Mieder "bodice").
SEX_EXTRA = re.compile(
    r"\b(bodice|corset|girdle|petticoat|blouse|negligee|garter)\b", re.IGNORECASE
)
CHEM_ROOTS = ["chemical_element.n.01", "compound.n.02", "alloy.n.01"]
# Second animacy check: the lexicon tag misses some person nouns (Individuum, Bajazzo).
# People and animals, not plants (organism.n.01 caught Kartoffel "potato", Zwiebel "onion").
ANIMATE_ROOTS = ["person.n.01", "animal.n.01", "people.n.01"]
# Groups of people or animals (das Heer "army", das Rudel "pack"): not inanimate for our purposes.
GROUP_ROOTS = ["social_group.n.01", "animal_group.n.01", "military_unit.n.01"]
# Gloss evidence of an animate or group referent (first WordNet senses can be the meat or fur:
# "quail", "sable"; and compounds with an unlisted person head: Widerstandskämpfer).
ANIMATE_GLOSS = re.compile(
    r"\b(person who|someone who|one who|agent noun|species|genus|bird|mammal|insect|fish|"
    r"pack of|pack \(of|herd|flock|swarm|troop|army|crowd)\b|\([A-Z][a-z]+ [a-z]+\)",
)
# PI review 2026-10-07: every one of these must be excluded by the rules above (asserted).
PI_FLAGGED = {"Heer", "Rudel", "Wachtel", "Zobel", "Bohrer", "Kehrer", "Rasierer",
              "Widerstandskämpfer"}  # fmt: skip
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


def cell_ending(w: str) -> str:
    """Stratification cell key: a shared two-gender suffix if the noun has one, else the last two
    letters."""
    w = w.lower()
    for suf in MIXED_SUFFIXES:
        if w.endswith(suf) and len(w) >= len(suf) + 2:
            return "-" + suf
    return w[-2:]


def _under(concept: str, wn, roots, senses: int = 1) -> bool:
    for syn in wn.synsets(str(concept).replace(" ", "_"), pos="n")[:senses]:
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
    group = {wn.synset(r) for r in GROUP_ROOTS}

    def agent(w: str) -> bool:
        # -er agent/instrument nouns (Bohrer <- bohren, Rasierer <- rasieren): masculine by
        # derivation (a spelling cue inside the -er cell) and often person-or-tool ambiguous.
        # Only stem + -en is tested, so nouns that verbs were made from (Leder -> ledern) stay.
        w = w.lower()
        return w.endswith("er") and w[:-2] + "en" in verbs

    def nominalised(w: str) -> bool:
        w = w.lower()
        return w.endswith("n") and any(w[i:] in verbs for i in range(len(w) - 2))

    def compound(w: str) -> bool:
        w = w.lower()
        if any(w.startswith(p) and w[len(p) :] in nouns for p in PARTICLES if len(w) - len(p) >= 3):
            return True  # particle + noun (Abwasser, Zuspiel, Unverständnis): gender from the noun
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
                ("animate (wordnet)", _under(concept, wn, animate, senses=2)),
                ("group (wordnet)", _under(concept, wn, group)),
                ("animate (gloss)", bool(ANIMATE_GLOSS.search(g))), ("agent noun", agent(w)),
                ("nominalised verb", nominalised(w)), ("phase 4-5 held out", w in held),
                ("phase 0 shot", w in shots), ("sex-typed garment", bool(SEX_EXTRA.search(g))),
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
    c["ending"] = c["head"].map(cell_ending)
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
    base["ending"] = base.lemma.map(cell_ending)
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
    assert not set(pool.lemma) & PI_FLAGGED, set(pool.lemma) & PI_FLAGGED
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
