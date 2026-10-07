"""Phase 2 stimulus pools (Spanish). Every noun here goes through the Phase 0 "known" check
(all three Qwen sizes) before final sampling; see sample_phase2().

  stratum      TRAIN candidates (v4): every clean noun with no -o/-a/-á/-ó and no gender-predicting
               suffix; cells = final two letters, or -ete/-ote/-men (cell_ending). phase2.finalize
               draws the stratified primary set ("strat": every known noun in cells with both
               genders, compared within cells) and the nested equal-count set ("matched": equal m
               and f per cell, la llave / el puente) after the known filter.
  regular      TRAIN (comparison) and TEST: regular -o masculine / -a feminine nouns.
  exception    TEST ONLY: ending points to the wrong gender (el problema, la foto, el día, la mano).
  homograph    TEST ONLY: noun that is also a verb form, read in a noun frame ("mi camino") and a
               verb frame ("siempre camino" 1sg, "siempre cuenta" 3sg): the same pronoun-free frame for
               every item (v2 used yo/usted, which the verb ending forced, so the pronoun gave away
               the gender; see decisions.md).
  multi        TEST ONLY: Spanish same-spelling two-gender items (multi_gender_spec.csv).
"""

from __future__ import annotations

import json
import re

import pandas as pd

from .lexicon import dump_path, eligible, load_lexicon, source_tag

OUT = "data/stimuli/phase2_pool_v4.csv"
PREV = "data/stimuli/phase2_pool_v3.csv"  # regular + homograph sets carried over from here
# Pro-drop adverb frame, identical for 1sg and 3sg: licenses a finite verb, not a bare noun, and
# isn't an English word (unlike "no").
VERB_FRAME = "siempre {w}"
# Endings that predict gender in Spanish, excluded from the matched set (plus -o/-a themselves).
PREDICTIVE = re.compile(
    r"(?:ción|sión|dad|tad|tud|umbre|ez|eza|itis|sis|or|aje|án|ón|ín|ista|ante|ente|ie|ud|o|a|á|ó)$"
)
SEED = 0
# Sex-associated concepts (sexual anatomy, sex-typed clothing/cosmetics) carry SOCIAL gender; a
# grammatical-gender direction trained on them would already contain it (circular for Phase 5).
SEX_GLOSS = re.compile(
    r"\b(woman|women|female|feminine|male|men's|man's|masculine|ladies|girl|boy|breast|bra|"
    r"lingerie|menstrua\w*|pregnan\w*|vagin\w*|vulva|penis|phallus|testic\w*|clitor\w*|scrot\w*|"
    r"pubi\w*|ovar\w*|uter\w*|womb|semen|sperm|beard|moustache|mustache|skirt|dress|tie|lipstick)\b",
    re.IGNORECASE,
)
SEX_ROOTS = ["genitalia.n.01", "reproductive_organ.n.01", "garment.n.01", "undergarment.n.01",
             "jewelry.n.01", "cosmetic.n.01", "makeup.n.01"]  # fmt: skip


def flags(df: pd.DataFrame) -> pd.DataFrame:
    """en_overlap: ~ the English word, or a common English word. sex_assoc: why, or ''."""
    import difflib

    from wordfreq import zipf_frequency

    from .lexicon import EN_HOMOGRAPH_ZIPF, EN_SAME, _plain, _wordnet

    wn = _wordnet()
    roots = {wn.synset(r) for r in SEX_ROOTS}

    def sex(concept: str, gloss: str) -> str:
        if SEX_GLOSS.search(gloss or "") or SEX_GLOSS.search(concept or ""):
            return "gloss"
        head = (concept or "").split(" of ")[0].split()[-1:] or [""]
        for c in (str(concept).replace(" ", "_"), head[0]):
            for syn in wn.synsets(c, pos="n")[:1]:
                hyp = {h for path in syn.hypernym_paths() for h in path}
                hit = [r.name() for r in roots if r in hyp]
                if hit:
                    return "wordnet:" + hit[0]
        return ""

    df = df.copy()
    concept = df.concept_en.fillna("").astype(str)
    df["en_overlap"] = [
        difflib.SequenceMatcher(None, _plain(w), _plain(c)).ratio() >= EN_SAME
        or zipf_frequency(w, "en") >= EN_HOMOGRAPH_ZIPF
        for w, c in zip(df.lemma, concept, strict=True)
    ]
    df["sex_assoc"] = [
        sex(c, g) for c, g in zip(concept, df.gloss.fillna("").astype(str), strict=True)
    ]
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Training sets exclude English-overlapping and sex-associated nouns."""
    return df[~df.en_overlap & (df.sex_assoc == "")]


def verb_forms() -> tuple[dict, dict]:
    """1sg and 3sg present indicative forms of every Spanish verb -> infinitive."""
    first, third = {}, {}
    with open(dump_path("es"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("lang_code") != "es" or r.get("pos") != "verb" or " " in r.get("word", ""):
                continue
            for fm in r.get("forms", []):
                t = set(fm.get("tags", []))
                if {"present", "indicative", "singular"} <= t and "vos-form" not in t:
                    if "first-person" in t:
                        first.setdefault(fm["form"], r["word"])
                    if "third-person" in t:
                        third.setdefault(fm["form"], r["word"])
    return first, third


# Masculine-by-rule derivational suffixes hiding inside two-letter cells (billete, camarote,
# examen): each gets its own cell, as German -nis/-sel do (phase3_stimuli.MIXED_SUFFIXES).
ES_SUFFIX_CELLS = ["ete", "ote", "men"]


def cell_ending(w: str) -> str:
    for suf in ES_SUFFIX_CELLS:
        if w.endswith(suf) and len(w) >= len(suf) + 2:
            return "-" + suf
    return w[-2:]


def semantic_exclusions(df: pd.DataFrame) -> pd.Series:
    """The Phase 3 (German) semantic filters, for automatically built sets: second animacy check
    (WordNet person/animal, two senses; gloss evidence), groups, chemicals (all masculine in
    Spanish: el sodio), place-name glosses, sex-typed garments. '' = kept."""
    from . import phase3_stimuli as p3
    from .lexicon import _wordnet

    wn = _wordnet()
    roots = {k: {wn.synset(r) for r in v} for k, v in
             (("animate", p3.ANIMATE_ROOTS), ("group", p3.GROUP_ROOTS), ("chem", p3.CHEM_ROOTS))}  # fmt: skip
    gloss = df.gloss.astype(str) + " " + df.concept_en.astype(str)
    out = []
    for g, c in zip(gloss, df.concept_en, strict=True):
        why = [r for r, cond in [
            ("animate (wordnet)", p3._under(c, wn, roots["animate"], senses=2)),
            ("group (wordnet)", p3._under(c, wn, roots["group"])),
            ("animate (gloss)", bool(p3.ANIMATE_GLOSS.search(g))),
            ("chemical", bool(p3.CHEMICAL.search(g)) or p3._under(c, wn, roots["chem"])),
            ("proper name", bool(p3.PROPER.search(g))),
            ("sex-typed garment", bool(p3.SEX_EXTRA.search(g))),
        ] if cond]  # fmt: skip
        out.append("; ".join(why))
    return pd.Series(out, index=df.index)


def _semantic(df: pd.DataFrame) -> pd.DataFrame:
    return df[semantic_exclusions(df) == ""]


def matched(el: pd.DataFrame, zmin: float = 2.0) -> pd.DataFrame:
    """All candidates for the spelling-controlled training sets (no sampling here): the final
    stratified and equal-count sets are drawn in phase2.finalize after the known filter."""
    n = _semantic(_clean(flags(el[(el.zipf >= zmin) & ~el.lemma.str.contains(PREDICTIVE)])))
    n["ending"] = n.lemma.map(cell_ending)
    return n.assign(set="stratum")


def regular(el: pd.DataFrame, n_per: int = 150, zmin: float = 3.0) -> pd.DataFrame:
    r = _semantic(_clean(flags(el[(el.es_regular == "yes") & (el.zipf >= zmin)])))
    return pd.concat(
        g.sample(min(n_per, len(g)), random_state=SEED) for _, g in r.groupby("gender")
    ).assign(set="regular")


def exceptions(lex: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    spec = pd.read_csv("data/stimuli/exceptions_spec.csv", comment="#", keep_default_na=False)
    lx = lex.set_index("lemma")
    rows, dropped = [], []
    for r in spec.itertuples():
        if isinstance(r.override_gender, str) and r.override_gender:
            base = lx.loc[r.lemma] if r.lemma in lx.index else pd.Series(dtype=object)
            base = base.iloc[0] if isinstance(base, pd.DataFrame) else base
            rows.append({**base.to_dict(), "lemma": r.lemma, "gender": r.override_gender,
                         "set": f"exception_{r.subgroup}", "override": r.override_reason,
                         "concept_en": r.override_gloss, "gloss": r.override_gloss})  # fmt: skip
            continue
        if r.lemma not in lx.index:
            dropped.append((r.lemma, "not in lexicon"))
            continue
        e = lx.loc[r.lemma]
        e = e.iloc[0] if isinstance(e, pd.DataFrame) else e
        why = [
            msg for cond, msg in [
                (e.gender not in ("m", "f"), f"genders={e.genders}"),
                (e.animacy == "animate", f"animate({e.animacy_reason})"),
                (bool(e.also_pos), f"also {e.also_pos}"),
                (bool(e.also_form), f"also {e.also_form}"),
                (bool(e.regions), f"regional {e.regions}"),
                (bool(e.marked), f"marked {e.marked}"),
            ] if cond
        ]  # fmt: skip
        expected = "f" if r.lemma[-1] in "oó" else "m"
        if e.gender in ("m", "f") and e.gender != expected:
            why.append(f"not an exception (gender {e.gender})")
        if why:
            dropped.append((r.lemma, "; ".join(why)))
            continue
        rows.append({**e.to_dict(), "lemma": r.lemma, "set": f"exception_{r.subgroup}"})
    return pd.DataFrame(rows), dropped


def homographs(lex: pd.DataFrame, n_per: int = 40, zmin: float = 3.0) -> pd.DataFrame:
    from wordfreq import zipf_frequency

    first, third = verb_forms()
    base = lex[
        lex.gender.isin(["m", "f"]) & (lex.animacy == "inanimate") & (lex.marked == "")
        & (lex.regions == "") & (lex.also_pos == "") & (lex.zipf >= zmin)
        & ~lex.initial_a_f.astype(str).eq("True")
    ]  # fmt: skip
    m = base[(base.gender == "m") & base.lemma.str.endswith("o") & base.lemma.isin(first)]
    f = base[(base.gender == "f") & base.lemma.str.endswith("a") & base.lemma.isin(third)]
    m = m.assign(verb=m.lemma.map(first), verb_frame=VERB_FRAME)
    f = f.assign(verb=f.lemma.map(third), verb_frame=VERB_FRAME)
    both = pd.concat([m, f])
    both = both[[zipf_frequency(v, "es") >= 3.0 for v in both.verb]]  # verb reading must be common
    both = _semantic(both)
    return pd.concat(
        g.sample(min(n_per, len(g)), random_state=SEED) for _, g in both.groupby("gender")
    ).assign(set="homograph")


def multi() -> pd.DataFrame:
    s = pd.read_csv("data/stimuli/multi_gender_spec.csv", keep_default_na=False)
    s = s[(s.decision == "keep") & (s.lang == "es")]
    return s[["lemma", "type", "gloss_m", "gloss_f"]].assign(set="multi", gender="multi", lang="es")


def build() -> pd.DataFrame:
    from .phase0 import LANG_CONFIG

    shots = {n for n, _ in LANG_CONFIG["es"]["shots"]}  # Phase 0 quiz examples can't be stimuli
    lex = load_lexicon("es")
    lex = lex[~lex.lemma.isin(shots)]
    el = eligible(lex)
    ex, dropped = exceptions(lex)
    # Test and comparison sets are carried over from the previous version minus whatever the new
    # filters flag, never redrawn, so old and new results differ only by known removals and can be
    # compared on shared items (v3 -> v4: side-agent catch).
    prev = pd.read_csv(PREV, keep_default_na=False)
    lx = el.set_index("lemma")
    carried = []
    for name in ("regular", "homograph"):
        old = prev[prev.set == name]
        cur = old.drop(columns=[c for c in ("en_overlap", "sex_assoc") if c in old]).copy()
        cur["concept_en"] = cur.concept_en.where(cur.concept_en != "", cur.lemma.map(lx.concept_en))
        carried.append(_semantic(cur))
    parts = [matched(el), *carried, ex]
    keep = ["lemma", "gender", "concept_en", "gloss", "zipf", "set", "verb", "verb_frame", "ending",
            "loan", "override"]  # fmt: skip
    from .phase3_stimuli import etymology

    pool = pd.concat([p.reindex(columns=keep) for p in parts])
    ety = etymology(set(pool.lemma), "es")
    pool["loan"] = pool.lemma.map(ety).fillna("unknown")
    pool = pool.drop_duplicates("lemma", keep="last")  # a test-set membership wins over training
    # Flags on every set (training sets were already filtered on them; test sets keep flagged
    # items so each test can be run with and without them).
    pool = flags(pool)
    pool["lang"] = "es"
    pool["source"] = source_tag("es")
    pool = pool[
        [
            "lang",
            "lemma",
            "gender",
            "concept_en",
            "set",
            "source",
            "en_overlap",
            "sex_assoc",
            *[c for c in keep if c not in ("lemma", "gender", "concept_en", "set")],
        ]
    ]
    pool.to_csv(OUT, index=False)
    multi().to_csv("data/stimuli/phase2_multi_v2.csv", index=False)
    print(pool.groupby(["set", "gender"]).size().unstack(fill_value=0).to_string())
    print("exceptions dropped:")
    for w, why in dropped:
        print(f"  {w}: {why}")
    return pool
