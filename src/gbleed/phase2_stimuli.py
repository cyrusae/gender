"""Phase 2 stimulus pools (Spanish). Every noun here goes through the Phase 0 "known" check
(all three Qwen sizes) before final sampling; see sample_phase2().

  matched      TRAIN (main): nouns whose ending carries no gender information by construction:
               no -o/-a/-á/-ó, no gender-predicting suffix, and equal numbers of m and f for each
               final two letters (la llave / el puente, la luz / el lápiz, la crisis / el análisis).
  regular      TRAIN (comparison) and TEST: regular -o masculine / -a feminine nouns.
  exception    TEST ONLY: ending points to the wrong gender (el problema, la foto, el día, la mano).
  homograph    TEST ONLY: noun that is also a verb form, read in a noun frame ("mi camino") and a
               verb frame ("yo camino" for 1sg -o, "usted cuenta" for 3sg -a; usted, not él, so the
               subject carries no gender).
  multi        TEST ONLY: Spanish same-spelling two-gender items (multi_gender_spec.csv).
"""

from __future__ import annotations

import json
import re

import pandas as pd

from .lexicon import dump_path, eligible, load_lexicon, source_tag

OUT = "data/stimuli/phase2_pool_v1.csv"
# Endings that predict gender in Spanish, excluded from the matched set (plus -o/-a themselves).
PREDICTIVE = re.compile(
    r"(?:ción|sión|dad|tad|tud|umbre|ez|eza|itis|sis|or|aje|án|ón|ín|ista|ante|ente|ie|ud|o|a|á|ó)$"
)
SEED = 0


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


def matched(el: pd.DataFrame, zmin: float = 2.5) -> pd.DataFrame:
    n = el[(el.zipf >= zmin) & ~el.lemma.str.contains(PREDICTIVE)].copy()
    n["ending"] = n.lemma.str[-2:]
    parts = []
    for _, g in n.groupby("ending"):
        k = min((g.gender == "m").sum(), (g.gender == "f").sum())
        if k:
            for _, gg in g.groupby("gender"):
                parts.append(gg.sample(k, random_state=SEED))
    return pd.concat(parts).assign(set="matched")


def regular(el: pd.DataFrame, n_per: int = 150, zmin: float = 3.0) -> pd.DataFrame:
    r = el[(el.es_regular == "yes") & (el.zipf >= zmin)]
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
    m = m.assign(verb=m.lemma.map(first), verb_frame="yo {w}")
    f = f.assign(verb=f.lemma.map(third), verb_frame="usted {w}")
    both = pd.concat([m, f])
    both = both[[zipf_frequency(v, "es") >= 3.0 for v in both.verb]]  # verb reading must be common
    return pd.concat(
        g.sample(min(n_per, len(g)), random_state=SEED) for _, g in both.groupby("gender")
    ).assign(set="homograph")


def multi() -> pd.DataFrame:
    s = pd.read_csv("data/stimuli/multi_gender_spec.csv", keep_default_na=False)
    s = s[(s.decision == "keep") & (s.lang == "es")]
    return s[["lemma", "type", "gloss_m", "gloss_f"]].assign(set="multi", gender="multi", lang="es")


def build() -> pd.DataFrame:
    lex = load_lexicon("es")
    el = eligible(lex)
    ex, dropped = exceptions(lex)
    parts = [matched(el), regular(el), ex, homographs(lex)]
    keep = ["lemma", "gender", "concept_en", "gloss", "zipf", "set", "verb", "verb_frame", "ending",
            "override"]  # fmt: skip
    pool = pd.concat([p.reindex(columns=keep) for p in parts])
    pool = pool.drop_duplicates("lemma", keep="last")  # a test-set membership wins over training
    # English overlap flag (Phase 2 is monolingual, so flagged, not excluded): the lemma is ~ the
    # English word (similarity to the English gloss >= 0.9) or a common English word (zipf >= 3).
    import difflib

    from wordfreq import zipf_frequency

    from .lexicon import EN_HOMOGRAPH_ZIPF, EN_SAME, _plain

    pool["en_overlap"] = [
        difflib.SequenceMatcher(None, _plain(w), _plain(c)).ratio() >= EN_SAME
        or zipf_frequency(w, "en") >= EN_HOMOGRAPH_ZIPF
        for w, c in zip(pool.lemma, pool.concept_en.fillna(""), strict=True)
    ]
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
            *[c for c in keep if c not in ("lemma", "gender", "concept_en", "set")],
        ]
    ]
    pool.to_csv(OUT, index=False)
    multi().to_csv("data/stimuli/phase2_multi_v1.csv", index=False)
    print(pool.groupby(["set", "gender"]).size().unstack(fill_value=0).to_string())
    print("exceptions dropped:")
    for w, why in dropped:
        print(f"  {w}: {why}")
    return pool
