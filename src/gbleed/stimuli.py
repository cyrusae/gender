"""Stimulus list loading and validation.

Long format, one row per (language, noun):
    lang        de | es
    lemma       bare noun, no article (e.g. Brücke, puente)
    gender      m | f   (neuter is out of scope)
    concept_en  English gloss; rows sharing a concept_en form a cross-language pair
    set         free-form tag (e.g. flipped, control, suffix_ung)
    source      where the gold gender label came from
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED = ["lang", "lemma", "gender", "concept_en", "set", "source"]
LANGS = {"de", "es"}
GENDERS = {"m", "f"}


def load_stimuli(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, comment="#")
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    for col in ("lang", "lemma", "gender"):
        df[col] = df[col].str.strip()
    bad_lang = df[~df.lang.isin(LANGS)]
    bad_gender = df[~df.gender.isin(GENDERS)]
    if len(bad_lang) or len(bad_gender):
        raise ValueError(
            f"{path}: bad lang rows {bad_lang.index.tolist()}, "
            f"bad gender rows {bad_gender.index.tolist()}"
        )
    dupes = df[df.duplicated(["lang", "lemma"], keep=False)]
    if len(dupes):
        raise ValueError(f"{path}: duplicate (lang, lemma): {dupes.lemma.tolist()}")
    return df.reset_index(drop=True)


def flipped_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Concepts present in both languages with opposite gender.

    Only rows tagged set=flipped/control are paired, so unrelated nouns that
    happen to share an English gloss aren't counted."""
    df = df[df["set"].isin(["flipped", "control"])]
    if df.empty:
        return pd.DataFrame(
            columns=["concept_en", "de_lemma", "es_lemma", "de_gender", "es_gender"]
        )
    wide = df.pivot_table(
        index="concept_en", columns="lang", values=["lemma", "gender"], aggfunc="first"
    ).dropna()
    wide.columns = [f"{lang}_{field}" for field, lang in wide.columns]
    return wide[wide.de_gender != wide.es_gender].reset_index()
