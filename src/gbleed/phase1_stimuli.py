"""Phase 1 stimuli: what the spelling eraser is trained and validated on.

- verbs: regular Spanish -ar verbs as 1sg/3sg present pairs (hablo/habla). The -o/-a
  ending marks *person*, not gender, so an eraser trained to remove it targets spelling.
  Forms come from Wiktionary's conjugation tables; a pair is kept only if both forms
  follow the regular pattern (rules out pensar -> pienso) and neither form is also a
  noun or adjective (canto "song", trabajo "work": there -o/-a would be gender).
- nonce: invented Spanish-looking minimal pairs, stem + a / stem + o (breld-a/breld-o),
  checked to be absent from the Spanish Wiktionary (every word and inflected form) and to
  have zero frequency in Spanish, English, Portuguese, Italian, French and German.
  Split train/test by stem, fixed seed (CLAUDE.md held-out rule).
- nouns: regular inanimate nouns (-o masculine, -a feminine) from the lexicon, a
  diagnostic for "did the eraser remove gender too?" (design doc pitfall 1).
"""

from __future__ import annotations

import itertools
import json
import random
import unicodedata
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .lexicon import dump_path, eligible, load_lexicon, source_tag

OUT_DIR = Path("data/stimuli")
NONCE_LANGS = ("es", "en", "pt", "it", "fr", "de")

# Spanish-plausible syllable parts for nonce stems: (C)(C)V(C) + C(C) + a/o.
ONSETS = ["b", "br", "bl", "c", "cr", "cl", "d", "dr", "f", "fr", "fl", "g", "gr", "gl",
          "l", "m", "n", "p", "pr", "pl", "r", "s", "t", "tr", "v", "ch", "j"]  # fmt: skip
VOWELS = ["a", "e", "i", "o", "u"]
CODAS = ["", "l", "n", "r", "s"]
# Last consonant(s) before the ending. No "m" (keeps -ma, the Greek masculine pattern, out)
# and no "c"/"g" (spelling changes before -o/-a would be irrelevant, but avoid ambiguity).
# Which final consonants may follow each coda inside a Spanish word
# (al-ba, fal-da, car-ta, per-la, can-ta, as-no, is-la, pas-ta). No 3-consonant clusters.
_SINGLE = {"b", "d", "f", "l", "n", "p", "r", "s", "t", "v", "ch", "ll"}
MEDIAL_OK = {
    "": None,  # filled below: anything
    "l": {"b", "d", "f", "p", "t", "v", "ch", "s"},
    "n": {"d", "f", "t", "s", "ch", "v"},
    "r": {"b", "d", "f", "l", "n", "p", "s", "t", "v", "ch"},
    "s": {"l", "n", "p", "t"},
}
FINAL_ONSETS = ["b", "d", "f", "l", "n", "p", "r", "s", "t", "v", "ch", "ll",
                "br", "dr", "pr", "tr", "bl", "pl"]  # fmt: skip


MEDIAL_OK[""] = set(FINAL_ONSETS)


def strip_accents(w: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", w)
        if unicodedata.category(c) != "Mn" or c == "\u0303"
    )  # keeps ñ (n + combining tilde) distinct from n


def scan_spanish():
    """All Spanish word forms, plus -ar verb 1sg/3sg present forms and the set of
    strings that exist as any non-verb word (lemma or inflected form)."""
    path = dump_path("es")
    all_forms, nominal, verbs = set(), set(), {}
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc="scan es", unit=" entries"):
            r = json.loads(line)
            if r.get("lang_code") != "es":
                continue
            word, pos = r.get("word", ""), r.get("pos")
            forms = {fm.get("form", "") for fm in r.get("forms", [])}
            all_forms.add(word.lower())
            all_forms |= {x.lower() for x in forms}
            if pos != "verb":  # nouns, adjectives, pronouns (aquella), determiners, ...
                # Accent-insensitive: casual text drops accents (termino ~ término).
                nominal.add(strip_accents(word.lower()))
                nominal |= {strip_accents(x.lower()) for x in forms}
            if pos == "verb" and word.endswith("ar") and " " not in word:
                one = three = None
                for fm in r.get("forms", []):
                    t = set(fm.get("tags", []))
                    if {"present", "indicative", "singular"} <= t and not t & {"vos-form"}:
                        if "first-person" in t and one is None:
                            one = fm["form"]
                        if "third-person" in t and three is None:
                            three = fm["form"]
                if one and three:
                    verbs.setdefault(word, (one, three))
    return all_forms, nominal, verbs


def build_verbs(verbs: dict, nominal: set, n: int) -> pd.DataFrame:
    from wordfreq import zipf_frequency

    rows = []
    for inf, (one, three) in verbs.items():
        stem = inf[:-2]
        if one != stem + "o" or three != stem + "a":
            continue  # irregular or stem-changing
        if strip_accents(one) in nominal or strip_accents(three) in nominal:
            continue  # canto (noun), aquella (pronoun): the ending would be gender
        if zipf_frequency(inf, "es") < 2.0:
            continue  # obscure lemma whose forms belong to a common verb (elijar ~ elegir)
        z1, z3 = zipf_frequency(one, "es"), zipf_frequency(three, "es")
        if min(z1, z3) < 2.0 or max(zipf_frequency(one, "en"), zipf_frequency(three, "en")) >= min(
            z1, z3
        ):
            continue
        rows.append({"lemma": inf, "stem": stem, "form_o": one, "form_a": three,
                     "zipf_o": z1, "zipf_a": z3})  # fmt: skip
    df = pd.DataFrame(rows)
    df["zipf_min"] = df[["zipf_o", "zipf_a"]].min(axis=1)
    df = df.sort_values("zipf_min", ascending=False).head(n).reset_index(drop=True)
    df["source"] = source_tag("es")
    return df


def build_nonce(all_forms: set, n: int, seed: int) -> pd.DataFrame:
    from wordfreq import zipf_frequency

    rng = random.Random(seed)
    cands = [
        o + v + c + fo
        for o, v, c, fo in itertools.product(ONSETS, VOWELS, CODAS, FINAL_ONSETS)
        if fo in MEDIAL_OK[c]
    ]
    rng.shuffle(cands)
    rows = []
    for stem in cands:
        fa, fo = stem + "a", stem + "o"
        if fa in all_forms or fo in all_forms:
            continue
        if any(zipf_frequency(w, lang) > 0 for w in (fa, fo) for lang in NONCE_LANGS):
            continue
        rows.append({"stem": stem, "form_a": fa, "form_o": fo})
        if len(rows) == n:
            break
    df = pd.DataFrame(rows)
    order = list(range(len(df)))
    rng.shuffle(order)
    df["split"] = [
        "train" if i < len(df) // 2 else "test"
        for i in sorted(range(len(df)), key=order.__getitem__)
    ]
    df["source"] = "generated; absent from es Wiktionary + zero wordfreq in " + "/".join(
        NONCE_LANGS
    )
    return df


def build_nouns(n_per_gender: int, seed: int) -> pd.DataFrame:
    el = eligible(load_lexicon("es"))
    el = el[(el.es_regular == "yes") & (el.zipf >= 3.0)]
    parts = [
        g.sample(min(n_per_gender, len(g)), random_state=seed) for _, g in el.groupby("gender")
    ]
    df = pd.concat(parts)[["lemma", "gender", "concept_en", "zipf"]].reset_index(drop=True)
    df["source"] = source_tag("es")
    return df


def build_all(version: int = 1, n_verbs: int = 300, n_nonce: int = 300, n_nouns: int = 150,
              seed: int = 0) -> dict:  # fmt: skip
    all_forms, nominal, verbs = scan_spanish()
    out = {
        "verbs": build_verbs(verbs, nominal, n_verbs),
        "nonce": build_nonce(all_forms, n_nonce, seed),
        "nouns": build_nouns(n_nouns, seed),
    }
    for name, df in out.items():
        p = OUT_DIR / f"phase1_{name}_v{version}.csv"
        df.to_csv(p, index=False)
        print(f"{name}: {len(df)} rows -> {p}")
    return out
