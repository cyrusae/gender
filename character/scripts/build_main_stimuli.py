"""Main-study stimulus builder (DRAFT until the instruct pre-registration is signed off; 2026-10-27).

Inputs (all fixed earlier, no model involved):
- nouns: character/results/confound/pool_191_DRAFT.csv (191 nouns; translations + Wiktionary genders)
- modifiers: PI list v2 (character/stimuli/modifiers_v2_labeled_draft_provisional.csv, 70 + 4 plain)
  + wugs (character/stimuli/modifiers_v2_wugs.csv, 6 study wugs + frindle extra)
- mass nouns: the 24 adopted 2026-10-26 (docs/decisions.md)

Rules:
- a/an from the noun's sound: vowel letter -> "an", except YOO_START (unit...); h is "a" except
  SILENT_H. Fixed-article modifiers keep the PI's article (it belongs to the adjective).
- mass nouns drop every indefinite article ("great electricity", "electricity familiar to you",
  plain = "electricity"); the/my/your frames unchanged.
- postnominal pair: "{article} {noun} familiar to you", article from the noun.
- idiom flags carried from the PI's notes (confirmed, real, perfect, unnatural, common): the coder
  checks outputs for the idiom (e.g. "bachelor" after "confirmed").
Writes character/stimuli/main_character_v1_DRAFT.csv (noun x modifier phrase) and
character/stimuli/main_writer_v1_DRAFT.csv (noun x language foreign word in everyday spelling).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from gen import LANG_NAMES, plain_spelling

MASS = {"pollution", "nature", "tennis", "lightning", "socialism", "physics", "carbon",
        "electricity", "propaganda", "statistics", "mucus", "aviation", "insurance", "diabetes",
        "poverty", "terrorism", "cleanliness", "concrete", "east", "south", "dust", "communism",
        "milk", "zinc"}  # fmt: skip
YOO_START = {"unit", "university", "union", "uniform", "user", "utensil", "euro", "one"}
SILENT_H = {"hour", "honor", "honour", "heir", "honesty"}
IDIOMS = {"confirmed": "bachelor", "real": "real (man|woman|lady|gentleman)",
          "perfect": "perfect (gift|match|stranger|storm)", "unnatural": "unnatural (act|acts)",
          "common": "common (woman|girl|man)"}  # fmt: skip


def indef(noun: str) -> str:
    w = noun.lower()
    if w in SILENT_H:
        return "an"
    if w in YOO_START:
        return "a"
    return "an" if w[0] in "aeiou" else "a"


def phrase(article: str, word: str, tmpl: str, noun: str) -> str:
    mass = noun in MASS
    if word == "none:":  # plain forms
        if tmpl.startswith("{a/an}"):
            return noun if mass else f"{indef(noun)} {noun}"
        return tmpl.replace("{the}", "the").replace("{noun}", noun)
    if tmpl.startswith("{article}"):  # postnominal
        rest = tmpl.replace("{article} ", "").replace("{noun}", noun)
        return rest if mass else f"{indef(noun)} {rest}"
    p = tmpl.replace("{noun}", noun)
    if mass and article in ("a", "an"):
        p = p.split(" ", 1)[1]
    return p


def main() -> None:
    pool = pd.read_csv(HERE / "results/confound/pool_191_DRAFT.csv", keep_default_na=False)
    v2 = pd.read_csv(
        HERE / "stimuli/modifiers_v2_labeled_draft_provisional.csv", keep_default_na=False
    )
    v2 = v2[v2.word != ""]
    wugs = pd.read_csv(HERE / "stimuli/modifiers_v2_wugs.csv", keep_default_na=False)
    mods = [(r.article, r.word, r.phrase, "pi_v2") for r in v2.itertuples()]
    mods += [
        ("a", r.wug, r.text, "wug_extra" if r.wug == "frindle" else "wug")
        for r in wugs.itertuples()
    ]
    rows = []
    for n in pool.en:
        for art, word, tmpl, src in mods:
            if word == "none:":
                key = "plain_" + tmpl.split()[0].strip("{}").replace("/", "")
            elif src.startswith("wug"):
                key = f"wug_{word}"
            else:  # article kept in the key: "the ideal" and "an ideal" are separate modifiers
                key = f"{art} {word}".strip()
            idiom = IDIOMS.get(word, "")
            rows.append({"noun": n, "modifier": key, "source": src, "mass": n in MASS,
                         "mod_noun": phrase(art, word, tmpl, n), "idiom_regex": idiom})  # fmt: skip
    ch = pd.DataFrame(rows)
    assert not ch.duplicated(["noun", "modifier"]).any()
    assert not ch.mod_noun.str.contains(r"\{|\}").any()
    ch.to_csv(HERE / "stimuli/main_character_v1_DRAFT.csv", index=False)
    wr = [{"en": r["en"], "lang": lg, "word": plain_spelling(r[f"{lg}_word"], lg),
           "gender": r.get(f"{lg}_g", "") if lg in ("de", "ru", "es", "hi", "ar") else ""}
          for _, r in pool.iterrows() for lg in LANG_NAMES]  # fmt: skip
    pd.DataFrame(wr).to_csv(HERE / "stimuli/main_writer_v1_DRAFT.csv", index=False)
    print(
        f"character: {len(ch)} rows ({ch.noun.nunique()} nouns x {ch.modifier.nunique()} modifiers)"
    )
    print(f"writer: {len(wr)} noun x language words")


if __name__ == "__main__":
    main()
