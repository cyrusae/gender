"""Russian word pool for Phase 4 (third-language test): masculine vs feminine nouns in -ь.

Outside -ь, Russian gender follows the ending almost perfectly (-а/-я f, consonant m, -о/-е n),
so the only spelling-matched masculine/feminine comparison is inside the soft-sign cell
(день m "day" / ночь f "night"). Filters carried over from German (phase3_stimuli) where they
apply, plus Russian-specific ones:

- single gender m/f, ends in -ь, inanimate by BOTH Wiktionary's grammatical animacy tag and our
  English-gloss animacy check (lexicon), no other part of speech, no register marks, declinable
- gender-predicting suffixes inside -ь dropped: -ость/-есть (abstract nouns, always f),
  sibilant + ь (ночь, речь: always f), -тель (instrument/agent nouns, mostly m)
- semantic classes with one gender: month names (all m)
- place names, chemicals, animals/people/groups (WordNet, two senses; gloss), sex-typed garments
- formations from another -ь noun (полуось "half-axle" <- ось "axle"): the gender comes from the
  base noun, as German particle + noun formations
- loan status from etymology templates (inherited/derived from Slavic = native; borrowed = loan,
  except from Old Church Slavonic, whose borrowings are old and native-looking: native)

Cells = last two letters (consonant + ь) x loan status; only cells with both genders are kept.
Gold genders come from Wiktionary (kaikki.org), never from a language model.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from .lexicon import _wordnet, dump_path, load_lexicon
from .phase3_stimuli import (
    ANIMATE_GLOSS,
    ANIMATE_ROOTS,
    BORROWED,
    CHEM_ROOTS,
    CHEMICAL,
    FORMATION,
    GROUP_ROOTS,
    INHERITED,
    PROPER,
    SEX_EXTRA,
    _under,
)

POOL = "data/stimuli/phase4_ru_pool_v1.csv"
ZMIN = 2.0
SLAVIC = {"sla-pro", "orv", "zle-ort", "zle-mru", "zle-ono", "cu", "ine-pro", "ru"}
MONTHS = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)


def suffix_class(w: str) -> str:
    if w.endswith(("ость", "есть")):
        return "-ость"
    if re.search(r"[жчшщ]ь$", w):
        return "sibilant+ь"
    if w.endswith("тель"):
        return "-тель"
    return ""


def etymology_ru(words: set[str], nouns: set[str]) -> dict[str, tuple[str, str]]:
    """lemma -> (loan / native / unknown, base noun if formed from another lexicon noun)."""
    out: dict[str, tuple[str, str]] = {}
    with open(dump_path("ru"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            w = r.get("word", "")
            if r.get("lang_code") != "ru" or r.get("pos") != "noun" or w not in words:
                continue
            if out.get(w, ("unknown", ""))[0] != "unknown":
                continue
            lab, base = "unknown", ""
            for t in r.get("etymology_templates", []):
                n, args = t.get("name", ""), t.get("args", {})
                src = args.get("2", "")
                if n in FORMATION:
                    parts = {  # affixes are written with a hyphen (-ость, полу-): not base nouns
                        re.sub("[́̀]", "", v).lower()
                        for k, v in args.items()
                        if k != "1" and isinstance(v, str) and "-" not in v
                    }
                    hits = sorted((parts & nouns) - {w})
                    if hits:
                        base = hits[0]
                if n in BORROWED and lab == "unknown":
                    # Church Slavonic borrowings (жизнь, время) are old and look native: native.
                    lab = "native" if src == "cu" else "loan"
                elif n in INHERITED and lab == "unknown":
                    lab = "native"
                elif n in ("der", "der+") and lab == "unknown":
                    lab = "native" if src in SLAVIC else "loan"
                elif n in FORMATION and lab == "unknown":
                    lab = "native"
            out[w] = (lab, base)
    return out


def build() -> pd.DataFrame:
    lex = load_lexicon("ru")
    nouns = set(lex.lemma)
    soft = lex[lex.gender.isin(["m", "f"]) & (lex.ru_ending == "soft") & (lex.zipf >= ZMIN)].copy()
    wn = _wordnet()
    chem = {wn.synset(r) for r in CHEM_ROOTS}
    animate = {wn.synset(r) for r in ANIMATE_ROOTS}
    group = {wn.synset(r) for r in GROUP_ROOTS}
    ety = etymology_ru(set(soft.lemma), nouns)
    soft["loan"] = soft.lemma.map(lambda w: ety.get(w, ("unknown", ""))[0])
    soft["base_noun"] = soft.lemma.map(lambda w: ety.get(w, ("unknown", ""))[1])
    soft["suffix"] = soft.lemma.map(suffix_class)
    why = []
    for r in soft.itertuples():
        g = f"{r.gloss} {r.concept_en}"
        reasons = [
            name for name, cond in [
                ("grammatically animate", r.ru_animacy != "inan"),
                ("animate (lexicon)", r.animacy != "inanimate"),
                ("other part of speech", isinstance(r.also_pos, str) and r.also_pos != ""),
                ("register mark", isinstance(r.marked, str) and r.marked != ""),
                ("indeclinable", bool(r.ru_indecl)),
                (f"suffix {r.suffix}", r.suffix != ""),
                ("month name", bool(MONTHS.search(g))),
                # "The FIFA World Cup" (мундиаль, PI review): a gloss naming one specific thing
                ("proper name", bool(PROPER.search(g)) or bool(re.match(r"(The|the) [A-Z]", g))),
                ("chemical", bool(CHEMICAL.search(g)) or _under(r.concept_en, wn, chem)),
                ("animate (wordnet)", _under(r.concept_en, wn, animate, senses=2)),
                ("group (wordnet)", _under(r.concept_en, wn, group)),
                # English-gloss evidence alone (Latin plant names, "fish" in "net for fishing")
                # is overridden when Russian grammar marks the noun inanimate (PI review
                # 2026-10-08: plants and objects were dropped; Russian animacy is grammatical).
                ("animate (gloss)", bool(ANIMATE_GLOSS.search(g)) and r.ru_animacy != "inan"),
                ("sex-typed garment", bool(SEX_EXTRA.search(g))),
                (f"formed from {r.base_noun}", r.base_noun != ""),
            ] if cond
        ]  # fmt: skip
        why.append("; ".join(reasons))
    soft["excluded"] = why
    soft["cell"] = soft.lemma.str[-2:] + "|" + soft.loan
    keep = soft[soft.excluded == ""]
    both = keep.groupby("cell").gender.nunique()
    soft["set"] = ""
    ok = (soft.excluded == "") & soft.cell.isin(both[both == 2].index)
    soft.loc[ok, "set"] = "strat_ru"
    cols = ["lang", "lemma", "gender", "concept_en", "gloss", "zipf", "loan", "cell", "suffix",
            "base_noun", "ru_animacy", "animacy", "set", "excluded"]  # fmt: skip
    out = soft[cols].sort_values(
        ["set", "cell", "gender", "zipf"], ascending=[False, True, True, False]
    )
    out.to_csv(POOL, index=False)
    return out


def review_sheet(df: pd.DataFrame, path: str) -> None:
    s = df[df.set == "strat_ru"]
    g = {"m": "masc", "f": "fem"}
    lines = [
        "# Russian -ь nouns for Phase 4: English review sheet",
        "",
        "*Generated from `" + POOL + "` (branch `russian`). Genders and glosses come from "
        "Wiktionary (kaikki.org); nothing here comes from a language model. Every noun ends in "
        "the soft sign -ь, the one Russian ending shared by masculine and feminine nouns, so "
        "spelling can't give the gender away.*",
        "",
        (
            "**What to check (English only):** flag the row number of any noun that names a person, "
            "an animal, a group of people or animals, a place, or a month/day, or whose gloss looks "
            "like a mistranslation. You don't need to read Russian."
        ),
        "",
        (
            f"Kept: {(s.gender == 'm').sum()} masculine / {(s.gender == 'f').sum()} feminine in "
            f"{s.cell.nunique()} cells (last two letters x loan status, both genders present). "
            f"Excluded by the filters: {(df.excluded != '').sum()} (reasons in the CSV)."
        ),
        "",
        "| # | gloss | gender | cell | loan |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(s.itertuples(), 1):
        gl = re.sub(r"\s+", " ", str(r.gloss))[:100]
        lines.append(f"| {i} | {gl} | {g[r.gender]} | {r.cell.split('|')[0]} | {r.loan} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
