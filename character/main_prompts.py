"""Prompt builders for the first main run (prereg v1, adopted 2026-10-27).

Two experiments, each in two frames (Q5):
- writer_main: Writer frame = PI writer prompt, 8 languages x (foreign, English word) + plain
  (17 arms); Consider frame = the same prompt with the persona removed, 8 foreign + 1 English
  (9 arms). Word quoted, no article; English-word arms drop "or its English translation".
- character_main: Consider frame = PI's original "Consider {MOD_NOUN}..." prompt; Writer frame =
  "You are a writer planning a short story. " + the same prompt. Noun phrases from the frozen
  modifier rules (scripts/build_main_stimuli.phrase), so pilot nouns can be smoke-tested with the
  same code.
The `modifier` field is "<frame>|<arm or modifier key>" so every cell is unique; `frame` and `arm`
are also stored. Name instruction = `free_noted` (shared with the pilots).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts"))
from build_main_stimuli import IDIOMS, phrase

PERSONA = "You are a writer planning a short story. "


VOWELS = "aeiouáéíóúü"


def es_stressed_a(w: str) -> bool:
    """Spanish feminine nouns starting with stressed a/ha take "el" (el arma, el agua, el hacha).
    Stress from the spelling rules: an accent on the first vowel (área), or no written accent, two
    syllables, and a final vowel, n or s (stress then falls on the first of the two)."""
    import re

    x = w.lower()
    if re.match(r"h?á", x):
        return True
    if not re.match(r"h?a", x) or re.search("[áéíóú]", x):
        return False
    syll = len(re.findall(r"[aeiouü]+", x))  # vowel groups ~ syllables (au, ai count once)
    return syll == 2 and x[-1] in "aeiouns"


def citation_form(word: str, lang: str, gender: str) -> str:
    """German/Spanish word with its definite article (prereg v1 amendment 1, 2026-10-29)."""
    if lang == "de":
        return {"m": "der", "f": "die", "n": "das"}[gender] + " " + word
    if lang == "es":
        art = "el" if gender == "m" or es_stressed_a(word) else "la"
        return f"{art} {word}"
    raise ValueError(lang)


def _q(word: str) -> str:
    return f'"{word}"'


def build_writer_main(
    cfg: dict, nouns: list[str], trans: dict, lang_names: dict, genders: dict | None = None
) -> list[dict]:
    """genders: {noun: {lang: m/f/n}}; when given, de/es also get citation-form arms
    ("<lang>:citation", word with its definite article) in both frames."""
    instr = cfg["name_conditions"][cfg["name_condition"]]
    T = cfg["templates"]
    out = []
    for noun in nouns:
        tr = trans.get(noun, {})
        cells = [("writer", "plain", None, noun, T["writer_plain"])]
        for lg, fw in tr.items():
            cells += [("writer", f"{lg}:foreign", lg, fw, T["writer_foreign"]),
                      ("writer", f"{lg}:english", lg, noun, T["writer_english"]),
                      ("consider", f"{lg}:foreign", lg, fw, T["consider_foreign"])]  # fmt: skip
            g = (genders or {}).get(noun, {}).get(lg, "")
            if lg in ("de", "es") and g:
                cf = citation_form(fw, lg, g)
                cells += [("writer", f"{lg}:citation", lg, cf, T["writer_foreign"]),
                          ("consider", f"{lg}:citation", lg, cf, T["consider_foreign"])]  # fmt: skip
        cells.append(("consider", "english", None, noun, T["consider_english"]))
        for frame, arm, lg, word, t in cells:
            p = t.replace("{LANGUAGE}", lang_names.get(lg, "")).replace("{WORD}", _q(word))
            p = p.replace("{NAME_INSTR}", instr)
            out.append({"noun": noun, "modifier": f"{frame}|{arm}", "frame": frame, "arm": arm,
                        "mod_noun": word, "lang": lg or "", "foreign_word": tr.get(lg, "") if lg else "",
                        "prompt": p, "name_condition": cfg["name_condition"]})  # fmt: skip
    return out


def load_modifiers(cfg: dict) -> list[tuple[str, str, str, str]]:
    """(key, article, word, template) from the PI's v2 list + wugs (same keys as the frozen CSV)."""
    rows = []
    v2 = list(csv.DictReader((HERE / cfg["modifier_list"]).open()))
    for r in v2:
        if not r["word"]:
            continue
        if r["word"] == "none:":
            key = "plain_" + r["phrase"].split()[0].strip("{}").replace("/", "")
        else:
            key = f"{r['article']} {r['word']}".strip()
        rows.append((key, r["article"], r["word"], r["phrase"]))
    for r in csv.DictReader((HERE / cfg["wug_list"]).open()):
        rows.append((f"wug_{r['wug']}", "a", r["wug"], r["text"]))
    return rows


def build_character_main(cfg: dict, nouns: list[str]) -> list[dict]:
    instr = cfg["name_conditions"][cfg["name_condition"]]
    out = []
    for frame in ("consider", "writer"):
        prefix = PERSONA if frame == "writer" else ""
        for noun in nouns:
            for key, art, word, tmpl in load_modifiers(cfg):
                mn = phrase(art, word, tmpl, noun)
                p = prefix + cfg["template"].replace("{NAME_INSTR}", instr)
                p = p.replace("{MOD_NOUN}", mn).replace("{NOUN}", noun)
                out.append({"noun": noun, "modifier": f"{frame}|{key}", "frame": frame, "arm": key,
                            "mod_noun": mn, "idiom_regex": IDIOMS.get(word, ""), "prompt": p,
                            "name_condition": cfg["name_condition"]})  # fmt: skip
    return out
