"""Word structure from Wiktionary etymology templates (playbook Stage 3), for any language.

A noun counts as *complex (Wiktionary)* if an etymology template of its noun entry is a
structure template: compound, affix/af, suffix, prefix, confix, blend (and their short names).
The template's expansion gives the parts (*Wasser + Kocher*, *скорый + -ость*); the last part
starting with "-" is recorded as its suffix, which can define suffix cells. Coverage is
Wiktionary's: words it lists only as inherited (*радость* "joy" ← Old East Slavic) count as
simple even when they are historically derived, so this complements a hand rule rather than
replacing it. Written once per language to data/lexicon/{lang}_structure.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import paths
from .lexicon import LANG_NAMES, LEX_DIR

STRUCTURE_TEMPLATES = {"compound", "com", "affix", "af", "suffix", "suf", "prefix", "pre",
                       "confix", "con", "blend", "univerbation", "univ"}  # fmt: skip


def _parts(expansion: str) -> list[str]:
    import re

    e = re.sub(r"\s*\([^)]*\)", "", expansion)  # drop transliterations and glosses
    return [p.strip().strip('"“”') for p in e.split(" + ") if p.strip()]


def _plain(w: str) -> str:
    """Lowercase without stress marks (Russian parts are written ско́рый)."""
    return w.lower().replace("\u0301", "").replace("\u0300", "")


def build(lang: str) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    words: set[str] = set()  # every headword of the dump, any part of speech
    dump = paths.kaikki(f"kaikki.org-dictionary-{LANG_NAMES[lang]}.jsonl")
    with open(dump) as f:
        for line in f:
            if '"pos": "noun"' not in line:
                if (k := line.find('"word": "')) >= 0:
                    words.add(_plain(line[k + 9 : line.find('"', k + 9)]))
                continue
            j = json.loads(line)
            words.add(_plain(str(j.get("word", ""))))
            if j.get("pos") != "noun":
                continue
            w = j.get("word")
            for t in j.get("etymology_templates", []):
                if t.get("name") in STRUCTURE_TEMPLATES and w not in rows:
                    parts = _parts(t.get("expansion", ""))
                    suf = next(
                        (p for p in reversed(parts) if p.startswith("-") and not p.endswith("-")),
                        "",
                    )  # not -о- interfixes
                    rows[w] = {"lemma": w, "wk_template": t["name"], "wk_parts": " + ".join(parts),
                               "wk_suffix": suf}  # fmt: skip
                    break
    df = pd.DataFrame(rows.values(), columns=["lemma", "wk_template", "wk_parts", "wk_suffix"])

    # transparent: at least one non-affix part is still a word of the language (Wasser + Kocher,
    # Arm + -el), not only reconstructed or older-stage parts (*kom- + *bereti; wint + brāwa)
    def transparent(parts: str) -> bool:
        stems = [
            p
            for p in parts.split(" + ")
            if p and not p.startswith(("-", "*")) and not p.endswith("-")
        ]
        return any(_plain(p) in words for p in stems)

    df["wk_transparent"] = df.wk_parts.map(transparent)
    df.to_csv(Path(LEX_DIR) / f"{lang}_structure.csv", index=False)
    return df


def load(lang: str) -> pd.DataFrame:
    p = Path(LEX_DIR) / f"{lang}_structure.csv"
    return pd.read_csv(p, keep_default_na=False) if p.exists() else build(lang)


def rule(lang: str, transparent_only: bool = True):
    """word -> 'wk:<template>' if Wiktionary gives it a structure template (by default only a
    transparent one: some part is still a word of the language), else ''."""
    d = load(lang)
    if transparent_only:
        d = d[d.wk_transparent.astype(str) == "True"]
    t = d.set_index("lemma").wk_template.to_dict()
    return lambda w: f"wk:{t[w]}" if w in t else ""
