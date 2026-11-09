"""Freeze the C1 test-noun list (embeddings/docs/c1-prereg.md, adopted 2026-11-08).

Rule: Wiktionary lexicon (data/lexicon/es_nouns.csv) nouns with a single gender (m/f), animacy
inanimate, no multi_type, no also_form, no also_pos (PI: drop words that are also another part of
speech), in the first 200,000 MUSE wiki.multi.es words, and not in any of Zhou et al.'s word lists
(noun lists, definitional pairs, attribute words, occupation pairs). `in_50k` marks the primary set.

    python3 embeddings/scripts/c1_nouns.py [es|fr]   (French: addendum in docs/c1-prereg.md)
"""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Z = ROOT / "embeddings/external/Bias_in_Gendered_Languages"
OUT = ROOT / "embeddings/data/c1_nouns_{lang}_v1.csv"
ATTR = ["hombre", "niño", "padre", "masculino", "abuelo", "esposo", "hijo", "tio",
        "niña", "femenina", "tía", "hija", "esposa", "mujer", "madre", "abuela"]  # fmt: skip


def main(lang="es"):
    out = Path(str(OUT).format(lang=lang))
    rank = {}
    with open(Z / f"data/wiki.multi.{lang}.vec.txt", encoding="utf-8", errors="ignore") as f:
        next(f)
        for i, line in enumerate(f):
            rank.setdefault(line.split(" ", 1)[0], i)
    defs = json.load(open(Z / f"{lang.upper()}_definitional_pairs.json"))
    excl = set(ATTR) if lang == "es" else {w for p in defs for w in p}  # FR attributes = FR defs
    for name in (f"masculine_nouns_{lang}", f"feminine_nouns_{lang}"):
        excl |= {w.strip() for w in open(Z / name, encoding="utf-8") if w.strip()}
    excl |= {w for p in defs for w in p}
    for line in open(Z / f"{lang.upper()}_occupation_words_with_EN_translations", encoding="utf-8"):
        excl |= set(line.rstrip("\n").split("\t")[:2])
    rows = []
    for r in csv.DictReader(open(ROOT / f"data/lexicon/{lang}_nouns.csv", encoding="utf-8")):
        w = r["lemma"]
        if (r["gender"] in ("m", "f") and r["animacy"] == "inanimate" and not r["multi_type"]
                and not r["also_form"] and not r["also_pos"] and w in rank and w not in excl):  # fmt: skip
            spell = ({"es_ending": r["es_ending"], "es_exception": r["es_exception"]} if lang == "es"
                     else {"fr_suffix": r["fr_suffix"], "fr_suffix_regular": r["fr_suffix_regular"],
                           "fr_final_e": r["fr_final_e"]})  # fmt: skip
            rows.append({"lemma": w, "gender": r["gender"], **spell, "concept_en": r["concept_en"],
                         "zipf": r["zipf"], "muse_rank": rank[w], "in_50k": rank[w] < 50000})  # fmt: skip
    rows.sort(key=lambda x: x["muse_rank"])
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    p = [r for r in rows if r["in_50k"]]
    print(len(rows), "nouns (200k);", len(p), "in 50k:",
          sum(r["gender"] == "m" for r in p), "m /", sum(r["gender"] == "f" for r in p), "f")
    print("sha256", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "es")
