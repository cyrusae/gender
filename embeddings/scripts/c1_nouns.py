"""Freeze the C1 test-noun list (embeddings/docs/c1-prereg.md, adopted 2026-11-08).

Rule: Wiktionary lexicon (data/lexicon/es_nouns.csv) nouns with a single gender (m/f), animacy
inanimate, no multi_type, no also_form, no also_pos (PI: drop words that are also another part of
speech), in the first 200,000 MUSE wiki.multi.es words, and not in any of Zhou et al.'s word lists
(noun lists, definitional pairs, attribute words, occupation pairs). `in_50k` marks the primary set.

    python3 embeddings/scripts/c1_nouns.py
"""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Z = ROOT / "embeddings/external/Bias_in_Gendered_Languages"
OUT = ROOT / "embeddings/data/c1_nouns_es_v1.csv"
ATTR = ["hombre", "niño", "padre", "masculino", "abuelo", "esposo", "hijo", "tio",
        "niña", "femenina", "tía", "hija", "esposa", "mujer", "madre", "abuela"]  # fmt: skip


def main():
    rank = {}
    with open(Z / "data/wiki.multi.es.vec.txt", encoding="utf-8", errors="ignore") as f:
        next(f)
        for i, line in enumerate(f):
            rank.setdefault(line.split(" ", 1)[0], i)
    excl = set(ATTR)
    for name in ("masculine_nouns_es", "feminine_nouns_es"):
        excl |= {w.strip() for w in open(Z / name, encoding="utf-8") if w.strip()}
    excl |= {w for p in json.load(open(Z / "ES_definitional_pairs.json")) for w in p}
    for line in open(Z / "ES_occupation_words_with_EN_translations", encoding="utf-8"):
        excl |= set(line.rstrip("\n").split("\t")[:2])
    rows = []
    for r in csv.DictReader(open(ROOT / "data/lexicon/es_nouns.csv", encoding="utf-8")):
        w = r["lemma"]
        if (r["gender"] in ("m", "f") and r["animacy"] == "inanimate" and not r["multi_type"]
                and not r["also_form"] and not r["also_pos"] and w in rank and w not in excl):  # fmt: skip
            rows.append({"lemma": w, "gender": r["gender"], "es_ending": r["es_ending"],
                         "es_exception": r["es_exception"], "concept_en": r["concept_en"],
                         "zipf": r["zipf"], "muse_rank": rank[w], "in_50k": rank[w] < 50000})  # fmt: skip
    rows.sort(key=lambda x: x["muse_rank"])
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    p = [r for r in rows if r["in_50k"]]
    print(len(rows), "nouns (200k);", len(p), "in 50k:",
          sum(r["gender"] == "m" for r in p), "m /", sum(r["gender"] == "f" for r in p), "f")
    print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
