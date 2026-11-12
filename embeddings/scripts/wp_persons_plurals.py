"""Plural forms for the frozen person nouns (data/wp_persons_es_frozen_v1.csv), for the person-noun
generic-masculine study (docs/persons-genmasc-prereg.md, DRAFT). Rule: Spanish Wiktionary noun entry
of the masculine headword; masc_pl = form tagged exactly ["plural"], fem_pl = form tagged
["feminine", "plural"]; if missing, the feminine headword's ["plural"] form. Coverage in the first
50k MUSE words is recorded, not filtered (the study's floor decides). Nothing is scored here.

    uv run --no-project --python 3.12 --with pandas python embeddings/scripts/wp_persons_plurals.py
"""

import json
from pathlib import Path

import pandas as pd
from wp_cues_build import KK, vocab

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = pd.read_csv(ROOT / "data/wp_persons_es_frozen_v1.csv", keep_default_na=False, dtype=str)
    want = set(d.masc) | set(d.fem)
    forms = {}
    with open(KK / "kaikki.org-dictionary-Spanish.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"pos": "noun"' not in line:
                continue
            j = json.loads(line)
            w = j.get("word")
            if w in want and w not in forms:
                forms[w] = {tuple(x.get("tags", [])): x["form"] for x in j.get("forms", [])}
    V = vocab("es", 50_000)
    mp = [forms.get(m, {}).get(("plural",), "") for m in d.masc]
    fp = [forms.get(m, {}).get(("feminine", "plural"), "") or forms.get(f, {}).get(("plural",), "")
          for m, f in zip(d.masc, d.fem)]  # fmt: skip
    d = d.assign(masc_pl=mp, fem_pl=fp)
    d["pl_in_50k"] = [bool(a) and bool(b) and a in V and b in V for a, b in zip(d.masc_pl, d.fem_pl)]
    d[["masc", "fem", "masc_pl", "fem_pl", "group", "pl_in_50k"]].to_csv(ROOT / "data/wp_persons_es_plurals_v0.csv", index=False)
    print(d.groupby("group").pl_in_50k.agg(["sum", "count"]).to_dict())
    print(d[(d.group == "person") & d.pl_in_50k][["masc_pl", "fem_pl"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
