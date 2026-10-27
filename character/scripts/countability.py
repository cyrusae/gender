"""Countability of each pool noun in the sense we translated (no models; 2026-10-26).

The main sense is the largest English translation group (feasibility.main_sense_tables). Mass =
that sense is tagged "uncountable" and not "countable"; if the sense has no countability tag, the
entry's headword line decides ("(uncountable)" only). Mass nouns get the bare noun as their plain
indefinite form ("Consider honey"), PI option 2. Writes character/results/confound/countability_191.csv.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from feasibility import main_sense_tables

from gbleed import paths

R = HERE.parents[0] / "results/confound"
pool = pd.read_csv(R / "pool_191_DRAFT.csv", keep_default_na=False)
want = set(pool.en)
dump = paths.kaikki("kaikki.org-dictionary-English.jsonl")
main = {w: v["sense"] for w, v in main_sense_tables(dump, want).items()}
info = {}
with dump.open() as fh:
    for ln in fh:
        if '"translations"' not in ln or '"pos": "noun"' not in ln:
            continue
        d = json.loads(ln)
        w = d.get("word")
        if d.get("pos") != "noun" or w not in want or w in info:
            continue
        head = " ".join(h.get("expansion", "") for h in d.get("head_templates") or [])
        for s in d.get("senses", []):
            labs = {
                t.get("sense") or (s.get("glosses") or [""])[0] for t in s.get("translations") or []
            }
            labs.add((s.get("glosses") or [""])[0])
            if main.get(w) in labs:
                tags = set(s.get("tags") or [])
                if "uncountable" in tags and "countable" not in tags:
                    c = "mass"
                elif "countable" in tags:
                    c = "count"
                elif re.search(r"\(uncountable\)", head):
                    c = "mass"
                else:
                    c = "count"
                info[w] = {"count_class": c, "sense": main[w][:60], "head": head[:60]}
                break
x = pool[["en", "conc"]].copy()
x["count_class"] = [info.get(w, {}).get("count_class", "unresolved") for w in x.en]
x["sense"] = [info.get(w, {}).get("sense", main.get(w, "")[:60]) for w in x.en]
x.to_csv(R / "countability_191.csv", index=False)
print(x.count_class.value_counts().to_dict())
print(x[x.count_class != "count"].to_string(index=False))
