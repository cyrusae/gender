"""Translations of the burned pilot nouns for the writer-prompt pilot (no models; 2026-10-26).

Same rules as the main pool (feasibility.resolve on the main-sense translation table; round trip
and gold gender from each language's own Wiktionary dump via roundtrip.scan). Pilot nouns are
excluded from the main pipeline (reserved), so they are resolved here. A language that fails for a
noun is left blank and that arm is skipped for that noun.
Writes instruct/stimuli/pilot0_translations_v1.csv.
"""

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from feasibility import GENDERED, main_sense_tables, norm, resolve
from roundtrip import LANGS, scan

from gbleed import paths

NOUNS = pd.read_csv(HERE.parents[0] / "stimuli/pilot0_nouns_v2.csv", comment="#").noun.tolist()
tables = main_sense_tables(paths.kaikki("kaikki.org-dictionary-English.jsonl"), set(NOUNS))
rows = {n: {"en": n, "sense": tables.get(n, {}).get("sense", "")} for n in NOUNS}
for lg in LANGS:
    targets = defaultdict(set)
    for n in NOUNS:
        r = resolve(tables.get(n, {}).get("trans", []), lg)
        rows[n][f"{lg}_table"] = r["status"]
        if r["status"] == "ok":
            rows[n][f"{lg}_cand"] = r["word"]
            rows[n][f"{lg}_tg"] = r.get("gender", "")
            targets[norm(r["word"], lg)].add(n)
    res = scan(lg, targets)
    for n in NOUNS:
        w = rows[n].get(f"{lg}_cand")
        rec = res.get(norm(w, lg)) if w else None
        ok = bool(rec and n in rec["rt"])
        g = ""
        if ok and lg in GENDERED:
            gd = rec["g_by_sense"]
            ok = len(gd) == 1 and rows[n][f"{lg}_tg"] in gd
            g = "".join(sorted(gd))
        rows[n][f"{lg}_word"] = w if ok else ""
        rows[n][f"{lg}_g"] = g if ok else ""
    print(f"{lg} done", flush=True)
out = pd.DataFrame(rows.values())
keep = ["en", "sense"] + [c for lg in LANGS for c in (f"{lg}_word", f"{lg}_g", f"{lg}_table")]
out[keep].to_csv(HERE.parents[0] / "stimuli/pilot0_translations_v1.csv", index=False)
print(out[["en"] + [f"{lg}_word" for lg in LANGS]].to_string(index=False))
print(out[["en"] + [f"{lg}_g" for lg in LANGS if lg in GENDERED]].to_string(index=False))
