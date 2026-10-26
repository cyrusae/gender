"""Double-gender supplementary set (secondary arm; PI 2026-10-26), no models.

A noun qualifies if, in at least one gendered study language, its translation is ONE word that
carries two genders (e.g. Arabic إصبع m/f), and everything else passes the main-pool rules:
- English side and translation-table rules as in feasibility.py (same pool, same main sense).
- "Double" language: all common (zipf >= 2.5) single-word candidates are the same word, and that
  word's tags (translation table) carry exactly two of m/f/n; in the word's OWN dump the round
  trip holds and the gender tags/headword line give the same two genders (gold).
- Every other study language: status ok in the translation table, round trip ok, and (gendered)
  own-dump gender single and equal to the table's.
Output: instruct/results/feasibility/double_gender_supplement_DRAFT.csv (overwrites the 8-noun
first draft, which only searched nouns that had passed the single-gender table filter).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from feasibility import (
    BAD_TAGS,
    GENDERED,
    GTAG,
    data,
    main_sense_tables,
    norm,
    zipf,
)
from roundtrip import LANGS, scan

from gbleed import paths

CORE = list(LANGS)
G = [lg for lg in CORE if lg in GENDERED]


def double_word(trans: list[dict], lang: str) -> dict | None:
    cands = []
    for t in trans:
        if t.get("lang_code") != lang:
            continue
        tags = set(t.get("tags") or [])
        w = (t.get("word") or "").strip()
        if not w or " " in w or tags & BAD_TAGS or any("-" in x and x[0].isupper() for x in tags):
            continue
        z = zipf(w, lang)
        if z is not None and z < 2.5:
            continue
        cands.append((norm(w, lang), {GTAG[x] for x in tags if x in GTAG and x != "common"}, w))
    words = {c[0] for c in cands}
    if len(words) != 1:
        return None
    gs = set().union(*(c[1] for c in cands))
    if len(gs) != 2:
        return None
    return {"word": cands[0][2], "g": "".join(sorted(gs))}


def main() -> None:
    cand = pd.read_csv(
        HERE.parents[0] / "results/feasibility/candidates.csv", keep_default_na=False
    )
    cand = cand.set_index("en")
    tables = main_sense_tables(paths.kaikki("kaikki.org-dictionary-English.jsonl"), set(cand.index))
    rows = []
    for en, tb in tables.items():
        r = cand.loc[en]
        dbl = {}
        ok = True
        for lg in CORE:
            if r[f"{lg}_status"] == "ok":
                continue
            if lg in G and r[f"{lg}_status"] == "mixed" and (x := double_word(tb["trans"], lg)):
                dbl[lg] = x
                continue
            ok = False
            break
        if ok and dbl:
            row = {"en": en, "conc": r.conc, "double_in": "|".join(dbl)}
            for lg in CORE:
                row[f"{lg}_word"] = dbl[lg]["word"] if lg in dbl else r[f"{lg}_word"]
                if lg in G:
                    row[f"{lg}_g"] = dbl[lg]["g"] if lg in dbl else r[f"{lg}_g"]
            rows.append(row)
    d = pd.DataFrame(rows)
    print(f"table-level candidates: {len(d)}", flush=True)
    keep = pd.Series(True, index=d.index)
    for lg in CORE:
        targets: dict[str, set[str]] = defaultdict(set)
        for en, fw in zip(d.en, d[f"{lg}_word"]):
            targets[norm(fw, lg)].add(en)
        res = scan(lg, targets)
        rt, gd = [], []
        for en, fw in zip(d.en, d[f"{lg}_word"]):
            rec = res.get(norm(fw, lg))
            rt.append(bool(rec and en in rec["rt"]))
            gd.append("".join(sorted(rec["g_by_match"])) if rec else "")
        keep &= pd.Series(rt, index=d.index)
        if lg in G:
            d[f"{lg}_g_dump"] = gd
            keep &= d[f"{lg}_g_dump"] == d[f"{lg}_g"]
        print(f"{lg}: still in {int(keep.sum())}", flush=True)
    out = d[keep].copy()
    print(f"\nfinal: {len(out)}")
    print(out["double_in"].value_counts().to_string())
    cols = ["en", "double_in", "conc"] + [
        c for lg in CORE for c in (f"{lg}_word", f"{lg}_g") if c in out
    ]
    out[cols].to_csv(
        HERE.parents[0] / "results/feasibility/double_gender_supplement_DRAFT.csv", index=False
    )
    print(out[["en", "double_in"] + [f"{lg}_g" for lg in G]].to_string(index=False))


if __name__ == "__main__":
    _ = data  # keep import (raw-data resolver used by feasibility)
    main()
