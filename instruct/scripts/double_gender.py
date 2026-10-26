"""Double-gender supplementary set (secondary arm; PI 2026-10-26), no models.

A noun qualifies if, in at least one gendered study language, its translation is ONE word that
carries two genders (e.g. Arabic إصبع m/f), and everything else passes the main-pool rules:
- English side and translation-table rules as in feasibility.py (same pool, same main sense).
- "Double" language: all common (zipf >= 2.5) single-word candidates are the same word, and that
  word's tags (translation table) carry exactly two of m/f/n; in the word's OWN dump the round
  trip holds and the gender tags/headword line give the same two genders (gold).
- Other study languages: status ok in the translation table, round trip ok, and (gendered)
  own-dump gender single and equal to the table's. v2 (PI 2026-10-26): a noun may FAIL in up to
  3 of the 8 languages (fewer than half); it then skips those arms. A double language is also
  found when the table gives one gender but the word's own dump gives two (that gender included).
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


MAX_FAIL = 3  # PI 2026-10-26: a supplementary noun may fail in fewer than half of the 8 languages


def main() -> None:
    cand = pd.read_csv(
        HERE.parents[0] / "results/feasibility/candidates.csv", keep_default_na=False
    )
    cand = cand.set_index("en")
    tables = main_sense_tables(paths.kaikki("kaikki.org-dictionary-English.jsonl"), set(cand.index))
    rows = []
    for en, tb in tables.items():
        r = cand.loc[en]
        if r.lexname in ("noun.person", "noun.animal"):  # same animacy rule as the main pool
            continue
        row = {"en": en, "conc": r.conc}
        n_cand_lang = 0
        for lg in CORE:
            w, g = "", ""
            if r[f"{lg}_status"] == "ok":
                w, g = r[f"{lg}_word"], r.get(f"{lg}_g", "")
            elif lg in G and r[f"{lg}_status"] == "mixed" and (x := double_word(tb["trans"], lg)):
                w, g = x["word"], x["g"]
            row[f"{lg}_word"] = w
            if lg in G:
                row[f"{lg}_g"] = g
            n_cand_lang += bool(w)
        if n_cand_lang >= len(CORE) - MAX_FAIL and any(row[f"{lg}_word"] for lg in G):
            rows.append(row)
    d = pd.DataFrame(rows)
    print(
        f"nouns with a usable table word in >= {len(CORE) - MAX_FAIL} languages: {len(d)}",
        flush=True,
    )
    status = pd.DataFrame(index=d.index)
    for lg in CORE:
        targets: dict[str, set[str]] = defaultdict(set)
        for en, fw in zip(d.en, d[f"{lg}_word"]):
            if fw:
                targets[norm(fw, lg)].add(en)
        res = scan(lg, targets)
        st = []
        for en, fw, g in zip(d.en, d[f"{lg}_word"], d[f"{lg}_g"] if lg in G else [""] * len(d)):
            rec = res.get(norm(fw, lg)) if fw else None
            if not rec or en not in rec["rt"]:
                st.append("fail")
                continue
            if lg not in G:
                st.append("ok")
                continue
            gd = "".join(sorted(rec["g_by_sense"]))
            if len(gd) == 2 and (len(g) == 2 and g == gd or len(g) == 1 and g in gd):
                st.append("double:" + gd)
            elif len(gd) == 1 and gd == g:
                st.append("ok")
            else:
                st.append("fail")
        status[lg] = st
        print(f"{lg} scanned", flush=True)
    n_fail = (status == "fail").sum(axis=1)
    n_dbl = status.apply(lambda c: c.str.startswith("double")).sum(axis=1)
    keep = (n_dbl >= 1) & (n_fail <= MAX_FAIL)
    out = d[keep].copy()
    for lg in CORE:
        out[f"{lg}_status"] = status.loc[keep, lg]
    out["n_fail"] = n_fail[keep]
    out["double_in"] = [
        "|".join(lg for lg in G if status.at[i, lg].startswith("double")) for i in out.index
    ]
    print(f"\nfinal: {len(out)}; by double language: {out.double_in.value_counts().to_dict()}; "
          f"by failures: {out.n_fail.value_counts().sort_index().to_dict()}")  # fmt: skip
    cols = ["en", "double_in", "n_fail", "conc"] + [
        c for lg in CORE for c in (f"{lg}_word", f"{lg}_status")
    ]
    out[cols].to_csv(
        HERE.parents[0] / "results/feasibility/double_gender_supplement_DRAFT.csv", index=False
    )
    print(out[["en", "double_in", "n_fail"] + [f"{lg}_status" for lg in G]].to_string(index=False))


if __name__ == "__main__":
    _ = data  # keep import (raw-data resolver used by feasibility)
    main()
