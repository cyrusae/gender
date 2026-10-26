"""Round-trip and gold-gender check on the feasibility candidates (no models).

For each candidate English noun and each language, the translation picked from the English
Wiktionary translation table (feasibility.py) is looked up in that language's OWN Wiktionary dump:
- round trip: some noun entry for that word has an English gloss containing the English noun (or
  its plural) as a whole word. Catches sense mismatches like Platte/"flat" without a manual check.
- gold gender (gendered languages): the gender tags of the matching entries in the language's own
  dump; must be a single gender and equal the translation table's tag. Disagreement -> dropped.

Rules fixed before looking at outcomes (2026-10-25). Usage:
  uv run python instruct/scripts/roundtrip.py --in instruct/results/feasibility/candidates.csv \
      --out instruct/results/feasibility
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feasibility import GENDERED, GTAG, norm

from gbleed import paths

LANGS = {"de": "German", "ru": "Russian", "es": "Spanish", "hi": "Hindi", "ar": "Arabic",
         "fi": "Finnish", "ja": "Japanese", "tr": "Turkish"}  # fmt: skip
WORD_RE = re.compile(r'"word": "((?:[^"\\]|\\.)*)"')


def plural_en(w: str) -> str:
    if re.search(r"(s|x|z|ch|sh)$", w):
        return w + "es"
    if re.search(r"[^aeiou]y$", w):
        return w[:-1] + "ies"
    return w + "s"


def entry_genders(d: dict) -> set[str]:
    tags = set(d.get("tags") or [])
    for s in d.get("senses") or []:
        tags |= set(s.get("tags") or [])
    g = {GTAG[t] for t in tags if t in GTAG and t != "common"}
    if not g:  # fall back to the head template's gender argument (m, f, n, m-in, f-an, ...)
        for ht in d.get("head_templates") or []:
            for k, v in (ht.get("args") or {}).items():
                if k in ("1", "g", "g1", "g2") and re.fullmatch(r"[mfn](-\w+)?", str(v)):
                    g.add(str(v)[0])
    return g


def scan(lang: str, targets: dict[str, set[str]]) -> dict[str, dict]:
    """{normalised foreign word: {'genders': set, 'rt': set of English words matched,
    'glosses': [..]}} for noun entries of the target words."""
    out: dict[str, dict] = defaultdict(lambda: {"genders": set(), "rt": set(), "glosses": [],
                                                 "g_by_match": set()})  # fmt: skip
    want = set(targets)
    dump = paths.kaikki(f"kaikki.org-dictionary-{LANGS[lang]}.jsonl")
    with dump.open() as fh:
        for ln in fh:
            if '"pos": "noun"' not in ln:
                continue
            words = {norm(json.loads(f'"{m}"'), lang) for m in WORD_RE.findall(ln)}
            if not words & want:
                continue
            d = json.loads(ln)
            w = norm(d.get("word", ""), lang)
            if d.get("pos") != "noun" or w not in want or d.get("lang_code") != lang:
                continue
            gl = [g for s in d.get("senses") or [] for g in s.get("glosses") or []]
            gs = entry_genders(d)
            rec = out[w]
            rec["genders"] |= gs
            rec["glosses"] += gl
            for en in targets[w]:
                pat = rf"\b({re.escape(en)}|{re.escape(plural_en(en))})\b"
                if any(re.search(pat, x, re.IGNORECASE) for x in gl):
                    rec["rt"].add(en)
                    rec["g_by_match"] |= gs
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = pd.read_csv(args.inp, keep_default_na=False)
    core = list(LANGS)
    base = d[(d[[f"{lg}_status" for lg in core]] == "ok").all(axis=1)].copy()
    print(f"usable in all 8 languages from the translation tables: {len(base)}", flush=True)

    for lang in core:
        targets: dict[str, set[str]] = defaultdict(set)
        for en, fw in zip(base.en, base[f"{lang}_word"]):
            targets[norm(fw, lang)].add(en)
        res = scan(lang, targets)
        rt, gd, gok = [], [], []
        for en, fw in zip(base.en, base[f"{lang}_word"]):
            rec = res.get(norm(fw, lang))
            if rec is None:
                rt.append("no_entry")
            else:
                rt.append("ok" if en in rec["rt"] else "no_gloss_match")
            if lang in GENDERED:
                g = "".join(sorted(rec["g_by_match"])) if rec else ""
                gd.append(g)
                gok.append(len(g) == 1)
        base[f"{lang}_rt"] = rt
        if lang in GENDERED:
            base[f"{lang}_g_dump"] = gd
            base[f"{lang}_g_ok"] = [ok and g == t for ok, g, t in zip(gok, gd, base[f"{lang}_g"])]
        base[f"{lang}_glosses"] = [
            " | ".join(dict.fromkeys(res[norm(fw, lang)]["glosses"]))[:200]
            if norm(fw, lang) in res else ""
            for fw in base[f"{lang}_word"]
        ]  # fmt: skip
        print(f"{lang}: round trip ok {(base[f'{lang}_rt'] == 'ok').mean():.0%}", flush=True)

    rt_ok = (base[[f"{lg}_rt" for lg in core]] == "ok").all(axis=1)
    g_ok = base[[f"{lg}_g_ok" for lg in core if lg in GENDERED]].all(axis=1)
    base["rt_all"] = rt_ok
    base["gender_all"] = g_ok
    base["final"] = rt_ok & g_ok
    base.to_csv(args.out / "roundtrip.csv", index=False)

    rep = ["# Round-trip and gold-gender check (8 languages)\n",
           f"Start: {len(base)} nouns usable in all 8 languages from the translation tables.\n",
           ("| language | round trip ok | no entry in own dump | entry, but no gloss match | "
            "own-dump gender single and matches table |"),
           "|---|---|---|---|---|"]  # fmt: skip
    for lg in core:
        v = base[f"{lg}_rt"].value_counts()
        gm = f"{base[f'{lg}_g_ok'].mean():.0%}" if lg in GENDERED else "n/a"
        rep.append(f"| {lg} | {v.get('ok', 0)} | {v.get('no_entry', 0)} | "
                   f"{v.get('no_gloss_match', 0)} | {gm} |")  # fmt: skip
    fin = base[base.final]
    cb = fin.conc_bin.value_counts()
    rep.append(f"\n**Pass every check in every language: {len(fin)}** (round trip alone {rt_ok.sum()}, "
               f"gender alone {g_ok.sum()}). By concreteness: concrete {cb.get('concrete', 0)}, "
               f"mid {cb.get('mid', 0)}, abstract {cb.get('abstract', 0)}. "
               f"With Glasgow valence: {(fin['val'] != '').sum()}.\n")  # fmt: skip
    g = [lg for lg in core if lg in GENDERED]
    pat = fin[[f"{lg}_g" for lg in g]].agg("".join, axis=1).value_counts()
    rep.append(f"{len(pat)} gender patterns (order {' '.join(g)}); per language: " + "; ".join(
        f"{lg} " + "/".join(f"{k}{v}" for k, v in fin[f"{lg}_g"].value_counts().items()) for lg in g
    ) + "\n")  # fmt: skip
    rep.append("## Sample of nouns dropped by the round trip (English glosses from that language's "
               "own dictionary)\n")  # fmt: skip
    rep.append("| English | language | word | its glosses |\n|---|---|---|---|")
    drops = []
    for lg in core:
        bad = base[(base[f"{lg}_rt"] == "no_gloss_match")]
        for _, r in bad.head(4).iterrows():
            drops.append(f"| {r.en} | {lg} | {r[f'{lg}_word']} | {r[f'{lg}_glosses'][:110]} |")
    rep += drops
    (args.out / "roundtrip_report.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
