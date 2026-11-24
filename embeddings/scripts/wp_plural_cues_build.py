"""Plural forms for the plural-cue arm of the word-pair plurals study (paper B; built 2026-11-24, before
any score).

By rule from the kaikki.org Wiktionary dumps (as wp_persons_plurals.py):
- cue adjectives (frozen v1 cues, primary subset): masc_pl = form tagged ["masculine", "plural"], else
  ["plural"]; fem_pl = ["feminine", "plural"], else the feminine headword's ["plural"]; invariant
  adjectives: ["plural"] (one form). Adjective entries only.
- English occupation glosses (occupation_mapping_v1, clean list): noun form tagged ["plural"]
  (first noun entry with one).
Coverage flags: cue plural forms in the first 200k MUSE words of the language; English plural in the
first 200k English words. Output: data/wp_plural_cues_v0.csv, data/wp_plural_occ_en_v0.csv.

    uv run --no-project --python 3.12 --with pandas python embeddings/scripts/wp_plural_cues_build.py
"""

import json
from pathlib import Path

import pandas as pd
from zhou_shift import Z

ROOT = Path(__file__).resolve().parents[1]
KK = Path("/Volumes/Lexar/gbleed/shared/kaikki")
DUMP = {"es": "Spanish", "fr": "French", "en": "English"}


def vocab(lang, n=200_000):
    out = set()
    with open(Z / f"data/wiki.multi.{lang}.vec.txt", encoding="utf-8", errors="ignore") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= n:
                break
            out.add(line.split(" ", 1)[0])
    return out


def forms(lang, words, pos):
    got = {}
    with open(KK / f"kaikki.org-dictionary-{DUMP[lang]}.jsonl", encoding="utf-8") as f:
        for line in f:
            if f'"pos": "{pos}"' not in line:
                continue
            j = json.loads(line)
            w = j.get("word")
            if w not in words or w in got:
                continue
            fs = {tuple(sorted(x.get("tags", []))): x.get("form") for x in j.get("forms", []) if x.get("form")}
            if fs:
                got[w] = fs
    return got


def main():
    cues = pd.read_csv(ROOT / "data/wp_cues_frozen_v1.csv", keep_default_na=False, dtype=str)
    cues = cues[cues.subset == "primary"].reset_index(drop=True)
    rows = []
    for lang in ("es", "fr"):
        c = cues[cues.lang == lang]
        F = forms(lang, set(c.masc) | set(c.fem), "adj")
        V = vocab(lang)
        for _, r in c.iterrows():
            fm, ff = F.get(r.masc, {}), F.get(r.fem, {})
            if r.type == "gendered":
                mp = fm.get(("masculine", "plural")) or fm.get(("plural",), "")
                fp = fm.get(("feminine", "plural")) or ff.get(("plural",), "")
            else:
                mp = fp = fm.get(("plural",)) or fm.get(("feminine", "masculine", "plural"), "")
            rows.append({"lang": lang, "type": r.type, "en": r.en, "masc": r.masc, "fem": r.fem,
                         "masc_pl": mp, "fem_pl": fp, "pl_in_200k": bool(mp) and mp in V and fp in V})  # fmt: skip
    P = pd.DataFrame(rows)
    P.to_csv(ROOT / "data/wp_plural_cues_v0.csv", index=False)
    mp = pd.read_csv(ROOT / "data/occupation_mapping_v1.csv")
    en_words = set(mp.en)
    FE = forms("en", en_words, "noun")
    VE = vocab("en")
    O = mp[["lang", "masc", "fem", "en", "clean_keep"]].copy()
    O["en_pl"] = [FE.get(e, {}).get(("plural",), "") for e in O.en]
    O["en_pl_in_200k"] = [bool(p) and p in VE for p in O.en_pl]
    O.to_csv(ROOT / "data/wp_plural_occ_en_v0.csv", index=False)
    print(P.groupby(["lang", "type"]).agg(n=("en", "size"), with_forms=("masc_pl", lambda x: (x != "").sum()),
                                          in_200k=("pl_in_200k", "sum")).to_string())  # fmt: skip
    print(O[O.clean_keep].groupby("lang").agg(n=("en", "size"), en_pl=("en_pl_in_200k", "sum")).to_string())


if __name__ == "__main__":
    main()
