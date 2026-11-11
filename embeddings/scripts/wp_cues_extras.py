"""Supplements to the cue candidate lists (docs/wordpair-cues-prereg.md, DRAFT), written to separate
files so the PI's review of data/wp_cues_{es,fr}_candidates_v0.csv isn't overwritten; merged at freeze.

1. data/wp_cues_flags_v0.csv, one row per candidate:
   - fem_also_noun: the feminine (or invariant) form also has a noun entry in Wiktionary (lista
     'list'). Like the occupation homographs (H), this could push retrieval away from the feminine
     occupation form, so it gets a sensitivity check.
   - name_homograph: the masculine or feminine form is also a given name in Wiktionary (MUSE
     vectors are lowercased, so bárbara also carries Bárbara; franco also carries Franco).
   - es_match (French only): a Spanish candidate that the MUSE es-fr dictionary pairs with this
     French masculine form; empty if none. A French sanity check that doesn't go through English.
2. data/wp_cues_es_additions_v0.csv: cues added by the PI (2026-11-11) and a disability/illness
   subset (descriptive, separate from the primary set), with corrected glosses.

    uv run --no-project --python 3.12 --with pandas python embeddings/scripts/wp_cues_extras.py
"""

import json
from pathlib import Path

import pandas as pd
from wp_cues_build import KK, MUSE, vocab

ROOT = Path(__file__).resolve().parents[1]
# (masc, fem or "", english gloss, source, subset)
ADD = [("brillante", "", "brilliant", "PI 2026-11-11", "primary"),
       ("sordo", "sorda", "deaf", "PI 2026-11-11 (gloss corrected from 'dull')", "disability"),
       ("mudo", "muda", "mute", "gloss corrected from 'silent'", "disability"),
       ("ciego", "ciega", "blind", "disability subset", "disability"),
       ("enfermo", "enferma", "sick", "disability subset", "disability"),
       ("sano", "sana", "healthy", "disability subset (contrast)", "disability"),
       ("loco", "loca", "crazy", "disability subset", "disability")]  # fmt: skip
NOT_ADDED = {"estúpido": "estúpida is outside the first 50k words",
             "sincero": "already a candidate",
             "discapacitado": "PI 2026-11-11; outside the first 50k words (rank 65,742; discapacitada 163,349)"}  # fmt: skip


def name_set(lang):
    out = set()
    with open(KK / f"kaikki.org-dictionary-{ {'es': 'Spanish', 'fr': 'French'}[lang]}.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"pos": "name"' in line and "given name" in line:
                out.add(json.loads(line)["word"].lower())
    return out


def noun_set(lang):
    out = set()
    with open(KK / f"kaikki.org-dictionary-{ {'es': 'Spanish', 'fr': 'French'}[lang]}.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"pos": "noun"' in line:
                out.add(json.loads(line)["word"])
    return out


def main():
    es = pd.read_csv(ROOT / "data/wp_cues_es_candidates_v0.csv")
    fr = pd.read_csv(ROOT / "data/wp_cues_fr_candidates_v0.csv")
    esfr = {}
    for line in open(MUSE / "dictionaries/es-fr.txt", encoding="utf-8"):
        a, b = line.split()
        esfr.setdefault(b, set()).add(a)
    rows = []
    for lang, d in (("es", es), ("fr", fr)):
        N, names = noun_set(lang), name_set(lang)
        for r in d.itertuples():
            form = r.fem if isinstance(r.fem, str) and r.fem else r.masc
            match = sorted(esfr.get(r.masc, set()) & set(es.masc)) if lang == "fr" else []
            rows.append({"lang": lang, "masc": r.masc, "fem_also_noun": form in N,
                         "name_homograph": r.masc in names or form in names,
                         "es_match": " ".join(match)})  # fmt: skip
    flags = pd.DataFrame(rows)
    flags.to_csv(ROOT / "data/wp_cues_flags_v0.csv", index=False)
    V, en = vocab("es", 50_000), vocab("en", 200_000)
    add = []
    for m, f, e, src, sub in ADD:
        ok = m in V and (not f or f in V) and e in en
        add.append({"lang": "es", "masc": m, "fem": f, "type": "gendered" if f else "invariant", "en": e,
                    "subset": sub, "source": src, "in_vocab": ok, "pi_keep": ""})  # fmt: skip
    pd.DataFrame(add).to_csv(ROOT / "data/wp_cues_es_additions_v0.csv", index=False)
    f_es = flags[flags.lang == "es"]
    f_fr = flags[flags.lang == "fr"]
    print("ES fem_also_noun:", int(f_es.fem_also_noun.sum()), "of", len(f_es), "e.g.",
          ", ".join(es.loc[f_es.fem_also_noun.to_numpy(), "fem"].dropna().head(15)))  # fmt: skip
    print("FR with ES dictionary match:", int((f_fr.es_match != "").sum()), "of", len(f_fr))
    print(pd.DataFrame(add).to_string(index=False))
    print("not added:", NOT_ADDED)


if __name__ == "__main__":
    main()
