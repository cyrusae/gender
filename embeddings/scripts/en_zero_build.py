"""Candidate English reference words for a sex-neutral English zero point
(docs/english-default-prereg.md, DRAFT). Built by rule for PI review; nothing scored here.

Source A (derived): English glosses of the frozen Spanish and French strict-tier epicenes
(data/zero_ref_{es,fr}_frozen_v1.csv), one word, in the first 200k MUSE English words, not in
Bolukbasi et al.'s gender-specific list, not a profession (debiaswe professions.json) and not the gloss
of a noun in our occupation groups (wp_persons_{es,fr}_frozen_v1.csv).
Source B (seed): generic words for people, listed by Claude (claude_suggest_unverified), filtered the
same way.
Also dropped (2026-11-20): glosses of nouns tested in Q2/Q3 (anchor and test sets kept apart), and
words whose Glasgow GEND (or their singular/plural partner's) is outside 3.5-4.5 (PI).
Primary set = the seed words that pass the filters (whatever their source); sensitivity set = the
other epicene glosses that pass.

    uv run --no-project --python 3.12 --with pandas python embeddings/scripts/en_zero_build.py
"""

import json
from pathlib import Path

import pandas as pd
from wp_cues_build import glasgow, vocab

ROOT = Path(__file__).resolve().parents[1]
PAIR = {"persons": "person", "people": "person", "individuals": "individual", "humans": "human", "members": "member",
        "residents": "resident", "strangers": "stranger", "children": "child", "kids": "kid", "babies": "baby",
        "adults": "adult", "citizens": "citizen", "friends": "friend", "neighbours": "neighbour",
        "youths": "youth", "adolescents": "adolescent", "teens": "teen", "teenagers": "teenager"}  # plural -> singular
SEED = """person people persons individual individuals human humans someone somebody anyone anybody everyone
everybody child children kid kids adult adults baby babies infant citizen citizens member members folks
resident residents youth stranger strangers
youths adolescent adolescents teen teens teenager teenagers""".split()  # last line added 2026-11-20 (PI)


def main():
    en = vocab("en", 200_000)
    spec = {w.lower() for w in json.load(open(ROOT / "external/debiaswe/data/gender_specific_full.json"))}
    prof = {p[0] for p in json.load(open(ROOT / "external/debiaswe/data/professions.json"))}
    occ, tested = set(), set()
    for lang in ("es", "fr"):
        f = pd.read_csv(ROOT / f"data/wp_persons_{lang}_frozen_v1.csv", keep_default_na=False, dtype=str)
        occ |= set(f[f.group == "occupation"].en)
        tested |= set(f.en)
    gl = glasgow()
    singular = {**PAIR, **{v: v for v in PAIR.values()}}
    plurals = {}
    for pl, sg in PAIR.items():
        plurals.setdefault(sg, []).append(pl)

    def family(w):  # a word with its singular/plural partner(s)
        sg = singular.get(w, w)
        return {sg, *plurals.get(sg, [])}

    def gendered(w):  # PI 2026-11-20: Glasgow GEND outside 3.5-4.5 for the word or its singular/plural
        return any(gl.get(x) is not None and not 3.5 <= gl.get(x) <= 4.5 for x in family(w))
    rows = {}
    for lang in ("es", "fr"):
        r = pd.read_csv(ROOT / f"data/zero_ref_{lang}_frozen_v1.csv")
        for x in r[(r["set"] == "epicene") & (r["tier"] == "strict")].itertuples():
            e = str(x.concept_en).lower().strip()
            rows.setdefault(e, {"word": e, "source": "epicene gloss", "from": []})["from"].append(f"{lang}:{x.lemma}")
    for w in SEED:
        rows.setdefault(w, {"word": w, "source": "seed (claude_suggest_unverified)", "from": []})
    out = []
    for e, r in rows.items():
        why = ("multiword" if " " in e else "not in 200k" if e not in en else "gender-specific" if family(e) & spec
               else "profession" if family(e) & prof else "occupation-group gloss" if e in occ
               else "gloss of a tested noun" if family(e) & tested
               else "Glasgow-gendered" if gendered(e) else "")  # fmt: skip
        out.append({**r, "from": " ".join(r["from"]), "primary_set": e in SEED, "rank": en.get(e, ""), "glasgow_gend": gl.get(e, ""),
                    "rule_drop": why, "pi_keep": ""})  # fmt: skip
    d = pd.DataFrame(out).sort_values(["rule_drop", "source", "word"])
    d.to_csv(ROOT / "data/en_zero_candidates_v0.csv", index=False)
    k = d[d.rule_drop == ""]
    print("candidates", len(d), "| kept by rule", len(k), "| dropped", d.rule_drop.value_counts().to_dict())
    print("primary (generic person words):", len(k[k.primary_set]), " ".join(k[k.primary_set].word))
    print("sensitivity (epicene glosses, not primary):", len(k[~k.primary_set]))


if __name__ == "__main__":
    main()
