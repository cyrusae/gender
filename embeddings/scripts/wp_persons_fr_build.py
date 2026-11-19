"""French person-noun list for the French replication of the person-noun studies
(docs/persons-fr-prereg.md, DRAFT). Built entirely by rule; no label comes from a model or from an
unreviewed judgement. Nothing is scored here.

Group labels are inherited through English: the Spanish list (data/wp_persons_es_frozen_v1.csv) was
reviewed by the PI and assigns each English gloss a group (person / occupation / identity). A French
pair gets that group when its English gloss matches a Spanish entry's gloss exactly; French pairs whose
gloss isn't in the Spanish list, or whose gloss has more than one group there, are left out
(`group` empty, counted). So French persons and occupations are the same concepts as in Spanish,
where French has a distinct feminine form.

Candidates (kaikki.org French dump, part of speech noun): headword = masculine singular, lowercase
letters (accents, hyphen-free); feminine = form tagged exactly ["feminine"], different from the
headword; both in the first 50k MUSE French words; English gloss as wp_cues_build.gloss, one word, in
the first 200k MUSE English words. Drops as in Spanish: name homographs (either form a French given
name); identity nouns where either form is a place name (PI rule, 2026-11-11).
Flags (same rules as wp_persons_flags.py, on the French dump): fem_also_noun, masc_also_noun,
language_name, masc_identical_en, place_name. Plural forms (["plural"], ["feminine", "plural"]) for
the generic-masculine study, with a 200k-vocabulary flag; a feminine plural missing from both entries
is formed by the regular rule (+s, unless the form ends in s/x/z).

    uv run --no-project --python 3.12 --with pandas python embeddings/scripts/wp_persons_fr_build.py
"""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from wp_cues_build import KK, gloss, vocab
from wp_persons_build import PERSON
from wp_persons_flags import MALE_EQ, PLACE

ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"^[a-zàâæçéèêëîïôœùûüÿ]+$")
FEM_EQ = re.compile(r"^(female|feminine) (equivalent|form|version) of")
LANG = re.compile(r"^(the |a )?([a-z-]+ ){0,3}(language|dialect)\b|language (spoken|of)|\((language|language family|dialect)\)")


def main():
    V50, V200, en = vocab("fr", 50_000), vocab("fr", 200_000), vocab("en", 200_000)
    es = pd.read_csv(ROOT / "data/wp_persons_es_frozen_v1.csv", keep_default_na=False, dtype=str)
    g_by_en = es.groupby("en").group.agg(lambda s: sorted(set(s)))
    names, place, nonperson, fem_other, cands, forms = set(), set(), {}, set(), {}, {}
    langname, senses_of, plural_of, fem_strict = set(), {}, {}, set()
    prof = {p[0] for p in json.load(open(ROOT / "external/debiaswe/data/professions.json"))}
    with open(KK / "kaikki.org-dictionary-French.jsonl", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            w, pos = j.get("word", ""), j.get("pos")
            senses = j.get("senses", [])
            if pos == "name":
                gl = " ".join(" ".join(s.get("glosses", [])) for s in senses).lower()
                if "given name" in gl or any("given-name" in s.get("tags", []) for s in senses):
                    names.add(w.lower())
                if PLACE.search(gl):
                    place.add(w.lower())
                continue
            if pos != "noun":
                continue
            senses_of.setdefault(w, [" ".join(s.get("glosses", [])).lower() for s in senses])
            plural_of.setdefault(w, next((x["form"] for x in j.get("forms", []) if x.get("tags") == ["plural"]), ""))
            if w in cands or not WORD.match(w) or w not in V50:
                continue
            fem = next((x["form"] for x in j.get("forms", []) if x.get("tags") == ["feminine"]), None)
            if not fem or fem == w or fem not in V50:
                continue
            fm = {tuple(x.get("tags", [])): x["form"] for x in j.get("forms", [])}
            cands[w] = (fem, gloss(j))
            forms[w] = (fm.get(("plural",), ""), fm.get(("feminine", "plural"), ""))
    # same rule as wp_persons_flags.py: a sense is a non-person sense unless it is "male/female
    # equivalent of", matches the person pattern, or contains a word that is itself a person gloss
    pwords = {e for _, e in cands.values()} | prof | {w + "s" for w in prof} | {"woman", "women", "wife", "girl", "lady", "men"}
    for w, gs in senses_of.items():
        for g in gs:
            if not g:
                continue
            if not FEM_EQ.match(g):  # as wp_persons_build.py (any sense other than "female equivalent of")
                fem_other.add(w)
            if not FEM_EQ.match(g) and not PERSON.search(g) and not set(re.findall(r"[a-z]+", g)) & pwords:
                fem_strict.add(w)
            if not MALE_EQ.match(g) and not PERSON.search(g) and not set(re.findall(r"[a-z]+", g)) & pwords:
                nonperson.setdefault(w, []).append(g[:60])
                if LANG.search(g):
                    langname.add(w)
    rows, unmatched, ambiguous = [], 0, 0
    for m, (f, e) in sorted(cands.items()):
        if " " in e or e not in en:
            continue
        grps = g_by_en.get(e)
        if grps is None:
            unmatched += 1
            group = ""
        elif len(grps) > 1:
            ambiguous += 1
            group = ""
        else:
            group = grps[0]
        mp, fp = forms[m]
        fp = fp or plural_of.get(f, "") or (f + "s" if not f.endswith(("s", "x", "z")) else "")  # regular -s plural
        rows.append({"lang": "fr", "masc": m, "fem": f, "en": e, "group": group,
                     "rank_m": V50[m], "rank_f": V50[f],
                     "name_homograph": m in names or f in names,
                     "fem_also_noun": f in fem_other, "fem_nonperson_sense": f in fem_strict, "masc_also_noun": m in nonperson,
                     "language_name": m in langname, "masc_identical_en": m == e,
                     "place_name": m in place or f in place,
                     "masc_pl": mp, "fem_pl": fp,
                     "pl_in_200k": bool(mp) and bool(fp) and mp in V200 and fp in V200,
                     "masc_other_senses": " | ".join(nonperson.get(m, [])[:3])})  # fmt: skip
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "data/wp_persons_fr_candidates_v0.csv", index=False)
    keep = (d.group != "") & ~d.name_homograph & ~((d.group == "identity") & d.place_name)
    F = d[keep].drop(columns=["masc_other_senses"]).sort_values(["group", "masc"])
    out = ROOT / "data/wp_persons_fr_v0.csv"
    F.to_csv(out, index=False)
    print("candidates", len(d), "| gloss not in Spanish list", unmatched, "| ambiguous", ambiguous)
    print("kept by group:", F.groupby("group").size().to_dict())
    print("dropped: names", int((d.group != "").sum() - keep.sum()), "(names + identity place names)")
    for g in ("person", "occupation"):
        f = F[F.group == g]
        print(g, {c: int(f[c].sum()) for c in ("fem_also_noun", "masc_also_noun", "masc_identical_en", "place_name")},
              "plurals in 200k", int(f.pl_in_200k.sum()))  # fmt: skip
    print("sha256", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
