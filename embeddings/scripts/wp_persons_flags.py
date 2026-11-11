"""Masculine-side flags for the person-noun candidates (docs/wordpair-persons-prereg.md, DRAFT).
Written to data/wp_persons_es_flags_v0.csv so the PI's review of the candidate list isn't touched;
merged by masc at freeze. Outcome-blind: nothing is scored here.

- masc_also_noun: the masculine form has a Spanish Wiktionary noun sense that isn't a person sense
  (gloss matches neither the person pattern nor any English person word from the candidate glosses
  or Bolukbasi's professions list, and isn't "male equivalent of ..."), e.g. a thing or field.
- language_name: one of those senses names a language (el portugués 'Portuguese language').
- place_name (added 2026-11-11, PI): either form is also a Spanish place name in Wiktionary (country,
  region, city…), e.g. argentina, siria: a pos "name" entry whose glosses mention
  country/region/city/province/state/island/kingdom/etc.
  PI: drop identity nouns with this flag.
- masc_identical_en: the masculine form is spelled exactly like its English gloss (doctor, director).
  Spanish Wikipedia also contains English uses of such strings, which can pull the Spanish vector
  toward the English word and favour the masculine form for a reason unrelated to gender.

    uv run --no-project --python 3.12 --with pandas python embeddings/scripts/wp_persons_flags.py
"""

import json
import re
from pathlib import Path

import pandas as pd
from wp_cues_build import KK
from wp_persons_build import PERSON

ROOT = Path(__file__).resolve().parents[1]
PLACE = re.compile(r"\b(country|region|city|province|state|island|peninsula|kingdom|republic|town|area|territory|continent|capital)\b")
MALE_EQ = re.compile(r"^(male|masculine) (equivalent|form|version) of")


def main():
    d = pd.read_csv(ROOT / "data/wp_persons_es_candidates_v0.csv", keep_default_na=False)
    want = set(d.masc)
    prof = {x[0] for x in json.load(open(ROOT / "external/debiaswe/data/professions.json"))}
    pwords = set(d.en.str.lower()) | prof | {w + "s" for w in prof} | {"woman", "women", "wife", "girl", "lady", "men"}
    other, lang, place = {}, {}, set()
    forms = want | set(d.fem)
    with open(KK / "kaikki.org-dictionary-Spanish.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"pos": "name"' in line:
                j = json.loads(line)
                g = " ".join(" ".join(s.get("glosses", [])) for s in j.get("senses", [])).lower()
                if j.get("word", "").lower() in forms and PLACE.search(g):
                    place.add(j["word"].lower())
                continue
            if '"pos": "noun"' not in line:
                continue
            j = json.loads(line)
            w = j.get("word")
            if w not in want:
                continue
            for s in j.get("senses", []):
                g = " ".join(s.get("glosses", [])).lower()
                if not g or MALE_EQ.match(g) or PERSON.search(g) or set(re.findall(r"[a-z]+", g)) & pwords:
                    continue
                other.setdefault(w, []).append(g[:60])
                if re.match(r"^(the |a )?([a-z-]+ ){0,3}(language|dialect)\b", g) or re.search(r"language (spoken|of)|\((language|language family|dialect)\)", g):
                    lang[w] = True
    out = pd.DataFrame({"masc": d.masc, "en": d.en,
                        "masc_also_noun": d.masc.isin(other), "language_name": d.masc.isin(lang),
                        "masc_identical_en": d.masc == d.en.str.lower(),
                        "masc_other_senses": d.masc.map(lambda w: " | ".join(other.get(w, [])[:3])),
                        "place_name": d.masc.isin(place) | d.fem.isin(place),
                        "place_form": [" ".join(w for w in (m, f) if w in place) for m, f in zip(d.masc, d.fem)]})  # fmt: skip
    out.to_csv(ROOT / "data/wp_persons_es_flags_v0.csv", index=False)
    print({c: int(out[c].sum()) for c in ("masc_also_noun", "language_name", "masc_identical_en", "place_name")})
    print("identical:", " ".join(out[out.masc_identical_en].masc))
    print("language:", " ".join(out[out.language_name].masc))


if __name__ == "__main__":
    main()
