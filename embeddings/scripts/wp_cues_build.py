"""Candidate adjective cues for the expanded word-pair test (docs/wordpair-cues-prereg.md, DRAFT).
Builds a list by rule for PI review; nothing is scored here.

Rules (Spanish and French, kaikki.org Wiktionary dumps, part of speech adj):
- headword = masculine singular, lowercase letters only; the feminine singular is the form tagged
  exactly ["feminine"]. Gendered if that form differs from the headword; invariant if the entry
  lists no separate feminine singular (e.g. feliz, whose plural is tagged feminine and masculine).
- English gloss = first sense's first gloss, first comma/semicolon item, parentheses removed; must
  be one word in the first 200k MUSE English words.
- both forms in the first 50k MUSE words of the language (the size of the rebuilt spaces).
- person-applicable: the English gloss is on a published list of person-describing adjectives, as
  collected by Garg et al. 2018 (github.com/nikhgarg/EmbeddingDynamicStereotypes, data/, fetched
  2026-11-11, copies + SHA256SUMS in /Volumes/Lexar/gbleed/shared/person_adjectives/): Williams &
  Best's Adjective Check List items (300), Katz-Braly/Princeton traits, and Garg's appearance,
  intelligence and sensitivity lists. The gloss may come from the first or second sense
  (`sense` column); a sense-2 match is flagged for review.
- stereotype columns: Williams & Best score (1990 row; the list's own masculinity-femininity
  scale) and Glasgow GEND where rated.
- excluded: senses tagged for register (colloquial, slang, figurative…) or region (Mexico, Spain…);
  gender-defining words (femenino/masculino, féminin/masculin); forms that are one of Zhou's occupation forms; superlatives/comparatives as headwords.
Flags for review: also_noun (the masculine form also has a noun entry), zhou_cue (in Zhou's 7).

    uv run --no-project --python 3.12 --with pandas python embeddings/scripts/wp_cues_build.py
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GIT = ROOT.parent
KK = Path("/Volumes/Lexar/gbleed/shared/kaikki")
MUSE = Path("/Volumes/Lexar/gbleed/shared/embeddings/muse")
DUMP = {"es": "Spanish", "fr": "French"}
ZHOU_CUES = {"es": {"trabajador", "bueno", "amable", "feliz", "fiable", "malo", "mejor"},
             "fr": {"travailleur", "bon", "aimable", "heureux", "fiable", "mauvais", "meilleur"}}  # fmt: skip
REGISTER = {"colloquial", "slang", "derogatory", "figuratively", "informal", "dated", "archaic",
            "obsolete", "rare", "vulgar", "dialectal", "pejorative", "humorous", "ironic"}
DEFINITIONAL = {"femenino", "masculino", "féminin", "masculin", "macho", "hembra", "mâle", "femelle"}
WORD = re.compile(r"^[a-zàâäáéèêëíìîïóòôöúùûüñçœæ]+$")


def vocab(lang, n):
    out = {}
    with open(MUSE / f"wiki.multi.{lang}.vec", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= n:
                break
            out.setdefault(line.split(" ", 1)[0], i)
    return out


def glasgow():
    g = pd.read_csv(GIT / "data/raw/glasgow/glasgow_norms_scott2019.csv", header=[0, 1])
    col = [i for i, c in enumerate(g.columns) if c[0] == "GEND"][0]
    return dict(zip(g.iloc[:, 0].astype(str).str.lower(), pd.to_numeric(g.iloc[:, col], errors="coerce")))


def gloss(entry):
    for s in entry.get("senses", []):
        for g in s.get("glosses", []):
            g = re.sub(r"\([^)]*\)", "", g)
            g = re.split(r"[,;]", g)[0].strip().lower()
            if g:
                return g
    return ""


def glosses(entry, k=2):
    out = []
    for s in entry.get("senses", [])[:k]:
        g = (s.get("glosses") or [""])[0]
        g = re.split(r"[,;]", re.sub(r"\([^)]*\)", "", g))[0].strip().lower()
        tags = set(s.get("tags") or [])
        if tags & REGISTER or any(t[:1].isupper() for t in tags):  # register or regional sense
            g = ""
        out.append(g)
    return out


def person_adjectives():
    P = Path("/Volumes/Lexar/gbleed/shared/person_adjectives")
    words = set()
    for f in ("adjectives_williamsbest.txt", "adjectives_princeton.txt", "adjectives_appearance.txt",
              "adjectives_intelligencegeneral.txt", "adjectives_sensitive.txt"):  # fmt: skip
        words |= {w.strip().lower() for w in open(P / f) if w.strip()}
    wb = pd.read_csv(P / "adjectives_williamsbest.csv")
    wb = wb[wb.year == 1990]
    words |= set(wb.word.str.lower())
    return words, dict(zip(wb.word.str.lower(), wb.score))


def main():
    en = vocab("en", 200_000)
    gl = glasgow()
    person, wbs = person_adjectives()
    mp = pd.read_csv(ROOT / "data/occupation_mapping_v1.csv")
    for lang in ("es", "fr"):
        V = vocab(lang, 50_000)
        occ = set(mp[mp.lang == lang].masc) | set(mp[mp.lang == lang].fem)
        nouns, adjs = set(), {}
        with open(KK / f"kaikki.org-dictionary-{DUMP[lang]}.jsonl", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                w, pos = j.get("word", ""), j.get("pos")
                if pos == "noun":
                    nouns.add(w)
                if pos != "adj" or not WORD.match(w) or w in adjs:
                    continue
                tags = [(x["form"], x.get("tags", [])) for x in j.get("forms", [])]
                g0 = " ".join(sum((s.get("glosses", []) for s in j.get("senses", [])[:1]), [])).lower()
                if re.match(r"(superlative|comparative|feminine|plural|masculine) (degree )?of ", g0):
                    continue
                fem = next((fm for fm, t in tags if t == ["feminine"]), None)
                adjs[w] = (fem if fem and fem != w else None, glosses(j))
        rows = []
        for m, (f, gs) in adjs.items():
            hit = next(((i + 1, g) for i, g in enumerate(gs) if g in person and g in en), None)
            if m not in V or (f and f not in V) or not hit or m in DEFINITIONAL:
                continue
            sense, e = hit
            if m in occ or (f and f in occ):
                continue
            rows.append({"lang": lang, "masc": m, "fem": f or "", "type": "gendered" if f else "invariant", "en": e,
                         "sense": sense, "wb_score_1990": wbs.get(e, ""), "glasgow_gend": gl.get(e, ""), "rank_m": V[m], "rank_f": V.get(f, "") if f else "",
                         "also_noun": m in nouns, "zhou_cue": m in ZHOU_CUES[lang], "pi_keep": ""})  # fmt: skip
        d = pd.DataFrame(rows).sort_values(["type", "rank_m"])
        d.to_csv(ROOT / f"data/wp_cues_{lang}_candidates_v0.csv", index=False)
        print(lang, d.groupby("type").size().to_dict(), "also_noun", int(d.also_noun.sum()),
              "zhou cues present", sorted(d[d.zhou_cue].masc))  # fmt: skip


if __name__ == "__main__":
    main()
