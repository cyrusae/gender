"""English female/male person-word pairs (P24c, PI 2026-10-12): candidates from human-made sources,
automatic checks, then PI review (docs/reports/english-gender-pairs-review.csv).

Sources (pairing only; each word's sex comes from its WordNet person-sense definition):
  Wiktionary (kaikki English dump): first current noun sense "female/male equivalent of X",
    "a female/male X", or two words whose first glosses differ only by female <-> male words
  WordNet: definition "a female/woman X", or mirror definitions (woman <-> man, wife <-> husband...)
  Bolukbasi et al. (2016) equalize_pairs.json (order not consistent in the file: re-oriented)
Checks per word: one alphabetic word, singular (WordNet morphy = itself), Zipf >= 3, not a social-
direction word, a person sense among its first two WordNet noun senses ("person sense"); flagged
(shown, PI may rescue) if that sense is not >= 50% of the word's tagged WordNet usage. Rows where
a word has no person sense at all are not shown (pronouns, plurals, animals, mislinks).
Sex of a word = first sex word in its person-sense definition (f / m / generic = definition has
none); pair type = female vs male | female vs generic | unknown.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import paths

KAIKKI_EN = "kaikki.org-dictionary-English.jsonl"  # in paths.kaikki_dir()
CACHE = "en_gender_pairs.json"  # Wiktionary candidates (slow to extract)
BOLUKBASI = "data/raw/bolukbasi/equalize_pairs.json"  # github.com/tolga-b/debiaswe (MIT)
SHEET = "docs/reports/english-gender-pairs-review.csv"
F_WORDS = {"female", "woman", "women", "girl", "her", "she", "wife", "mother", "daughter",
           "sister", "lady"}  # fmt: skip
M_WORDS = {"male", "man", "men", "boy", "his", "he", "husband", "father", "son", "brother",
           "gentleman"}  # fmt: skip
SWAP = {"female": "male", "woman": "man", "women": "men", "girl": "boy", "her": "his",
        "she": "he", "wife": "husband", "mother": "father", "daughter": "son",
        "sister": "brother", "lady": "gentleman"}  # fmt: skip
SWAP |= {v: k for k, v in SWAP.items()}
SKIP_TAGS = {"obsolete", "archaic", "alt-of", "form-of", "misspelling"}
GENERIC_X = {"person", "human", "animal", "one", "individual", "child", "adult"}


def _swap(g: str) -> str:
    return re.sub(r"[a-z]+", lambda t: SWAP.get(t.group(0), t.group(0)), g)


def wiktionary_candidates() -> list:
    """[(word_a, word_b, rule)] from the English dump; cached (the dump is 3.3 GB)."""
    if paths.kaikki(CACHE).exists():
        return json.loads(paths.kaikki(CACHE).read_text())
    first, out = {}, []
    with open(paths.kaikki(KAIKKI_EN)) as f:
        for line in f:
            if not line.startswith('{"pos": "noun"'):
                continue
            d = json.loads(line)
            w = d.get("word", "")
            if d.get("lang_code") != "en" or not (w.isalpha() and w.islower()) or w in first:
                continue
            for s in d.get("senses", []):
                if set(s.get("tags", [])) & SKIP_TAGS or not s.get("glosses"):
                    continue
                g = re.sub(r"\s+", " ", s["glosses"][0].lower().strip(" .;"))
                first[w] = g
                m = re.match(r"^(?:the )?(?:female|male) (?:equivalent|counterpart|form) of "
                             r"(?:an? )?([a-z]+)$", g) or re.match(r"^an? (?:female|male) ([a-z]+)$", g)  # fmt: skip
                if m and m.group(1) not in GENERIC_X:
                    out.append((w, m.group(1), "Wiktionary 'female/male (equivalent of) X'"))
                break
    by = defaultdict(list)
    for w, g in first.items():
        by[g].append(w)
    for w, g in first.items():
        s = _swap(g)
        if s != g:
            out += [(w, m, "Wiktionary mirror glosses") for m in by.get(s, []) if m != w]
    paths.kaikki(CACHE).write_text(json.dumps(out))
    return out


def wordnet_candidates(wn) -> list:
    first = {}
    for ss in wn.all_synsets("n"):
        for lem in ss.lemmas():
            w = lem.name()
            if w.isalpha() and w.islower() and w not in first and wn.synsets(w, "n")[0] == ss:
                first[w] = ss.definition().lower()
    by = defaultdict(list)
    for w, g in first.items():
        by[g].append(w)
    out = []
    for w, g in first.items():
        m = re.match(r"^an? (?:female|woman) ([a-z]+)$", g)
        if m:
            out.append((w, m.group(1), "WordNet 'a female X'"))
        s = _swap(g)
        if s != g:
            out += [(w, m, "WordNet mirror definitions") for m in by.get(s, []) if m != w]
    return out


def person_sense(wn, w: str):
    person = wn.synset("person.n.01")
    for s in [s for s in wn.synsets(w, "n") if w in [x.name() for x in s.lemmas()]][:2]:
        if person in {h for path in s.hypernym_paths() for h in path}:
            return s
    return None


def sex_of(wn, w: str) -> str:
    s = person_sense(wn, w)
    if s is None:
        return "unknown"
    for t in re.findall(r"[a-z]+", s.definition().lower()):
        if t in F_WORDS:
            return "f"
        if t in M_WORDS:
            return "m"
    return "generic"


def _mixed(wn, w: str) -> str:
    """Flag a definition that mentions both sexes (its sex label is then a guess: grandchild =
    "a child of your son or daughter")."""
    s = person_sense(wn, w)
    t = set(re.findall(r"[a-z]+", s.definition().lower())) if s else set()
    return f"{w}: definition mentions both sexes" if t & F_WORDS and t & M_WORDS else ""


def check(wn, w: str) -> tuple[bool, str]:
    """(shown, flag): shown = could be used at all; flag = reason a PI rescue is needed."""
    from wordfreq import zipf_frequency

    from .english import SOCIAL_PAIRS, _dominant

    if not w.isalpha() or not w.islower():
        return False, "not one lowercase word"
    if wn.morphy(w, "n") not in (None, w):
        return False, "plural/inflected form"
    if w in {x for p in SOCIAL_PAIRS for x in p}:
        return False, "social-direction word"
    if zipf_frequency(w, "en") < 3:
        return False, "rare (Zipf < 3)"
    s = person_sense(wn, w)
    if s is None:
        return False, "no person sense"
    tot = sum(x.count() for ss in wn.synsets(w) for x in ss.lemmas() if x.name() == w)
    if tot and not _dominant(wn, w, s):
        return True, f"{w}: person sense not the dominant use"
    return True, ""


def build_sheet(pi_marks: str | None = None) -> pd.DataFrame:
    """The review sheet; `pi_marks` = an earlier sheet whose keep marks carry over (by pair)."""
    from .lexicon import _wordnet

    wn = _wordnet()
    src = defaultdict(set)
    cands = [*wiktionary_candidates(), *wordnet_candidates(wn)]
    cands += [(a.lower(), b.lower(), "Bolukbasi et al. 2016")
              for a, b in json.loads(Path(BOLUKBASI).read_text())]  # fmt: skip
    for a, b, r in cands:
        if a != b:
            src[frozenset((a, b))].add(r)
    rows = []
    for pair, rules in src.items():
        if len(pair) != 2:
            continue
        a, b = sorted(pair)
        (sa, fa), (sb, fb) = check(wn, a), check(wn, b)
        if not (sa and sb):
            continue
        xa, xb = sex_of(wn, a), sex_of(wn, b)
        if xb == "f" and xa != "f":
            a, b, xa, xb, fa, fb = b, a, xb, xa, fb, fa
        typ = ("female vs male" if (xa, xb) == ("f", "m") else
               "female vs generic" if (xa, xb) == ("f", "generic") else "unknown")  # fmt: skip
        flags = "; ".join(x for x in (fa, fb, _mixed(wn, a), _mixed(wn, b)) if x)
        rows.append({"female": a, "male_or_generic": b, "type": typ,
                     "status": "kept" if not flags and typ != "unknown" else "flagged",
                     "flag": flags if typ != "unknown" else f"sexes {xa}/{xb}; {flags}".strip("; "),
                     "source": " + ".join(sorted(rules)),
                     "female_definition": person_sense(wn, a).definition(),
                     "male_definition": person_sense(wn, b).definition(),
                     "keep (PI: y/n)": ""})  # fmt: skip
    keyc = "keep (PI: y/n)"
    if pi_marks:
        # PI-edited or PI-added pairs not found by the sources: checked like any other
        old = pd.read_csv(pi_marks, keep_default_na=False)
        have = {frozenset((r["female"], r["male_or_generic"])) for r in rows}
        for r in old.to_dict("records"):
            a, b = r["female"], r["male_or_generic"]
            if frozenset((a, b)) in have or r[keyc] == "n":
                continue
            (sa, fa), (sb, fb) = check(wn, a), check(wn, b)
            if sa and sb:
                rows.append({"female": a, "male_or_generic": b, "type": "PI pair",
                             "status": "flagged" if fa or fb else "kept",
                             "flag": "; ".join(x for x in (fa, fb) if x), "source": "PI",
                             "female_definition": person_sense(wn, a).definition(),
                             "male_definition": person_sense(wn, b).definition(), keyc: ""})  # fmt: skip
    df = pd.DataFrame(rows).sort_values(["status", "type", "female"], ascending=[False, True, True])
    if pi_marks:
        mark = {frozenset((r["female"], r["male_or_generic"])): r[keyc]
                for r in old.to_dict("records") if r[keyc]}  # fmt: skip
        df[keyc] = [mark.get(frozenset((a, b)), "")
                    for a, b in zip(df.female, df.male_or_generic, strict=True)]  # fmt: skip
    df.to_csv(SHEET, index=False)
    print(df.groupby(["status", "type"]).size().to_string())
    return df
