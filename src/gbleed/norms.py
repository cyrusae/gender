"""Human concreteness ratings (Brysbaert, Warriner & Kuperman 2014, Behavior Research Methods
46:904-911): ~40k English words rated 1 (abstract) to 5 (concrete).

The authors' download link is dead; we use a third-party mirror and verify it against the paper
(39,954 entries = 37,058 words + 2,896 two-word expressions) and a fixed SHA-256.

Spanish/German nouns are rated via their English gloss: every comma/semicolon-separated
alternative of the first gloss is looked up (British spellings normalised, but only when WordNet
confirms the two spellings are the same word), and matches are averaged. Fallback: the head noun of the first alternative, taking the word after "of" when the
first word is only a classifier ("type of pepper" -> pepper).
"""

from __future__ import annotations

import hashlib
import re
import urllib.request
from functools import cache
from pathlib import Path

import pandas as pd

PATH = Path("data/raw/norms/Concreteness_ratings_Brysbaert_et_al_BRM.txt")
MIRROR = (
    "https://raw.githubusercontent.com/ArtsEngine/concreteness/master/"
    "Concreteness_ratings_Brysbaert_et_al_BRM.txt"
)
SHA256 = "0b4082dbd38585b0ee1fd258145b7a50592f8d0d98e5fc6b6844ceef3cd8ecc8"
CLASSIFIERS = {"type", "kind", "sort", "variety", "piece", "form", "diminutive", "species",
               "genus", "variant"}  # fmt: skip
UK_US = [(r"our", "or"), (r"(?<=[^aeiou])re$", "er"), (r"ise$", "ize"), (r"isation$", "ization"),
         (r"ogue$", "og"), (r"ae", "e"), (r"oe", "e")]  # fmt: skip


@cache
def ratings() -> dict[str, float]:
    if not PATH.exists():
        PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MIRROR, PATH)
    if hashlib.sha256(PATH.read_bytes()).hexdigest() != SHA256:
        raise ValueError(f"{PATH}: checksum mismatch (not the verified copy)")
    df = pd.read_csv(PATH, sep="\t", keep_default_na=False)
    assert len(df) == 39954, "unexpected entry count"
    return dict(zip(df.Word.str.lower(), df["Conc.M"].astype(float), strict=True))


def _same_word(a: str, b: str) -> bool:
    """WordNet lists British spellings as synonyms of the American entry (colour/color share a
    synset; scourer/scorer and boeing/being don't), so only accept a respelling it confirms."""
    from .lexicon import _wordnet

    wn = _wordnet()
    return bool(set(wn.synsets(a.replace(" ", "_"))) & set(wn.synsets(b.replace(" ", "_"))))


def _lookup(phrase: str, R: dict) -> float | None:
    phrase = " ".join(phrase.split())  # normalise whitespace ("indigo  " -> "indigo")
    if phrase in R:
        return R[phrase]
    words = phrase.split()
    for pat, rep in UK_US:  # normalise the last word's spelling (neighbourhood, sepulchre)
        alt = " ".join([*words[:-1], re.sub(pat, rep, words[-1])]) if words else phrase
        if alt != phrase and alt in R and _same_word(phrase, alt):
            return R[alt]
    return None


def rate(gloss: str) -> tuple[float | None, str]:
    """(rating, how) with how in {'gloss', 'head', 'none'}."""
    R = ratings()
    g = re.sub(r"\([^)]*\)", "", str(gloss))
    alts = [
        re.sub(r"^(a|an|the|to)\s+", "", a.strip().lower()).rstrip(".")
        for a in re.split(r"[,;/]", g)
    ]
    alts = [a for a in alts if a]
    hits = [v for a in alts if (v := _lookup(a, R)) is not None]
    if hits:
        return sum(hits) / len(hits), "gloss"
    if alts:
        first = alts[0]
        parts = first.split(" of ")
        head_words = parts[0].split()
        if len(parts) > 1 and head_words and head_words[-1] in CLASSIFIERS:
            head = parts[1].split()[0] if parts[1].split() else ""
        else:
            head = head_words[-1] if head_words else ""
        v = _lookup(head, R) if head else None
        if v is not None:
            return v, "head"
    return None, "none"
