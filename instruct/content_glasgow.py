"""Model-free content score: mean Glasgow gender rating of the words in a paragraph
(1 = very feminine, 7 = very masculine; Scott et al. 2019). Human ratings only.

Masked before scoring (so the score reflects the description, not the pronoun choice):
pronouns, gendered person nouns (woman/man...), the character's name and any capitalised word,
and the study noun. Words are lowercased and matched to Glasgow headwords; the single-sense
entry is used (Glasgow lists some words with senses in parentheses, which are skipped).
Reports n_rated (how many tokens had a rating) so thin paragraphs can be filtered.
"""

from __future__ import annotations

import csv
import functools
import re
import subprocess
from pathlib import Path

GLASGOW = Path(__file__).resolve().parents[1] / "data/raw/glasgow/glasgow_norms_scott2019.csv"
MASK = {"he", "him", "his", "himself", "she", "her", "hers", "herself", "they", "them", "their",
        "theirs", "themselves", "themself", "woman", "women", "man", "men", "girl", "boy", "lady",
        "gentleman", "mother", "father", "wife", "husband", "queen", "king", "sister", "brother",
        "daughter", "son", "female", "male", "feminine", "masculine", "name"}  # fmt: skip


@functools.cache
def gend_table() -> dict[str, float]:
    p = GLASGOW
    if not p.exists():  # worktree: data/raw lives in the main checkout
        common = subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], text=True
        ).strip()
        p = Path(common).parent / "data/raw/glasgow/glasgow_norms_scott2019.csv"
    rows = list(csv.reader(p.open()))[2:]
    return {r[0].lower(): float(r[26]) for r in rows if r[0] and "(" not in r[0]}


def glasgow_score(para: str, noun: str = "", name: str = "") -> dict:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", para or "")
    skip = MASK | {noun.lower(), noun.lower() + "s"} | {p.lower() for p in name.split()}
    g = gend_table()
    vals = [g[w.lower()] for i, w in enumerate(words)
            if w.lower() not in skip and not (w[0].isupper() and i > 0) and w.lower() in g]  # fmt: skip
    return {"glasgow_gend_mean": round(sum(vals) / len(vals), 3) if vals else None,
            "glasgow_n_rated": len(vals), "n_words": len(words)}  # fmt: skip
