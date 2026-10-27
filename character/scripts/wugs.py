"""Wug (nonce) modifiers for the character prompt, generated with Wuggy (no models).

Recipe fixed before generating (2026-10-26):
- Templates: common English adjectives (WordNet adjective, not also a noun sense first; zipf >= 4),
  exactly 2 syllables, 5-7 letters, so the nonce words are adjective-shaped.
- Wuggy orthographic_english candidates; keep lowercase a-z, not a WordNet lemma, zero wordfreq in
  en + the 8 study languages (de ru es hi ar fi ja tr; non-Latin scripts can't collide, kept for
  the record) and fr/it/pt; not within edit distance 1 of any English adjective with zipf >= 2.5
  (so it can't be read as a typo); not containing the substring of a pilot noun.
- Sound score (orthographic proxy): soft = share of consonants in {l m n r w y} + share of vowels
  in {e i}; hard = share of consonants in {p t k b d g} + share of vowels in {o u a}.
  score = soft - hard. Soft wugs: highest scores; hard wugs: lowest. Picks are matched on letter
  count and syllables (2), and must not share an ending with the other group's picks.
v2 (2026-10-26, after seeing v1 candidates, before any model): v1 hard picks were all vowel-initial
(taking "an", an article confound) and soft picks ended in -er/-en (read as comparatives/participles);
"a" counted as a back vowel is phonetically wrong for English spelling. Fixes: consonant-initial
only, no inflection-like endings, "a" neutral.
Output: character/stimuli/wug_candidates_v1.csv (top 10 per side, for review); final picks are made
from it by rank, not by hand.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gbleed.lexicon import _wordnet
from gbleed.nonce5 import _generate, _generator, _syllables

LANGS = ("en", "de", "es", "fr", "it", "pt", "ru", "hi", "ar", "fi", "ja", "tr")
SOFT_C, HARD_C = set("lmnrwy"), set("ptkbdg")
SOFT_V, HARD_V = set("ei"), set("ou")  # v2: "a" neutral (often front /ae/)
MORPH_END = re.compile(r"(er|en|ed|es|s|ly|ing|est|ish)$")  # v2: reads as English inflection
PILOT = ("pot", "keyboard", "tree", "door", "flower", "hammer", "ship")


def edit1(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    s, t = (a, b) if len(a) < len(b) else (b, a)
    return any(t[:i] + t[i + 1 :] == s for i in range(len(t)))


def score(w: str) -> float:
    cons = [c for c in w if c not in "aeiouy"]
    vows = [c for c in w if c in "aeiou"]
    sc = sum(c in SOFT_C for c in cons) / max(len(cons), 1)
    hc = sum(c in HARD_C for c in cons) / max(len(cons), 1)
    sv = sum(c in SOFT_V for c in vows) / max(len(vows), 1)
    hv = sum(c in HARD_V for c in vows) / max(len(vows), 1)
    return round(sc + sv - hc - hv, 3)


def main() -> None:
    wn = _wordnet()
    adjs = {
        lem.name().lower()
        for s in wn.all_synsets("a")
        for lem in s.lemmas()
        if re.fullmatch(r"[a-z]+", lem.name())
    }
    common_adj = {a for a in adjs if zipf_frequency(a, "en") >= 2.5}
    templates = sorted(
        a
        for a in common_adj
        if zipf_frequency(a, "en") >= 4
        and _syllables(a) == 2
        and 5 <= len(a) <= 7
        and not (wn.synsets(a, pos="n") and wn.synsets(a)[0].pos() == "n")
    )
    g = _generator("orthographic_english")
    res, skipped = _generate(g, templates, 10)
    print(f"templates {len(templates)}, skipped {len(skipped)}, raw candidates {len(res)}")
    rows, seen = [], set()
    for r in res:
        w = r["pseudoword"]
        if not w or w in seen or not re.fullmatch(r"[a-z]+", w):
            continue
        seen.add(w)
        if wn.synsets(w) or any(zipf_frequency(w, lg) > 0 for lg in LANGS):
            continue
        if _syllables(w) != 2 or any(p in w for p in PILOT):
            continue
        if w[0] in "aeiou" or MORPH_END.search(
            w
        ):  # v2: article "a" on both sides; no fake morphology
            continue
        if any(edit1(w, a) for a in common_adj if abs(len(a) - len(w)) <= 1):
            continue
        rows.append({"wug": w, "template": r["word"], "len": len(w), "score": score(w)})
    d = pd.DataFrame(rows).sort_values("score")
    print(f"pass all filters: {len(d)}")
    out = pd.concat([d.head(10).assign(side="hard"), d.tail(10).iloc[::-1].assign(side="soft")])
    out["source"] = "wuggy orthographic_english; filters in character/scripts/wugs.py"
    path = Path(__file__).parents[1] / "stimuli/wug_candidates_v1.csv"
    d.iloc[::-1].assign(source=out.source.iloc[0]).to_csv(
        path.with_name("wug_candidates_v1_all.csv"), index=False
    )
    out.to_csv(path, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
