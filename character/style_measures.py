"""Model-free style measures and thinking-leakage flag (prereg v1 amendment 3, 2026-11-07).

Style (descriptive secondary for Q4; literature predicts thinking-on prose is more structured /
formal / less character-driven, Feng, Dou & Kong 2025 §4.1). Computed on the paragraph only (Name
line removed). Metrics fixed before any main-noun output:
- n_words: word count
- mean_sent_len: words per sentence (split on . ! ?)
- ttr_100: type-token ratio over the first 100 words (word variety; NaN if < 100 words)
- formal_markers: per 100 words, count of connective/structuring words (however, moreover,
  furthermore, therefore, thus, additionally, consequently, overall, essentially, ultimately,
  firstly, secondly, finally)
- list_like: 1 if the paragraph contains a line starting with a bullet/number or a colon-led list

Leakage (thinking-off cells): flag answers whose text BEFORE the paragraph proper looks like
reasoning: starts with a reasoning opener (okay, alright, let me, let's, first, i need to,
i should, hmm, so the user, the user wants) or contains "<think>" or "</think>".
"""

from __future__ import annotations

import re

FORMAL = {"however", "moreover", "furthermore", "therefore", "thus", "additionally",
          "consequently", "overall", "essentially", "ultimately", "firstly", "secondly", "finally"}  # fmt: skip
LEAK = re.compile(
    r"^\s*(okay|ok,|alright|let me|let's|first,|i need to|i should|hmm|so the user|the user wants)",
    re.IGNORECASE,
)


def style(para: str) -> dict:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", para or "")
    sents = [s for s in re.split(r"[.!?]+", para or "") if re.search(r"[A-Za-z]", s)]
    first = [w.lower() for w in words[:100]]
    return {
        "n_words": len(words),
        "mean_sent_len": round(len(words) / len(sents), 2) if sents else float("nan"),
        "ttr_100": round(len(set(first)) / 100, 3) if len(first) == 100 else float("nan"),
        "formal_markers": round(100 * sum(w.lower() in FORMAL for w in words) / len(words), 3)
        if words
        else float("nan"),
        "list_like": int(bool(re.search(r"(?m)^\s*([-*•]|\d+[.)])\s", para or ""))),
    }


def leaked(answer: str) -> bool:
    a = answer or ""
    return bool(LEAK.match(a)) or "<think>" in a or "</think>" in a
