"""Deterministic coding of generated character paragraphs (no model judge).

Primary outcome per completion: `pron` = which third-person pronouns the ANSWER uses
  she  : only feminine forms (she/her/hers/herself)
  he   : only masculine forms (he/him/his/himself)
  they : only they-forms (they/them/their/theirs/themselves/themself), no he/she
  mixed: both he- and she-forms (often two characters, or a slip)
  none : no personal pronoun (e.g. 'it', second person, or describes the object)
Secondary: gendered person nouns (woman/man ...), CoT flags (talks about gender? about other
languages / grammatical gender? about ships being 'she'?).

Usage: python instruct/code_outputs.py RESULTS.jsonl [...] --out SUMMARY.csv
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

F = r"\b(she|her|hers|herself)\b"
M = r"\b(he|him|his|himself)\b"
T = r"\b(they|them|their|theirs|themselves|themself)\b"
IT = r"\b(it|its|itself)\b"
F_NOUNS = r"\b(woman|women|girl|lady|ladies|mother|grandmother|queen|matriarch|daughter|sister|wife|widow|heroine|actress|maiden|crone)\b"
M_NOUNS = r"\b(man|men|boy|gentleman|gentlemen|father|grandfather|king|patriarch|son|brother|husband|widower|hero|actor|lad|fellow|guy)\b"
COT_GENDER = r"\b(gender|male|female|masculine|feminine|man or (a )?woman|woman or (a )?man|he or she|she or he|pronouns?)\b"
COT_LANG = r"\b(german|spanish|french|italian|latin|grammatical(ly)? gender|grammatical gender)\b"
COT_SHIP_SHE = r"\bships?\b[^.]{0,60}\bshe\b|\bshe\b[^.]{0,60}\bships?\b"


def _n(pat: str, text: str) -> int:
    return len(re.findall(pat, text, flags=re.IGNORECASE))


def code(answer: str, think: str | None = None) -> dict:
    a = answer or ""
    f, m, t = _n(F, a), _n(M, a), _n(T, a)
    if f and m:
        pron = "mixed"
    elif f:
        pron = "she"
    elif m:
        pron = "he"
    elif t:
        pron = "they"
    else:
        pron = "none"
    first = re.search(f"{F}|{M}|{T}", a, flags=re.IGNORECASE)
    out = {
        "pron": pron,
        "n_f": f,
        "n_m": m,
        "n_they": t,
        "n_it": _n(IT, a),
        "first_pron": first.group(0).lower() if first else "",
        "n_fnoun": _n(F_NOUNS, a),
        "n_mnoun": _n(M_NOUNS, a),
        "answer_words": len(a.split()),
    }
    if think is not None:
        out |= {
            "cot_words": len(think.split()),
            "cot_gender": _n(COT_GENDER, think) > 0,
            "cot_lang": _n(COT_LANG, think) > 0,
            "cot_ship_she": bool(re.search(COT_SHIP_SHE, think, flags=re.IGNORECASE)),
        }
    return out


def split_think(raw: str) -> tuple[str | None, str, bool]:
    """Return (think, answer, think_closed). Qwen3 writes <think>...</think> then the answer."""
    if "</think>" in raw:
        th, ans = raw.split("</think>", 1)
        return th.replace("<think>", "").strip(), ans.strip(), True
    if "<think>" in raw:  # truncated mid-thought
        return raw.replace("<think>", "").strip(), "", False
    return None, raw.strip(), True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import pandas as pd

    rows = []
    for fp in args.files:
        for line in fp.read_text().splitlines():
            r = json.loads(line)
            c = code(r["answer"], r.get("think") if r["thinking"] else None)
            rows.append(
                {
                    k: r[k]
                    for k in ("model", "noun", "modifier", "thinking", "sample", "finish_reason")
                }
                | c
            )
    d = pd.DataFrame(rows)
    coded = args.out.with_name(args.out.stem + "_coded.csv")
    d.to_csv(coded, index=False)
    keys = ["model", "thinking", "noun", "modifier"]
    g = d.groupby(keys)
    s = g.size().rename("n").to_frame()
    for lab in ("she", "he", "they", "mixed", "none"):
        s[lab] = g["pron"].apply(lambda x, lab=lab: (x == lab).mean())
    s["share_f"] = g["pron"].apply(
        lambda x: (x == "she").sum() / max(1, ((x == "she") | (x == "he")).sum())
    )
    s["truncated"] = g["finish_reason"].apply(lambda x: (x == "length").mean())
    if "cot_gender" in d:
        for c in ("cot_gender", "cot_lang", "cot_ship_she"):
            s[c] = g[c].mean()
    s.reset_index().to_csv(args.out, index=False)
    print(s.reset_index().to_string(index=False))
    print(Counter(d["pron"]))


if __name__ == "__main__":
    main()
