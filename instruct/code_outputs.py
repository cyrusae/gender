"""Deterministic coding of generated character paragraphs (no model judge).

Primary outcome per completion: `pron` = which third-person pronouns the ANSWER uses
  she  : only feminine forms (she/her/hers/herself)
  he   : only masculine forms (he/him/his/himself)
  they : only they-forms (they/them/their/theirs/themselves/themself), no he/she
  mixed: both he- and she-forms (often two characters, or a slip)
  it   : no he/she/they, and >= 2 it-forms: the character is called "it" (non-human character)
  none : no pronoun for the character (fewer than 2 it-forms)
Validity gates (v2 prompts): `noun_hit` (noun or plural as a whole word in the paragraph),
`name_ok` (last non-empty line is "Name: ..."), `truncated`, `error`. `valid` = all pass.
The paragraph (text before the Name line) is what gets pronoun-coded. The name's first given
name is looked up in SSA US baby names (all years): f / m (>= 95% one sex, the P24b rule) /
ambiguous / notfound.
Secondary: gendered person nouns (woman/man ...), CoT flags (talks about gender? about other
languages / grammatical gender? about ships being 'she'?).

Usage: python instruct/code_outputs.py RESULTS.jsonl [...] --out SUMMARY.csv
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import subprocess
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
    elif _n(IT, a) >= 2:  # person-character called "it" (one stray "it" can mean an object)
        pron = "it"
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


NAME_LINE = re.compile(r"^\W*name\W*:\W*(.+?)\W*$", re.IGNORECASE)
TITLES = {"mr", "mrs", "ms", "miss", "mx", "dr", "lady", "lord", "sir", "dame", "madame", "madam",
          "master", "mister", "old", "aunt", "auntie", "uncle", "grandma", "grandpa", "granny",
          "mother", "father", "brother", "sister", "mama", "papa", "the", "professor", "captain",
          "man", "woman", "young", "little", "big"}  # fmt: skip


def plural(noun: str) -> str:
    if re.search(r"(s|x|z|ch|sh)$", noun):
        return noun + "es"
    if re.search(r"[^aeiou]y$", noun):
        return noun[:-1] + "ies"
    return noun + "s"


def split_name(answer: str) -> tuple[str, str | None]:
    """(paragraph, name or None). The name must be the last non-empty line."""
    lines = (answer or "").rstrip().splitlines()
    if lines and (m := NAME_LINE.match(lines[-1].strip())):
        return "\n".join(lines[:-1]).strip(), m.group(1).strip().strip("*_\"'")
    return (answer or "").strip(), None


@functools.cache
def ssa_table() -> dict[str, tuple[int, int]]:
    """{Name: (female births, male births)} summed over all SSA years."""
    cands = [os.environ.get("GBLEED_SSA_DIR", ""), "data/raw/ssa/names", "../data/raw/ssa/names"]
    try:
        common = subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], text=True
        ).strip()
        cands.append(str(Path(common).parent / "data/raw/ssa/names"))
    except (OSError, subprocess.CalledProcessError):
        pass
    d = next((Path(c) for c in cands if c and list(Path(c).glob("yob*.txt"))), None)
    if d is None:
        raise SystemExit("SSA names not found (set GBLEED_SSA_DIR to the dir with yob*.txt)")
    t: dict[str, list[int]] = {}
    for f in sorted(d.glob("yob*.txt")):
        for ln in f.read_text().splitlines():
            nm, sex, k = ln.split(",")
            t.setdefault(nm, [0, 0])[sex == "M"] += int(k)
    return {k: (v[0], v[1]) for k, v in t.items()}


def code_name(name: str | None) -> dict:
    if not name:
        return {"name": "", "given": "", "title": "", "name_sex": "", "name_f_share": None}
    toks = re.findall(r"[A-Za-z][A-Za-z'\-]*", name)
    titles = [t for t in toks if t.lower().rstrip(".") in TITLES]
    given = next((t for t in toks if t.lower().rstrip(".") not in TITLES), "")
    f, m = ssa_table().get(given[:1].upper() + given[1:].lower(), (0, 0)) if given else (0, 0)
    sh = f / (f + m) if f + m else None
    sex = "notfound" if sh is None else "f" if sh >= 0.95 else "m" if sh <= 0.05 else "ambiguous"
    return {"name": name, "given": given, "title": " ".join(titles), "name_sex": sex,
            "name_f_share": sh}  # fmt: skip


def code_v2(answer: str, noun: str, finish_reason: str, think: str | None = None) -> dict:
    para, name = split_name(answer)
    hit = re.search(rf"\b({re.escape(noun)}|{re.escape(plural(noun))})\b", para, re.IGNORECASE)
    gates = {
        "noun_hit": bool(hit),
        "name_ok": name is not None,
        "truncated": finish_reason == "length",
        "error": finish_reason == "error",
    }
    gates["valid"] = gates["name_ok"] and not (
        gates["noun_hit"] or gates["truncated"] or gates["error"]
    )
    n_par = len([b for b in re.split(r"\n\s*\n", para) if b.strip()])
    nm = code_name(name)
    nm["noun_in_name"] = bool(name and re.search(rf"\b{re.escape(noun)}", name, re.IGNORECASE))
    return gates | {"n_paragraphs": n_par} | nm | code(para, think)


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
            th = r.get("think") if r["thinking"] else None
            if "Name:" in r["prompt"]:
                c = code_v2(r["answer"], r["noun"], r["finish_reason"], th)
            else:
                c = code(r["answer"], th)
            keep = ("model", "noun", "modifier", "thinking", "sample", "finish_reason", "provider")
            rows.append({k: r.get(k) for k in keep} | c)
    d = pd.DataFrame(rows)
    coded = args.out.with_name(args.out.stem + "_coded.csv")
    d.to_csv(coded, index=False)
    keys = ["model", "thinking", "noun", "modifier"]
    s = d.groupby(keys).size().rename("n").to_frame()
    if "valid" in d:
        g0 = d.groupby(keys)
        for c in ("valid", "noun_hit", "name_ok", "truncated", "error"):
            s[c] = g0[c].mean()
        s["mean_paragraphs"] = g0["n_paragraphs"].mean()
        d = d[d["valid"]]  # outcome shares below are among valid outputs only
        s["n_valid"] = d.groupby(keys).size()
    g = d.groupby(keys)
    for lab in ("she", "he", "they", "mixed", "it", "none"):
        s[lab] = g["pron"].apply(lambda x, lab=lab: (x == lab).mean())
    s["share_f"] = g["pron"].apply(
        lambda x: (x == "she").sum() / max(1, ((x == "she") | (x == "he")).sum())
    )
    if "valid" not in s:
        s["truncated"] = g["finish_reason"].apply(lambda x: (x == "length").mean())
    if "name_sex" in d:
        for lab in ("f", "m", "ambiguous", "notfound"):
            s[f"name_{lab}"] = g["name_sex"].apply(lambda x, lab=lab: (x == lab).mean())
    if "cot_gender" in d:
        for c in ("cot_gender", "cot_lang", "cot_ship_she"):
            s[c] = g[c].mean()
    s.reset_index().to_csv(args.out, index=False)
    print(s.reset_index().to_string(index=False))
    print(Counter(d["pron"]))


if __name__ == "__main__":
    main()
