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


# Neopronoun sets (PI 2026-10-26). A set counts only if an ANCHOR form appears; WEAK forms (also
# ordinary words: 'em = them, I've, vis-a-vis, fae = fairies, German "sie", "e.g.") are counted only
# alongside an anchor from the same set. (?<!['\u2019]) keeps contractions out.
NEO = {
    "xe": ({"xe", "xem", "xyr", "xyrs", "xemself", "xir", "xirs", "xirself"}, set()),
    "ze_zie_sie": ({"ze", "zie", "zir", "zirs", "zirself", "hir", "hirs", "hirself"}, {"sie"}),
    "spivak": ({"ey", "eir", "eirs", "emself"}, {"em", "e", "es"}),
    "fae": ({"faer", "faers", "faeself"}, {"fae"}),
    "ve": ({"ver", "verself"}, {"ve", "vis"}),
    "thon": ({"thon", "thons", "thonself"}, set()),
}


def neo_count(text: str) -> tuple[int, str]:
    """(total neopronoun forms, '|'-joined sets found)."""
    words = re.findall(r"(?<!['\u2019])\b[a-z]+\b", (text or "").lower())
    tot, sets = 0, []
    for name, (anchors, weak) in NEO.items():
        a = sum(w in anchors for w in words)
        if a:
            tot += a + sum(w in weak for w in words)
            sets.append(name)
    return tot, "|".join(sets)


def _n(pat: str, text: str) -> int:
    return len(re.findall(pat, text, flags=re.IGNORECASE))


def code(answer: str, think: str | None = None) -> dict:
    a = answer or ""
    f, m, t = _n(F, a), _n(M, a), _n(T, a)
    neo, neo_sets = neo_count(a)
    if f and m:
        pron = "mixed"
    elif f:
        pron = "she"
    elif m:
        pron = "he"
    elif neo:
        pron = "neo"
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
        "n_neo": neo,
        "neo_sets": neo_sets,
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
NAME_INLINE = re.compile(
    r"(?<=[.!?\"\u201d])\s+\**Name\**\s*:\s*\**([^.!?\n]{1,60}?)\W*$"
)  # capital N only
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


def split_name(answer: str, pos: str = "after") -> tuple[str, str | None]:
    """(paragraph, name or None). The Name line must be the last (pos='after') or first
    (pos='before') non-empty line."""
    lines = [ln for ln in (answer or "").strip().splitlines()]
    i = -1 if pos == "after" else 0
    if lines and (m := NAME_LINE.match(lines[i].strip())):
        rest = lines[:-1] if pos == "after" else lines[1:]
        return "\n".join(rest).strip(), m.group(1).strip().strip("*_\"'")
    if pos == "after" and lines and (m := NAME_INLINE.search(lines[-1])):
        # inline case (PI 2026-10-26): "... asking for attention herself. Name: Mira"
        para = "\n".join([*lines[:-1], lines[-1][: m.start()]]).strip()
        return para, m.group(1).strip().strip("*_\"'")
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


CAP_WORD = re.compile(r"[A-Z][a-z'\u2019\-]+")


CLOSED_CLASS = {"she", "her", "hers", "he", "him", "his", "they", "them", "their", "it", "its",
                "the", "an", "a", "this", "that", "these", "those", "there", "then", "when",
                "while", "though", "although", "even", "yet", "but", "and", "or", "so", "if",
                "as", "at", "in", "on", "with", "without", "from", "to", "for", "of", "by",
                "every", "each", "any", "all", "some", "no", "not", "one", "most", "more"}  # fmt: skip


@functools.cache
def _wn():
    from gbleed.lexicon import _wordnet

    return _wordnet()


@functools.cache
def is_common_word(w: str) -> bool:
    """True if the lowercase form (or its base form) is a lowercase WordNet lemma or a closed-class
    word. WordNet stores proper nouns capitalised ('Arthur'), so names don't count as words."""
    lw = w.lower()
    if lw in CLOSED_CLASS:
        return True
    wn = _wn()
    bases = {lw, *(b for pos in "nvar" if (b := wn.morphy(lw, pos)))}
    return any(n == b for b in bases for syn in wn.synsets(b) for n in syn.lemma_names())


def spot_names(text: str) -> list[str]:
    """'free' condition, in text order. A capitalised word counts as a name if (a) it does not
    start a sentence and isn't 'I' (rule fixed with the PI 2026-10-25), or (b) it starts a sentence,
    is in the SSA name list, and is not a common English word (fix 2026-10-25: Gemma opens the
    paragraph with the name, which (a) alone missed). (a) also catches other proper nouns
    ('Earth'); (b) misses names that are also words ('Rose') unless they recur mid-sentence."""
    hits = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        words = re.findall(r"[\"'\u201c\u2018(]*[A-Za-z][A-Za-z'\u2019\-]*", sent)
        for k, w in enumerate(words):
            w = re.sub(r"['\u2019]s?$", "", w.lstrip("\"'\u201c\u2018("))  # possessive = a use
            if w == "I" or not CAP_WORD.fullmatch(w):
                continue
            if k > 0 or (w in ssa_table() and not is_common_word(w)):
                hits.append(w)
    return hits


def code_v2(
    answer: str, noun: str, finish_reason: str, think: str | None = None, name_pos: str = "after"
) -> dict:
    """name_pos: 'after' / 'before' (Name line required there), 'none' (told not to name),
    'free' (no name instruction; names spotted in the text), 'plan' (= after + "decide on a name
    first"), 'free_noted' (Name line last, but the name MAY appear in the paragraph: not gated)."""
    if name_pos in ("none", "free"):
        para, name = (answer or "").strip(), None
    else:
        para, name = split_name(answer, "before" if name_pos == "before" else "after")
    spotted = spot_names(para) if name_pos == "free" else []
    if spotted:
        name = spotted[0]
    hit = re.search(rf"\b({re.escape(noun)}|{re.escape(plural(noun))})\b", para, re.IGNORECASE)
    gates = {
        "noun_hit": bool(hit),
        "name_ok": name is not None or name_pos in ("none", "free"),
        "truncated": finish_reason == "length",
        "error": finish_reason == "error",
        # 2026-10-26: name-only answers ("Name: Everlyn", no paragraph) were passing every gate
        "no_paragraph": not re.search(r"[A-Za-z]{2,}", para),
    }
    nm = code_name(name)
    parts = [nm["given"]] + re.findall(r"[A-Za-z][A-Za-z'\-]*", name or "")
    parts = [p for p in parts if p and p.lower().rstrip(".") not in TITLES]
    gates["name_in_para"] = name_pos != "free" and any(
        re.search(rf"\b{re.escape(p)}\b", para) for p in parts
    )
    if name_pos == "free_noted":  # outcome, not a gate
        n_ment = len(re.findall(rf"\b{re.escape(nm['given'])}\b", para)) if nm["given"] else 0
        sp = spot_names(para)
        nm |= {"name_used_in_para": gates.pop("name_in_para"), "n_name_mentions": n_ment,
               "para_first_name": sp[0] if sp else "",
               "para_name_matches": bool(sp) and sp[0] == nm["given"]}  # fmt: skip
        gates["name_in_para"] = False
    gates["valid"] = gates["name_ok"] and not (
        gates["noun_hit"]
        or gates["name_in_para"]
        or gates["truncated"]
        or gates["error"]
        or gates["no_paragraph"]
    )
    n_par = len([b for b in re.split(r"\n\s*\n", para) if b.strip()])
    if name_pos == "free":
        nm |= {"named": bool(spotted), "spotted": "|".join(dict.fromkeys(spotted)),
               "n_name_mentions": spotted.count(spotted[0]) if spotted else 0}  # fmt: skip
    nm["noun_in_name"] = bool(name and re.search(rf"\b{re.escape(noun)}", name, re.IGNORECASE))
    return gates | {"n_paragraphs": n_par} | nm | code(para, think)


def split_think(raw: str) -> tuple[str | None, str, bool]:
    """Return (think, answer, think_closed). Qwen3 writes <think>...</think> then the answer;
    Gemma 4 writes <|channel>thought\n...<channel|> then the answer."""
    if "<|channel|>" in raw:  # gpt-oss harmony: analysis channel, then final channel
        raw = raw.replace("<|return|>", "").replace("<|end|>", "\n")
        ana, sep, fin = raw.partition("<|channel|>final<|message|>")
        th = ana.split("<|message|>", 1)[-1] if "analysis" in ana else ""
        th = th.split("<|start|>")[0].strip()
        if not sep:
            return (th or None), "", False
        return (th or None), fin.strip(), True
    if "<|channel>" in raw or "<channel|>" in raw:
        if "<channel|>" in raw:
            th, ans = raw.split("<channel|>", 1)
            th = th.replace("<|channel>", "").removeprefix("thought").strip()
            return (th or None), ans.strip(), True
        return raw.replace("<|channel>", "").removeprefix("thought").strip(), "", False
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
            # reasoning text: thinking runs, or summaries a closed model returns unasked (GPT)
            th = (r.get("think") or None) if (r["thinking"] or r.get("reasoning")) else None
            pos = r.get("name_condition") or ("after" if "Name:" in r["prompt"] else None)
            if pos:
                c = code_v2(r["answer"], r["noun"], r["finish_reason"], th, pos)
            else:
                c = code(r["answer"], th)
            keep = ("model", "name_condition", "noun", "modifier", "thinking", "sample",
                    "finish_reason", "provider")  # fmt: skip
            rows.append({k: r.get(k) for k in keep} | c)
    d = pd.DataFrame(rows)
    # resumed runs append: per completion keep the last non-error row (errors only if no retry won)
    kc = [
        k for k in ("model", "name_condition", "noun", "modifier", "thinking", "sample") if k in d
    ]
    d["_err"] = d["finish_reason"] == "error"
    d = d.iloc[d.reset_index().sort_values(["_err", "index"], ascending=[False, True]).index]
    d = d.drop_duplicates(kc, keep="last").sort_index().drop(columns="_err")
    coded = args.out.with_name(args.out.stem + "_coded.csv")
    d.to_csv(coded, index=False)
    if d["name_condition"].isna().all():
        d = d.drop(columns="name_condition")
    keys = [k for k in ("model", "name_condition", "thinking", "noun", "modifier") if k in d]
    s = d.groupby(keys).size().rename("n").to_frame()
    if "valid" in d:
        g0 = d.groupby(keys)
        for c in (
            "valid",
            "noun_hit",
            "name_ok",
            "name_in_para",
            "truncated",
            "error",
            "no_paragraph",
        ):
            s[c] = g0[c].mean()
        s["mean_paragraphs"] = g0["n_paragraphs"].mean()
        d = d[d["valid"]]  # outcome shares below are among valid outputs only
        s["n_valid"] = d.groupby(keys).size()
    g = d.groupby(keys)
    for lab in ("she", "he", "they", "neo", "mixed", "it", "none"):
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
