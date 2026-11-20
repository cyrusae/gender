"""Candidate expanded English attribute pairs (docs/english-default-prereg.md, addendum A, DRAFT).
Built by rule for PI review; nothing scored here.

Sources: Bolukbasi et al.'s equalize pairs (52) and definitional pairs (10), and Caliskan et al.'s sixteen
attribute words (as 8 pairs). Lower-cased and de-duplicated; multiword entries (`twin_brother`) dropped.
Orientation (male, female): by Glasgow GEND where both are rated (higher = male), otherwise as listed
with the source's order checked against a fixed list of reversed source pairs (wives/husbands,
ex_girlfriend/ex_boyfriend, and the definitional file, which lists female first).
Rule filters: no proper names (John/Mary); both words in the first 200k MUSE English words; not an English zero-point anchor or the
gloss of a noun tested in Q2/Q3; if Glasgow rates a word, it must be clearly gendered (male ≥ 5.5,
female ≤ 2.5). Claude's suggestions for non-person or mismatched pairs are marked
`claude_suggest_unverified` for the PI.
Contrast list (not attributes): the 40 most masculine and 40 most feminine Glasgow words in the 200k
vocabulary, regardless of meaning (stereotype-laden on purpose).

    uv run --no-project --python 3.12 --with pandas python embeddings/scripts/en_attr_build.py
"""

import json
from pathlib import Path

import pandas as pd
from wp_cues_build import glasgow, vocab

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "external/debiaswe/data"
NAMES = {"john", "mary"}
REVERSED = {("wives", "husbands"), ("ex_girlfriend", "ex_boyfriend")}
CALISKAN = list(zip(["male", "he", "him", "his", "man", "boy", "son", "brother"],
                    ["female", "she", "her", "hers", "woman", "girl", "daughter", "sister"]))  # fmt: skip
SUGGEST = {("monastery", "convent"): "drop: places, not people", ("prostate_cancer", "ovarian_cancer"): "drop: diseases",
           ("testosterone", "estrogen"): "drop: hormones", ("colt", "filly"): "drop: animals",
           ("gelding", "mare"): "drop: animals", ("fella", "granny"): "drop: not a pair (age differs)",
           ("catholic_priest", "nun"): "drop: multiword", ("dudes", "gals"): "check: informal",
           ("him", "her"): "check: her is also the possessive (his/her)",
           ("his", "hers"): "check: his appears twice (with her and hers)"}  # fmt: skip


def main():
    en, gl = vocab("en", 200_000), glasgow()
    anc = pd.read_csv(ROOT / "data/en_zero_frozen_v1.csv")
    tested = set()
    for lang in ("es", "fr"):
        tested |= set(pd.read_csv(ROOT / f"data/wp_persons_{lang}_frozen_v1.csv", keep_default_na=False).en)
    blocked = set(anc.word) | tested
    raw = [(a, b, "equalize") for a, b in json.load(open(D / "equalize_pairs.json"))]
    raw += [(m, f, "definitional") for f, m in json.load(open(D / "definitional_pairs.json"))]
    raw += [(m, f, "caliskan") for m, f in CALISKAN]
    rows, seen = [], set()
    for a, b, src in raw:
        a, b = a.lower(), b.lower()
        m, f = (b, a) if (a, b) in REVERSED else (a, b)
        if gl.get(m) is not None and gl.get(f) is not None and gl[m] < gl[f]:
            m, f = f, m
        if (m, f) in seen:
            continue
        seen.add((m, f))
        why = ("multiword" if "_" in m + f else "proper name" if {m, f} & NAMES else "not in 200k" if m not in en or f not in en
               else "anchor or tested gloss" if {m, f} & blocked
               else "not clearly gendered (Glasgow)" if (gl.get(m) is not None and gl[m] < 5.5) or (gl.get(f) is not None and gl[f] > 2.5)
               else "")  # fmt: skip
        sug = SUGGEST.get((a, b)) or SUGGEST.get((m, f)) or SUGGEST.get((f, m)) or ""
        rows.append({"male": m, "female": f, "source": src, "glasgow_m": gl.get(m, ""), "glasgow_f": gl.get(f, ""),
                     "rank_m": en.get(m, ""), "rank_f": en.get(f, ""), "rule_drop": why,
                     "claude_suggest_unverified": sug, "pi_keep": ""})  # fmt: skip
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "data/en_attr_candidates_v0.csv", index=False)
    k = d[d.rule_drop == ""]
    print("pairs", len(d), "| kept by rule", len(k), d.rule_drop.value_counts().to_dict())
    print("kept:", "; ".join(f"{m}/{f}" + (" *" if s else "") for m, f, s in zip(k.male, k.female, k.claude_suggest_unverified)))
    rated = sorted(((w, g) for w, g in gl.items() if w in en), key=lambda x: x[1])
    con = pd.DataFrame([{"word": w, "glasgow": g, "side": "female"} for w, g in rated[:40]]
                       + [{"word": w, "glasgow": g, "side": "male"} for w, g in rated[-40:]])  # fmt: skip
    con.to_csv(ROOT / "data/en_attr_contrast_glasgow_v0.csv", index=False)
    print("contrast female:", " ".join(con[con.side == "female"].word))
    print("contrast male:", " ".join(con[con.side == "male"].word))


if __name__ == "__main__":
    main()
