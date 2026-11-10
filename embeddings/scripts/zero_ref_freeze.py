"""Freeze the zero-point reference lists (docs/zhou-zero-point-prereg.md, adopted 2026-11-10).

Inputs: the PI-reviewed candidate files (es v1, fr v1) and the PI additions (es). Output:
data/zero_ref_{lang}_frozen_v1.csv with one row per reference word:
  set (epicene | common | collective | inanimate), tier (strict | broad), lemma, gender,
  flags for sensitivity checks: s1_identical_to_gloss, s2_en_more_frequent, s3_pi_loan_flag,
  s5_in_zhou_lists, s6_pi_borderline.
Rules (as reviewed): epicene/common rows kept unless pi_keep is False (ES: kept only if True);
additions with pi_status add (strict) / add_broad (broad); inanimate rows dropped if pi_keep is
False or if the PI moved them to a person set (eminencia) or flagged them as persons/groups
(voyer, philharmonie, already pi_keep False).

    uv run --no-project --python 3.12 --with pandas python embeddings/scripts/zero_ref_freeze.py
"""

import hashlib
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "embeddings/data"
LOAN = re.compile(r"english|french|loanword|language|reinstated", re.IGNORECASE)
MOVED_OUT_OF_INANIMATE = {"es": {"eminencia"}, "fr": set()}


def freeze(lang):
    d = pd.read_csv(DATA / f"zero_ref_candidates_{lang}_v1.csv")
    d["pi_comment"] = d.get("pi_comment", pd.Series(index=d.index, dtype=object))
    keep = d["pi_keep"].astype(str).str.lower()
    if lang == "es":
        person_ok = keep.isin(["true", "y", "1"])
    else:
        person_ok = ~keep.isin(["false", "n", "0"])
    persons = d[d["set"].isin(["epicene", "common"]) & person_ok].assign(tier="strict")
    ina = d[(d["set"] == "inanimate") & ~keep.isin(["false", "n", "0"])
            & ~d["lemma"].isin(MOVED_OUT_OF_INANIMATE[lang])].assign(tier="strict")  # fmt: skip
    rows = [persons, ina]
    add_path = DATA / f"zero_ref_additions_{lang}.csv"
    if add_path.exists():
        a = pd.read_csv(add_path)
        a = a[a["pi_status"].isin(["add", "add_broad"])].copy()
        a["tier"] = a["pi_status"].map({"add": "strict", "add_broad": "broad"})
        a["pi_comment"] = "PI addition: " + a["claude_note_unverified"].astype(str)
        rows.append(a[["set", "lemma", "gender", "tier", "pi_comment"]])
    f = pd.concat(rows, ignore_index=True)
    assert not f["lemma"].duplicated().any(), f[f["lemma"].duplicated()]["lemma"].tolist()
    lx = pd.read_csv(ROOT / f"data/lexicon/{lang}_nouns.csv", low_memory=False)
    lx = lx.drop_duplicates("lemma").set_index("lemma")
    f["concept_en"] = f["lemma"].map(lx["concept_en"])
    f["zipf"] = f["lemma"].map(lx["zipf"])
    f["zipf_en"] = f["lemma"].map(lx["zipf_en"])
    f["s1_identical_to_gloss"] = f["lemma"].str.lower() == f["concept_en"].astype(str).str.lower()
    f["s2_en_more_frequent"] = f["zipf_en"] > f["zipf"]
    note = f["pi_comment"].fillna("").astype(str) + " " + f.get("claude_crosscheck_unverified", "").fillna("").astype(str)
    f["s3_pi_loan_flag"] = note.str.contains(LOAN) & ~note.str.startswith("PI addition")
    f["s5_in_zhou_lists"] = f["in_zhou_lists"].fillna(False).astype(bool)
    f["s6_pi_borderline"] = f["pi_comment"].fillna("").astype(str).str.contains(r"\?")
    cols = ["set", "tier", "lemma", "gender", "concept_en", "zipf", "zipf_en", "s1_identical_to_gloss",
            "s2_en_more_frequent", "s3_pi_loan_flag", "s5_in_zhou_lists", "s6_pi_borderline", "pi_comment"]  # fmt: skip
    path = DATA / f"zero_ref_{lang}_frozen_v1.csv"
    f[cols].to_csv(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(lang, f.groupby(["set", "tier", "gender"]).size().to_dict(), "sha256", sha[:16])
    return sha


if __name__ == "__main__":
    for lang in ("es", "fr"):
        freeze(lang)
