"""Freeze the person-noun list (docs/wordpair-persons-prereg.md). Reads the PI's review
(data/wp_persons_es_candidates_v0_edited.csv, from the .numbers file; blank pi_keep = keep, PI
2026-11-11), Claude's corrected suggestions (wp_persons_es_suggest_fix_v0.csv) for the group, and the
flags (wp_persons_es_flags_v0.csv). Writes data/wp_persons_es_frozen_v1.csv.

Drops on top of the PI's: name homographs; identity nouns with a place-name form (PI); germano
(historical Germanic peoples, not modern Germans). Group overrides from the PI's comments: arqueólogo,
bioquímico → occupation; defendido, humano (PI kept) → person; empírico, judío → identity. Gloss fix: extraño →
stranger (PI). All other flags are kept as columns for the sensitivity runs.

    uv run --no-project --python 3.12 --with pandas python embeddings/scripts/wp_persons_freeze.py
"""

import hashlib
from pathlib import Path

import pandas as pd
from wp_cues_build import vocab

ROOT = Path(__file__).resolve().parents[1]
GROUP_FIX = {"arqueólogo": "occupation", "bioquímico": "occupation", "defendido": "person",
             "empírico": "identity", "judío": "identity", "humano": "person"}  # fmt: skip
EXTRA_DROP = {"germano": "historical Germanic peoples, not modern Germans"}


def main():
    D = ROOT / "data"
    pi = pd.read_csv(D / "wp_persons_es_candidates_v0_edited.csv", keep_default_na=False, dtype=str).iloc[:, :18]
    fix = pd.read_csv(D / "wp_persons_es_suggest_fix_v0.csv", keep_default_na=False).set_index("masc").corrected_suggestion
    fl = pd.read_csv(D / "wp_persons_es_flags_v0.csv", keep_default_na=False, dtype=str).drop(columns="en")
    d = pi.merge(fl, on="masc", how="left", validate="one_to_one")
    sug = d.masc.map(fix).fillna(d.claude_suggest_unverified)
    d["group_final"] = d.masc.map(GROUP_FIX).fillna(sug.where(~sug.str.startswith(("drop", "check")), ""))
    d.loc[d.masc == "extraño", "en"] = "stranger"
    keep = d.pi_keep.isin(["True", "", "None"])
    keep &= d.name_homograph != "True"
    keep &= ~((d.group_final == "identity") & (d.place_name == "True"))
    keep &= ~d.masc.isin(EXTRA_DROP)
    keep &= d.group_final != ""
    F = d[keep].copy()
    en = vocab("en", 200_000)
    assert F.en.isin(en).all(), F[~F.en.isin(en)].en.tolist()
    cols = ["masc", "fem", "en", "group_final", "rank_m", "rank_f", "glasgow_gend", "fem_also_noun", "masc_also_noun",
            "language_name", "masc_identical_en", "place_name"]
    F = F[cols].rename(columns={"group_final": "group"}).sort_values(["group", "masc"])
    F.to_csv(D / "wp_persons_es_frozen_v1.csv", index=False)
    print(F.groupby("group").size().to_dict())
    for g in ("person", "occupation"):
        f = F[F.group == g]
        print(g, {c: int((f[c] == "True").sum()) for c in ("fem_also_noun", "masc_also_noun", "masc_identical_en", "place_name")})
    print(hashlib.sha256((D / "wp_persons_es_frozen_v1.csv").read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
