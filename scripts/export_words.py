"""Word-level Phase 5 tables for the PI's exploration (descriptive; not the CC0 `eda/` set).

For every steering run (model × layer × noun/all positions) and the graded English readouts
R1 ("…es muy ___" read in English) and R1-EN (all-English prompts), at the working dose (+1 × α*),
over the primary wordings W1–W3 that passed the frame check, with the Gemma known-check filter:

  results/word_level/noun_steering.csv
      one row per run × readout × noun: the noun's graded score (feminine-rated adjectives up,
      masculine down; the same −β_GEND as the verdicts) for the real gender direction, the mean
      and 95th percentile of the random directions, the social and number directions, plus the
      noun's language, gender, English concept and set.
  results/word_level/adjective_shifts.csv
      one row per run × readout × adjective: mean log-probability shift (nats, vs unsteered) over
      nouns and wordings, for the real direction, the random mean, social and number, plus the
      adjective's Glasgow ratings (GEND 1 = feminine … 7 = masculine, VAL, AROU, SIZE) and Zipf.
  results/word_level/adjectives_{qwen3,gemma3}.csv
      the readout vocabularies themselves (copied from data/stimuli).

These are descriptions of what moved; the verdicts remain the pre-registered ones (runs.json /
verdicts.csv). Per-noun differences are noisy: one noun's score averages 3 prompts.

    uv run python scripts/export_words.py
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from gbleed.phase5_analysis import (
    PRIMARY_WORDINGS,
    failed_wordings,
    graded,
    load,
    ratings_for,
    shifts,
)

G = "worktrees/gender-gemma/results"
Q = "worktrees/gender-s2/results"
MODELS = {  # model: (sweep root, readout-check root, family)
    "Qwen/Qwen3-8B-Base": ("/Volumes/Lexar/gbleed/gender-s2/results/phase5",
                           f"{Q}/phase5_readoutcheck", "qwen3"),
    "Qwen/Qwen3-14B-Base": ("/Volumes/Lexar/gbleed/gender-s2/results/phase5",
                            f"{Q}/phase5_readoutcheck", "qwen3"),
    "google/gemma-3-4b-pt": (f"{G}/phase5", f"{G}/phase5_readoutcheck", "gemma3"),
    "google/gemma-3-12b-pt": (f"{G}/phase5", f"{G}/phase5_readoutcheck", "gemma3"),
}  # fmt: skip
READOUTS = ("R1", "R1-EN")
OUT = Path("results/word_level")


def run_tables(model_id, run_dir: Path, fam: str, failed: set):
    info = json.loads((run_dir / "vectors.json").read_text())
    tag = {"model": model_id.split("/")[1], "lang": info["lang"], "layer": info["layer"],
           "positions": info["positions"]}  # fmt: skip
    nouns, adjs = [], []
    for name in READOUTS:
        keys, lp, read = load(run_dir, name)
        keep = keys.mult.isin([0.0, 1.0]).to_numpy()
        drop = {w for lg, rd, w in failed if lg == info["lang"] and rd == name}
        keep = keep & keys.wording.isin(PRIMARY_WORDINGS[name] - drop).to_numpy()
        keys, lp = keys[keep].reset_index(drop=True), lp[keep]
        k, sh = shifts(keys, lp)
        rat = ratings_for(name, fam, read)
        sh = sh[:, : len(rat)]
        k["score"], _ = graded(sh, rat)
        w = k[k.mult == 1.0]
        piv = w.groupby(["lemma", "vec"]).score.mean().unstack("vec")
        rand = [c for c in piv.columns if c.startswith("rand")]
        n = pd.DataFrame({"real": piv["v"], "random_mean": piv[rand].mean(1),
                          "random_p95": piv[rand].quantile(0.95, axis=1),
                          "social": piv.get("social"), "number": piv.get("number")})  # fmt: skip
        n["real_minus_random"] = n.real - n.random_mean
        nouns.append(n.reset_index().assign(readout=name, **tag))
        m1 = (k.mult == 1.0).to_numpy()
        cols = {}
        for lab, sel in (("real", k.vec == "v"), ("random_mean", k.vec.str.startswith("rand")),
                         ("social", k.vec == "social"), ("number", k.vec == "number")):  # fmt: skip
            s = m1 & sel.to_numpy()
            cols[f"shift_{lab}"] = sh[s].mean(0) if s.any() else np.nan
        a = rat[["word", "GEND", "VAL", "AROU", "SIZE", "zipf"]].copy()
        for c, v in cols.items():
            a[c] = v
        adjs.append(a.assign(readout=name, **tag))
    return pd.concat(nouns), pd.concat(adjs)


def main() -> None:
    from gbleed import known_filter as kf

    # Gemma known-check results live in the Gemma worktree
    kf.ROOTS = {"es": f"{G}/phase2_known", "de": f"{G}/phase3_known"}
    kf.PAIR_ROOTS = {"es": f"{G}/phase4_known_es", "de": f"{G}/phase4_known_de"}
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv("data/stimuli/phase5_nouns_v1.csv", keep_default_na=False)
    meta = meta.drop_duplicates(["lang", "lemma"])[["lang", "lemma", "gender", "concept_en", "set"]]
    N, A = [], []
    for m, (root, rc, fam) in MODELS.items():
        failed = failed_wordings(m, rc)
        d = Path(root) / m.replace("/", "__")
        for run in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("._")):
            n, a = run_tables(m, run, fam, failed)
            N.append(n)
            A.append(a)
            print(m, run.name, len(n), flush=True)
    nouns = pd.concat(N)
    # R1-EN prompts use the noun's English concept in place of the noun: its "lemma" is the
    # concept; attach the source noun(s) by concept (a concept shared by two nouns joins both)
    en = nouns.readout == "R1-EN"
    src = meta.rename(columns={"lemma": "source_lemma"})
    a = nouns[~en].merge(meta, on=["lang", "lemma"], how="left")
    a["source_lemma"] = a.lemma
    b = nouns[en].merge(src, left_on=["lang", "lemma"], right_on=["lang", "concept_en"], how="left")
    nouns = pd.concat([a, b], ignore_index=True)
    first = ["model", "layer", "positions", "readout", "lang", "lemma", "source_lemma", "gender",
             "concept_en", "set"]  # fmt: skip
    nouns[first + [c for c in nouns.columns if c not in first]].to_csv(
        OUT / "noun_steering.csv", index=False
    )
    adj = pd.concat(A)
    first = ["model", "layer", "positions", "readout", "word"]
    adj[first + [c for c in adj.columns if c not in first]].to_csv(
        OUT / "adjective_shifts.csv", index=False
    )
    for fam in ("qwen3", "gemma3"):
        shutil.copy(f"data/stimuli/phase5_adjectives_{fam}_v1.csv", OUT / f"adjectives_{fam}.csv")


if __name__ == "__main__":
    main()
    print("=== EXPORT_WORDS done", flush=True)
