"""Exploratory (2026-11-14, PI): Spanish word similarity, Table 1's last row (upstream Z25: no released
code). SemEval-2017 Task 2, subtask 1, Spanish test set (500 pairs; Camacho-Collados et al. 2017; zip
sha256 2e7381b7…, stored at $EMB_STORE/semeval2017_task2/ or the monorepo store). Script committed
before running.

Score = cosine of the two words' vectors (renormalised, since shifted vectors aren't unit length).
Pairs where either item is missing from the vocabulary are dropped (coverage reported); multi-word
items are tried lowercased with spaces as underscores, then dropped. Reported: Pearson (the paper's
statistic) and Spearman, for the first 50k words (Zhou's notebook vocabulary; primary) and 200k.
Spaces: original; Shift Ori as implemented (occupation pairs only; our rebuild); Hybrid Ori isn't
computed because an orthogonal alignment leaves every Spanish cosine unchanged (identical to Shift
Ori; Z11). Zhou's Table 1: Original 0.7392, Shift Ori 0.7363, Hybrid Ori 0.7358.

    uv run --no-project --python 3.12 --with numpy,pandas,scipy,scikit-learn python embeddings/scripts/word_similarity_es.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from c1_run import directions
from scipy.stats import pearsonr, spearmanr
from zhou_shift import Z, load_vec, shift

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/word_similarity_es"
SEM = Path(os.environ.get("SEMEVAL_DIR", "/Volumes/Lexar/gbleed/shared/semeval2017_task2")) / "SemEval17-Task2/test/subtask1-monolingual"


def item(w, w2i):
    for c in (w, w.lower(), w.lower().replace(" ", "_")):
        if c in w2i:
            return c
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [line.rstrip("\n").split("\t") for line in open(SEM / "data/es.test.data.txt", encoding="utf-8")]
    gold = [float(x) for x in open(SEM / "keys/es.test.gold.txt")]
    E, w2i = load_vec(Z / "data/wiki.multi.es.vec.txt", nmax=200_000)
    w50 = {w: i for w, i in w2i.items() if i < 50_000}
    occ = [line.rstrip("\n").split("\t") for line in open(Z / "ES_occupation_words_with_EN_translations")]
    occp = [(m, f) for f, m, _ in occ if m in w50 and f in w50]
    ds, _ = directions(E[:50_000], w50, "es")
    spaces = {"original": E, "shift_ori (as implemented)": shift(E, w2i, occp, ds)}
    shifted = {w for p in occp for w in p}
    rows = []
    for vname, voc in (("50k (primary)", w50), ("200k", w2i)):
        keep = [(item(a, voc), item(b, voc), g) for (a, b), g in zip(pairs, gold)]
        keep = [k for k in keep if k[0] and k[1]]
        touched = sum(1 for a, b, _ in keep if a in shifted or b in shifted)
        for sname, X in spaces.items():
            Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
            sim = [Xn[voc[a]] @ Xn[voc[b]] for a, b, _ in keep]
            g = [k[2] for k in keep]
            rows.append({"vocab": vname, "space": sname, "n_pairs": len(keep), "of": len(pairs),
                         "pairs_with_shifted_word": touched, "pearson": pearsonr(sim, g)[0],
                         "spearman": spearmanr(sim, g)[0]})  # fmt: skip
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "results.csv", index=False)
    print(d.round(4).to_string(index=False))
    print("Zhou Table 1: Original 0.7392, Shift Ori 0.7363, Hybrid Ori 0.7358, De-Align 0.7392")


if __name__ == "__main__":
    main()
