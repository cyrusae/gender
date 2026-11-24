"""Exploratory (2026-11-23, PI): how much can Zhou et al.'s utility tests see of what Shift changes?

1. Translation: how many MUSE test queries (`{src}-{tgt}.5000-6500.txt`, as zhou_sec43_44.py) have a shifted
   occupation word (either form of Zhou et al.'s 58 ES / 23 FR pairs) as source or among the gold targets.
2. Spanish word similarity (SemEval-2017, as word_similarity_es.py): Spearman and Pearson of the *original*
   space as the vocabulary size changes, to test whether Table 1's Original (0.7392) vs Shift Ori (0.7363)
   difference is a vocabulary-size artefact like the translation one (Z18): coverage is the only thing that
   changes with vocabulary size here.

    uv run --no-project --python 3.12 --with numpy,pandas,scipy,scikit-learn python embeddings/scripts/utility_coverage.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from word_similarity_es import SEM, item
from zhou_sec43_44 import DD
from zhou_shift import Z, load_vec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/utility_coverage"
SIZES = (50_000, 75_000, 100_000, 125_000, 150_000, 175_000, 200_000)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for lang in ("es", "fr"):
        occ = set()
        for line in open(Z / f"{lang.upper()}_occupation_words_with_EN_translations"):
            f, m, _ = line.rstrip("\n").split("\t")
            occ |= {f, m}
        for d in (f"en-{lang}", f"{lang}-en"):
            q = {}
            for a, b in (x.split() for x in open(DD / f"{d}.5000-6500.txt", encoding="utf-8")):
                q.setdefault(a, set()).add(b)
            hit = sorted(a for a, bs in q.items() if a in occ or bs & occ)
            rows.append({"test": "translation", "set": d, "n": len(q), "n_touching_shifted": len(hit), "words": " ".join(hit)})
    pairs = [line.rstrip("\n").split("\t") for line in open(SEM / "data/es.test.data.txt", encoding="utf-8")]
    gold = [float(x) for x in open(SEM / "keys/es.test.gold.txt")]
    E, w2i = load_vec(Z / "data/wiki.multi.es.vec.txt", nmax=max(SIZES))
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    sim_rows = []
    for n in SIZES:
        voc = {w: i for w, i in w2i.items() if i < n}
        keep = [(item(a, voc), item(b, voc), g) for (a, b), g in zip(pairs, gold)]
        keep = [k for k in keep if k[0] and k[1]]
        s = [En[voc[a]] @ En[voc[b]] for a, b, _ in keep]
        g = [k[2] for k in keep]
        sim_rows.append({"vocab": n, "n_pairs": len(keep), "spearman": spearmanr(s, g)[0], "pearson": pearsonr(s, g)[0]})
    T, S = pd.DataFrame(rows), pd.DataFrame(sim_rows)
    T.to_csv(OUT / "translation_queries.csv", index=False)
    S.to_csv(OUT / "similarity_by_vocab.csv", index=False)
    print(T[["set", "n", "n_touching_shifted"]].to_string(index=False))
    print(S.round(4).to_string(index=False))
    print("Zhou Table 1 word similarity: Original 0.7392, Shift Ori 0.7363, Hybrid Ori 0.7358")


if __name__ == "__main__":
    main()
