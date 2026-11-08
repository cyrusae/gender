"""Gender ValNorm, English pilot: does an embedding's word-level gender lean track Glasgow GEND?

Validity check against published numbers, not a new result: Lewis & Lupyan (2020, Study 1a) report
r = 0.59 (fastText Wikipedia) and 0.63 (subtitles); Valentini et al. (2023, Table 1) r = 0.44-0.58.
Following them: attribute words A = female, woman, girl, sister, she, her, hers, daughter;
B = male, man, boy, brother, he, him, his, son (Valentini et al. 2023 fn. 2); Glasgow homonym senses
("word (sense)") averaged; attribute words excluded from the test words.

Scores per word: (1) mean cos(w, B) − mean cos(w, A) (Lewis & Lupyan's mean difference, oriented
male-positive to match GEND 7 = masculine); (2) SC-WEAT effect size (ValNorm's score): that
difference divided by the standard deviation of cos(w, x) over x in A ∪ B.

    uv run --no-project --python 3.12 --with numpy,pandas,scipy python embeddings/scripts/gvalnorm_en.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
GLASGOW = ROOT / "data/raw/glasgow/glasgow_norms_scott2019.csv"
EMB = {
    "muse_wiki_en": Path("/Volumes/Lexar/gbleed/shared/embeddings/muse/wiki.multi.en.vec"),
    "fasttext_cc_en": Path("/Volumes/Lexar/gbleed/shared/embeddings/fasttext_cc/cc.en.300.vec"),
}
OUT = ROOT / "embeddings/results/gvalnorm_en"
A = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]
B = ["male", "man", "boy", "brother", "he", "him", "his", "son"]


def glasgow():
    g = pd.read_csv(GLASGOW, header=[0, 1])
    words = g.iloc[:, 0].astype(str)
    gend = g[("GEND", "M")] if ("GEND", "M") in g.columns else g.iloc[:, g.columns.get_level_values(0).tolist().index("GEND")]
    df = pd.DataFrame({"word": words.map(lambda w: re.sub(r"\s*\(.*\)$", "", w).strip()), "gend": gend.astype(float)})
    return df.groupby("word", as_index=False)["gend"].mean()


def load(path, want, nmax=2_000_000):
    vecs = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= nmax:
                break
            w, rest = line.split(" ", 1)
            if w in want and w not in vecs:
                vecs[w] = np.fromstring(rest, sep=" ")
    return vecs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gl = glasgow()
    gl = gl[~gl["word"].isin(A + B)]
    want = set(gl["word"]) | set(A) | set(B)
    summary = {}
    for name, path in EMB.items():
        V = load(path, want)
        assert all(w in V for w in A + B), [w for w in A + B if w not in V]
        Am = np.array([V[w] / np.linalg.norm(V[w]) for w in A])
        Bm = np.array([V[w] / np.linalg.norm(V[w]) for w in B])
        d = gl[gl["word"].isin(V)].copy()
        W = np.array([V[w] / np.linalg.norm(V[w]) for w in d["word"]])
        ca, cb = W @ Am.T, W @ Bm.T
        d["mean_diff"] = cb.mean(1) - ca.mean(1)
        d["sc_weat"] = d["mean_diff"] / np.concatenate([ca, cb], 1).std(1, ddof=1)
        d.to_csv(OUT / f"{name}_per_word.csv", index=False)
        summary[name] = {"n_words": len(d), "n_glasgow": len(gl)}
        for s in ("mean_diff", "sc_weat"):
            r, p = pearsonr(d[s], d["gend"])
            summary[name][s] = {"pearson_r": r, "p": p, "spearman_rho": spearmanr(d[s], d["gend"])[0]}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
