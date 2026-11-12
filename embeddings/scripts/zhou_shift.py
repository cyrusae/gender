"""Rebuild Zhou et al. (2019) Shift Ori for Spanish and check it against their Table 1.

Their repo ships no mitigated embeddings, so this re-implements the mitigation from §3.1–3.2 and
the notebook cells it can be read from (bias_emnlp19.ipynb: doPCA cell 36, LinearSVC/LDA grammatical
direction cells 73–75, get_SG_component cell 78, offsets cells 273–275, modified_WEAT cell 222,
_weat_pval cell 230). Notebook quirks kept deliberately (they define the published numbers):
  - doPCA's centre is `a + b/2` (operator precedence), not the pair midpoint;
  - MWEAT uses raw dot products on the (unit-norm) MUSE vectors; test statistic is
    | |sum male-form assoc| - |sum female-form assoc| |;
  - p-value: 1000 random re-partitions of the occupation words, random.seed(520); the share of
    permutations tying the observed statistic is reported too (upstream Z10).
Usage: `... zhou_shift.py [es|fr]` (French attributes = the French definitional pairs, as cell 215).
Table 1 (ES): Original 3.6918 (p 0.000); Shift Ori 0.3090 (p 0.113); Hybrid Ori 0.3090 (p 0.733).

    uv run --no-project --python 3.12 --with numpy,scikit-learn python embeddings/scripts/zhou_shift.py
"""

import io
import json
import random
from itertools import filterfalse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC

Z = Path(__file__).resolve().parents[1] / "external/Bias_in_Gendered_Languages"
OUT = Path(__file__).resolve().parents[1] / "results/zhou_shift_es"
WEAT_M = ["hombre", "niño", "padre", "masculino", "abuelo", "esposo", "hijo", "tio"]
WEAT_F = ["niña", "femenina", "tía", "hija", "esposa", "mujer", "madre", "abuela"]
# Upstream Z19: Zhou's lists have *tio* (misspelling) and French duplicates + a name pair. Corrected
# lists are PRIMARY since 2026-11-11 (deviation notes in each prereg) and the default (PI). Set
# ZHOU_ATTR=published to reproduce Zhou's lists and the frozen result files committed before then
# (results/c1_*, gvalnorm_*, zhou_zero_point_*, crime_probe_*, zhou_construct*, zhou_sec43_44); the
# corrected outputs are in results/attrfix_sensitivity/. Rerunning with the default overwrites those
# frozen files with corrected values.
import os as _os  # noqa: E402

CORRECTED = _os.environ.get("ZHOU_ATTR", "corrected") != "published"
if CORRECTED:
    WEAT_M = ["hombre", "niño", "padre", "masculino", "abuelo", "esposo", "hijo", "tío"]


def attributes(lang, defs):
    """(male, female) attribute words: Spanish WEAT lists, French = definitional pairs (cell 215)."""
    if lang == "es":
        return WEAT_M, WEAT_F
    if CORRECTED:
        return (["homme", "mari", "garçon", "fils", "il", "père", "mâle", "frère", "oncle"],
                ["femme", "fille", "elle", "mère", "femelle", "sœur", "tante"])
    return [b for _, b in defs], [a for a, _ in defs]


def load_vec(path, nmax=50000):
    vecs, w2i = [], {}
    with io.open(path, encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            w, v = line.rstrip().split(" ", 1)
            vecs.append(np.fromstring(v, sep=" "))
            w2i[w] = len(w2i)
            if len(w2i) == nmax:
                break
    return np.vstack(vecs), w2i


def pca_dir(pairs, E, w2i):
    rows = []
    for a, b in pairs:
        centre = E[w2i[a]] + E[w2i[b]] / 2  # sic, as in the notebook
        rows += [E[w2i[a]] - centre, E[w2i[b]] - centre]
    return PCA(n_components=10).fit(np.array(rows)).components_[0]


def nouns(name, w2i):
    out = []
    for line in open(Z / name, encoding="utf-8"):
        w = line.strip()
        if w and w in w2i and w not in out:
            out.append(w)
    return out


def mweat(E, w2i, pairs, am=WEAT_M, af=WEAT_F):
    tm = tf = 0.0
    for m, f in pairs:
        tm += np.mean([E[w2i[a]] @ E[w2i[m]] for a in am]) - np.mean([E[w2i[a]] @ E[w2i[m]] for a in af])
        tf += np.mean([E[w2i[a]] @ E[w2i[f]] for a in am]) - np.mean([E[w2i[a]] @ E[w2i[f]] for a in af])
    return abs(abs(tm) - abs(tf))


def pval(E, w2i, pairs, n=1000, seed=520, am=WEAT_M, af=WEAT_F):
    random.seed(seed)
    stat = mweat(E, w2i, pairs, am, af)
    tw1 = [m for m, _ in pairs]
    ties = []
    allw = tw1 + [f for _, f in pairs]
    over = []
    for _ in range(n):
        c = random.sample(allw, len(tw1))
        rest = list(filterfalse(lambda x, c=c: x in c, allw))
        v = mweat(E, w2i, list(zip(c, rest)), am, af)
        over.append(v > stat)
        ties.append(abs(v - stat) < 1e-6)
    return stat, float(np.mean(over)), float(np.mean(ties))


def shift(E, w2i, pairs, ds):
    """Shift Ori: move each occupation pair along ds so the two forms are symmetric about 0."""
    E2 = E.copy()
    for m, f in pairs:
        off = -0.5 * (E[w2i[m]] @ ds + E[w2i[f]] @ ds)
        E2[w2i[m]] += off * ds / (ds @ ds)
        E2[w2i[f]] += off * ds / (ds @ ds)
    return E2


def main(lang="es"):
    out = OUT.parent / f"zhou_shift_{lang}"
    out.mkdir(parents=True, exist_ok=True)
    E, w2i = load_vec(Z / f"data/wiki.multi.{lang}.vec.txt")
    occ = [line.rstrip("\n").split("\t") for line in open(Z / f"{lang.upper()}_occupation_words_with_EN_translations")]
    pairs = [(m, f) for f, m, _ in occ]
    defs = json.load(open(Z / f"{lang.upper()}_definitional_pairs.json"))
    # attributes: ES_WEAT_male/female for Spanish; the definitional pairs for French (notebook cell 215)
    am, af = (WEAT_M, WEAT_F) if lang == "es" else ([b for _, b in defs], [a for a, _ in defs])
    d_pca = pca_dir(defs, E, w2i)
    m_n, f_n = nouns(f"masculine_nouns_{lang}", w2i), nouns(f"feminine_nouns_{lang}", w2i)
    gp = list(zip(f_n, m_n))[:3000]
    X = np.array([E[w2i[w]] for p in gp for w in p])
    y = np.tile([1, 2], len(gp))
    res = {"n_nouns_in_vocab": [len(m_n), len(f_n)], "n_occupation_pairs": len(pairs)}
    dirs = {}
    for name, clf in [("svc", LinearSVC()), ("lda", LinearDiscriminantAnalysis(n_components=1))]:
        clf.fit(X, y)
        g = clf.coef_.reshape(-1)
        dirs[name] = g / np.linalg.norm(g)
        res[f"cos_pca_grammatical_{name}"] = float(abs(d_pca @ dirs[name]))
    res["original"] = pval(E, w2i, pairs, am=am, af=af)
    for name, g in dirs.items():
        ds = d_pca - (d_pca @ g) * g / (g @ g)
        res[f"shift_ori_{name}"] = pval(shift(E, w2i, pairs, ds), w2i, pairs, am=am, af=af)
    res["shift_ori_pca_unorthogonalised"] = pval(shift(E, w2i, pairs, d_pca), w2i, pairs, am=am, af=af)
    json.dump(res, open(out / "results.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "es")
