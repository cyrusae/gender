"""Is Zhou et al.'s Shift Ori → MWEAT drop built in? (exploratory diagnostic, 2026-11-08)

MWEAT per word = mean dot with male attributes − mean dot with female attributes = w · a, with
a = mean(male attr) − mean(female attr). Shift moves each occupation word only along ds (the
orthogonalised semantic direction), so it can change w · a only through a's component along ds.
Reports cos(a, ds), cos(a, d_pca), and MWEAT recomputed with a's ds-component removed, before and
after Shift; plus which words are shifted vs scored (from the notebook cells).

    uv run --no-project --python 3.12 --with numpy,scikit-learn python embeddings/scripts/zhou_construct.py
"""

import json

import numpy as np
from sklearn.svm import LinearSVC
from zhou_shift import attributes, Z, load_vec, nouns, pca_dir, shift

OUT = Z.parents[1] / "results/zhou_construct.json"


def stat(E, w2i, pairs, a):
    tm = sum(E[w2i[m]] @ a for m, _ in pairs)
    tf = sum(E[w2i[f]] @ a for _, f in pairs)
    return abs(abs(tm) - abs(tf)), tm, tf


res = {}
for lang in ("es", "fr"):
    E, w2i = load_vec(Z / f"data/wiki.multi.{lang}.vec.txt")
    occ = [line.rstrip("\n").split("\t") for line in open(Z / f"{lang.upper()}_occupation_words_with_EN_translations")]
    pairs = [(m, f) for f, m, _ in occ]
    defs = json.load(open(Z / f"{lang.upper()}_definitional_pairs.json"))
    am, af = attributes(lang, defs)
    a = np.mean([E[w2i[w]] for w in am], 0) - np.mean([E[w2i[w]] for w in af], 0)
    d = pca_dir(defs, E, w2i)
    gp = list(zip(nouns(f"feminine_nouns_{lang}", w2i), nouns(f"masculine_nouns_{lang}", w2i)))[:3000]
    X = np.array([E[w2i[w]] for p in gp for w in p])
    g = LinearSVC().fit(X, np.tile([1, 2], len(gp))).coef_.reshape(-1)
    g /= np.linalg.norm(g)
    ds = d - (d @ g) * g
    ds /= np.linalg.norm(ds)
    a_perp = a - (a @ ds) * ds
    S = shift(E, w2i, pairs, ds)
    c = lambda u, v: float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))  # noqa: E731
    res[lang] = {
        "cos_attr_axis_ds": c(a, ds), "cos_attr_axis_d_pca": c(a, d), "cos_attr_axis_grammatical": c(a, g),
        "share_of_attr_axis_along_ds": c(a, ds) ** 2,
        "mweat_original": stat(E, w2i, pairs, a)[0],
        "mweat_original_ds_part_removed_from_attr_axis": stat(E, w2i, pairs, a_perp)[0],
        "mweat_after_shift": stat(S, w2i, pairs, a)[0],
        "mweat_after_shift_ds_part_removed": stat(S, w2i, pairs, a_perp)[0],
        "after_shift_Tm_Tf": stat(S, w2i, pairs, a)[1:],
    }
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
