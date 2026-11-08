"""Rebuild Zhou et al. (2019) De-Align and Hybrid Ori for Spanish; check against Table 1.

Paper §3.3: hard-debias English (Bolukbasi et al. 2016), align Spanish to the debiased English,
then (Hybrid) shift occupation pairs along the semantic gender direction with the origin as anchor.
The repo ships none of the intermediate files and the paper doesn't say how the alignment was done,
so this uses the standard choice: orthogonal Procrustes (as MUSE supervised alignment does) from
`wiki.multi.es` onto hard-debiased `wiki.multi.en`, fitted on MUSE es-en dictionary pairs.

Hard debias = debiaswe `debias()` (external/debiaswe, commit 6bb03ee) re-implemented in numpy on the
loaded vocabulary: neutralise every word not in gender_specific_full.json, renormalise, equalise the
52 equalize pairs (lower/title/upper case variants), renormalise.

Table 1 (ES): Original 3.6918; De-Align 3.5748 (p 0.001); Hybrid Ori 0.3090 (p 0.733).
Note: an orthogonal map leaves all Spanish-internal dot products unchanged, so with Procrustes
De-Align must equal Original and Hybrid Ori must equal Shift Ori on this Spanish-only statistic.

    uv run --no-project --python 3.12 --with numpy,scikit-learn python embeddings/scripts/zhou_hybrid.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.svm import LinearSVC
from zhou_shift import OUT as SHIFT_OUT
from zhou_shift import Z, load_vec, nouns, pca_dir, pval, shift

ROOT = Path(__file__).resolve().parents[1]
DW = ROOT / "external/debiaswe/data"
DICT = Path("/Volumes/Lexar/gbleed/shared/embeddings/muse/dictionaries/es-en.txt")
OUT = ROOT / "results/zhou_hybrid_es"


def drop(u, v):
    return u - v * (u @ v) / (v @ v)


def hard_debias(E, w2i):
    """debiaswe.debias.debias(), numpy version (unit-norm rows in, unit-norm rows out)."""
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    defs = json.load(open(DW / "definitional_pairs.json"))
    spec = set(json.load(open(DW / "gender_specific_full.json")))
    eq = json.load(open(DW / "equalize_pairs.json"))
    # debiaswe's doPCA uses the true pair midpoint
    rows = []
    for a, b in defs:
        if a in w2i and b in w2i:
            c = (E[w2i[a]] + E[w2i[b]]) / 2
            rows += [E[w2i[a]] - c, E[w2i[b]] - c]
    from sklearn.decomposition import PCA

    g = PCA(n_components=10).fit(np.array(rows)).components_[0]
    keep = np.array([w in spec for w in w2i])
    proj = E @ g
    E[~keep] -= np.outer(proj[~keep], g)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    cands = {x for e1, e2 in eq for x in [(e1.lower(), e2.lower()), (e1.title(), e2.title()), (e1.upper(), e2.upper())]}
    n_eq = 0
    for a, b in cands:
        if a in w2i and b in w2i:
            y = drop((E[w2i[a]] + E[w2i[b]]) / 2, g)
            z = np.sqrt(max(0.0, 1 - y @ y))
            if (E[w2i[a]] - E[w2i[b]]) @ g < 0:
                z = -z
            E[w2i[a]], E[w2i[b]] = z * g + y, -z * g + y
            n_eq += 1
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return E, n_eq


def procrustes(X, Y):
    u, _, vt = np.linalg.svd(Y.T @ X)
    return u @ vt  # W with W X_i ≈ Y_i


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    es, es_w2i = load_vec(Z / "data/wiki.multi.es.vec.txt")
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    en_db, n_eq = hard_debias(en, en_w2i)
    seen, src, tgt = set(), [], []
    for line in open(DICT, encoding="utf-8"):
        a, b = line.split()
        if a in es_w2i and b in en_w2i and a not in seen:
            seen.add(a)
            src.append(es_w2i[a])
            tgt.append(en_w2i[b])
    order = np.argsort(src)[:5000]  # the 5,000 most frequent Spanish entries, as MUSE's 0-5000 train split
    W = procrustes(es[np.array(src)[order]], en_db[np.array(tgt)[order]])
    es_da = es @ W.T
    res = {"n_equalised_pairs": n_eq, "n_dict_pairs": len(order),
           "orthogonality_error": float(np.abs(W @ W.T - np.eye(300)).max())}  # fmt: skip

    occ = [line.rstrip("\n").split("\t") for line in open(Z / "ES_occupation_words_with_EN_translations")]
    pairs = [(m, f) for f, m, _ in occ]
    defs = json.load(open(Z / "ES_definitional_pairs.json"))
    m_n, f_n = nouns("masculine_nouns_es", es_w2i), nouns("feminine_nouns_es", es_w2i)
    gp = list(zip(f_n, m_n))[:3000]
    res["original"] = pval(es, es_w2i, pairs)
    res["de_align"] = pval(es_da, es_w2i, pairs)
    for name, E in [("shift_ori_recomputed", es), ("hybrid_ori", es_da)]:
        d = pca_dir(defs, E, es_w2i)
        X = np.array([E[es_w2i[w]] for p in gp for w in p])
        g = LinearSVC().fit(X, np.tile([1, 2], len(gp))).coef_.reshape(-1)
        g /= np.linalg.norm(g)
        ds = d - (d @ g) * g
        res[name] = pval(shift(E, es_w2i, pairs, ds), es_w2i, pairs)
        res[f"{name}_cos_sem_gram"] = float(abs(d @ g))
    # cross-lingual check that the alignment does something: mean cos of dictionary pairs
    res["dict_cos_before"] = float(np.mean(np.sum(es[src][:1000] * en[tgt][:1000], 1)))
    res["dict_cos_after"] = float(np.mean(np.sum(es_da[src][:1000] * en_db[tgt][:1000], 1)))
    res["shift_ori_from_zhou_shift"] = json.load(open(SHIFT_OUT / "results.json"))["shift_ori_svc"]
    json.dump(res, open(OUT / "results.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
