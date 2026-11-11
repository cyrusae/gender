"""Zhou et al. 2019 Figures 1-2, redrawn as published and with corrected inputs (exploratory, 2026-11-10).

Panel A  as published: their 10 words; d_PCA with the notebook's centring slip; d_g = LinearSVC on
         their first 3,000 noun pairs (what produced cos 0.389). Words projected as unit vectors.
Panel B  corrected: true pair-midpoint PCA; d_g = LinearSVC on our Wiktionary-labelled C1 nouns
         (data/c1_nouns_es_v1.csv, in 50k); semantic axis orthogonalised to it (their d_s);
         all 58 occupation pairs; inanimate C1 nouns as a cloud; the epicene zero point marked.
Panel C  semantic-axis projections of the 58 masculine vs 58 feminine forms, distances from the
         epicene zero point rather than from 0 (the skew Zhou's Fig. 1 describes).

    uv run --no-project --python 3.12 --with numpy,pandas,scikit-learn,matplotlib python embeddings/scripts/zhou_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402
from zhou_shift import Z, load_vec, nouns, pca_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/zhou_figures"
FIG1 = ["mujer", "hombre", "doctora", "doctor", "abogada", "abogado", "cuchara", "cuchillo", "agua", "mapa"]
FIG2 = [("enfermero", "enfermera"), ("doctor", "doctora"), ("profesor", "profesora"),
        ("abogado", "abogada"), ("arquitecto", "arquitecta")]  # fmt: skip
M, F = "#3b6fb6", "#c0473b"


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def pca_true(pairs, E, w2i):
    rows = []
    for a, b in pairs:
        c = (E[w2i[a]] + E[w2i[b]]) / 2
        rows += [E[w2i[a]] - c, E[w2i[b]] - c]
    return PCA(n_components=10).fit(np.array(rows)).components_[0]


def orient(d, E, w2i):
    """Sign so that 'hombre' is positive (masculine side right, as in their figures)."""
    return d if E[w2i["hombre"]] @ d > E[w2i["mujer"]] @ d else -d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    E, w2i = load_vec(Z / "data/wiki.multi.es.vec.txt")
    U = unit(E)
    defs = json.load(open(Z / "ES_definitional_pairs.json"))
    occ = [line.rstrip("\n").split("\t") for line in open(Z / "ES_occupation_words_with_EN_translations")]
    pairs = [(m, f) for f, m, _ in occ]

    # as published
    gp = list(zip(nouns("feminine_nouns_es", w2i), nouns("masculine_nouns_es", w2i)))[:3000]
    X = np.array([E[w2i[w]] for p in gp for w in p])
    g_pub = unit(LinearSVC().fit(X, np.tile([0, 1], len(gp))).coef_.reshape(-1))  # masculine = 1 → positive
    d_pub = orient(unit(pca_dir(defs, E, w2i)), E, w2i)
    ds_pub = unit(d_pub - (d_pub @ g_pub) * g_pub)

    # corrected
    c1 = pd.read_csv(ROOT / "data/c1_nouns_es_v1.csv")
    c1 = c1[c1["in_50k"] & c1["lemma"].isin(w2i)]
    Xc = U[c1["lemma"].map(w2i)]
    g_cor = unit(LinearSVC().fit(Xc, (c1["gender"] == "m").astype(int)).coef_.reshape(-1))
    d_cor = orient(unit(pca_true(defs, E, w2i)), E, w2i)
    ds_cor = unit(d_cor - (d_cor @ g_cor) * g_cor)
    ref = pd.read_csv(ROOT / "data/zero_ref_es_frozen_v1.csv")
    epi = ref[(ref["set"] == "epicene") & (ref["tier"] == "strict") & ref["lemma"].isin(w2i)]
    pe = U[epi["lemma"].map(w2i)] @ ds_cor
    zero = 0.5 * (pe[(epi["gender"] == "m").to_numpy()].mean() + pe[(epi["gender"] == "f").to_numpy()].mean())

    sem_m = np.array([U[w2i[m]] @ ds_cor for m, _ in pairs])
    sem_f = np.array([U[w2i[f]] @ ds_cor for _, f in pairs])
    gra_m = np.array([U[w2i[m]] @ g_cor for m, _ in pairs])
    gra_f = np.array([U[w2i[f]] @ g_cor for _, f in pairs])
    ina_s, ina_g = Xc @ ds_cor, Xc @ g_cor

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1, 1.25, 1]})
    # A
    a = ax[0]
    for w in FIG1:
        x, y = U[w2i[w]] @ ds_pub, U[w2i[w]] @ g_pub
        col = M if w in ("hombre", "doctor", "abogado", "cuchillo", "mapa") else F
        a.scatter(x, y, c=col, s=28)
        a.annotate(w, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=9)
    a.axhline(0, c="grey", lw=.6)
    a.axvline(0, c="grey", lw=.6)
    a.set(title="A. As published (their directions, 10 hand-picked words)",
          xlabel="semantic gender direction (male →)", ylabel="grammatical gender direction (masc. →)")
    # B
    b = ax[1]
    b.scatter(ina_s, ina_g, c=np.where(c1["gender"] == "m", M, F), s=4, alpha=.12, lw=0, label="inanimate nouns (C1)")
    for (m, f), xm, xf, ym, yf in zip(pairs, sem_m, sem_f, gra_m, gra_f):
        b.plot([xf, xm], [yf, ym], c="grey", lw=.4, alpha=.6)
    b.scatter(sem_m, gra_m, c=M, s=18, label="occupations, masculine form")
    b.scatter(sem_f, gra_f, c=F, s=18, label="occupations, feminine form")
    b.axvline(0, c="grey", lw=.6)
    b.axvline(zero, c="black", ls="--", lw=1, label=f"epicene zero point ({zero:+.3f})")
    b.axhline(0, c="grey", lw=.6)
    b.legend(fontsize=8, loc="lower right")
    b.set(title="B. Corrected inputs, all 58 pairs + 2,000 inanimates",
          xlabel="semantic gender direction (male →)", ylabel="grammatical gender direction (masc. →)")
    # C
    c = ax[2]
    dm, df = sem_m - zero, sem_f - zero
    bins = np.linspace(min(dm.min(), df.min()), max(dm.max(), df.max()), 24)
    c.hist(dm, bins=bins, color=M, alpha=.6, label=f"masculine forms, mean {dm.mean():+.3f}")
    c.hist(df, bins=bins, color=F, alpha=.6, label=f"feminine forms, mean {df.mean():+.3f}")
    c.axvline(0, c="black", ls="--", lw=1)
    c.legend(fontsize=8)
    c.set(title="C. Distance from the zero point, semantic axis",
          xlabel="projection − epicene zero point (male →)", ylabel="occupation forms")
    fig.tight_layout()
    fig.savefig(OUT / "zhou_fig1_redrawn.png", dpi=150)

    # Fig. 2: five pairs on the semantic axis, as published vs corrected, with zero point
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for k, (d, z, title) in enumerate([(ds_pub, 0.0, "As published (zero = origin)"),
                                       (ds_cor, zero, "Corrected (dashed = epicene zero point)")]):  # fmt: skip
        a = ax2[k]
        for i, (m, f) in enumerate(FIG2):
            xm, xf = U[w2i[m]] @ d, U[w2i[f]] @ d
            a.plot([xf, xm], [i, i], c="grey", lw=.8)
            a.scatter([xm], [i], c=M, s=30)
            a.scatter([xf], [i], c=F, s=30)
        a.axvline(0, c="grey", lw=.8)
        if k:
            a.axvline(z, c="black", ls="--", lw=1)
        a.set_yticks(range(len(FIG2)), [f"{m} / {f}" for m, f in FIG2])
        a.set(title=title, xlabel="semantic gender direction (male →)")
    fig2.tight_layout()
    fig2.savefig(OUT / "zhou_fig2_redrawn.png", dpi=150)

    stats = {"cos_dpca_dg_published": float(abs(unit(pca_dir(defs, E, w2i)) @ g_pub)),
             "cos_dpca_dg_corrected": float(abs(d_cor @ g_cor)),
             "epicene_zero_point_semantic_axis": float(zero),
             "inanimate_mean_semantic_axis": {"m": float(ina_s[(c1["gender"] == "m").to_numpy()].mean()),
                                              "f": float(ina_s[(c1["gender"] == "f").to_numpy()].mean())},
             "occupations_mean_distance_from_zero": {"m": float(dm.mean()), "f": float(df.mean())},
             "occupations_mean_distance_from_origin": {"m": float(sem_m.mean()), "f": float(sem_f.mean())},
             "pairs_where_feminine_further_from_zero": int((np.abs(df) > np.abs(dm)).sum()),
             "n_pairs": len(pairs)}  # fmt: skip
    json.dump(stats, open(OUT / "stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
