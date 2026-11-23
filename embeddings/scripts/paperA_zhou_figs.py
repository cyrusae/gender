"""Paper A: Zhou et al. (2019) Figures 2-4 replicated, each with our additions (exploratory, 2026-11-23).
Figure 1 is scripts/zhou_figures.py (zhou_fig1_redrawn.png).

Fig. 2  five occupations on the semantic gender direction: Spanish masculine / feminine forms, the English
        word, and the English word after hard debiasing (Spanish aligned onto it, as De-Align).
        Replicated: their directions (notebook centring, LinearSVC on their nouns), origin as neutral.
        Added: the epicene zero point; all 58 pairs; and, per pair, which Spanish form the English word is
        closer to in the full space (cosine), not only on this one direction.
Fig. 3  as Fig. 2 after Hybrid Ori. Replicated: the two forms symmetric about the origin.
        Added: symmetric about the origin by construction of Shift (each pair's sum on d_s is set to 0),
        not about the epicene zero point; and the full-space check after Hybrid.
Fig. 4  per-occupation WEAT (English) vs MWEAT (Spanish, per pair |s(m)| − |s(f)|, the form that reproduces 0.45), their 10
        labelled occupations. Replicated: Spearman 0.45. Added: all 58 points, the clean-list (54) ρ.

Alignment (De-Align, Hybrid) is the Procrustes rebuild of scripts/zhou_hybrid.py (B9: not released).

    uv run --no-project --python 3.12 --with numpy,pandas,scikit-learn,matplotlib python embeddings/scripts/paperA_zhou_figs.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402
from zhou_hybrid import DICT, hard_debias, procrustes  # noqa: E402
from zhou_shift import Z, load_vec, nouns, pca_dir, shift  # noqa: E402
from figpanels import save_panels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paperA_zhou_figs"
FIG2 = [("enfermero", "enfermera", "nurse"), ("doctor", "doctora", "doctor"), ("profesor", "profesora", "professor"),
        ("abogado", "abogada", "lawyer"), ("arquitecto", "arquitecta", "architect")]  # fmt: skip
FIG4 = ["lawyer", "librarian", "nurse", "mathematician", "reader", "musician", "reporter", "adviser", "pharmacist", "chief"]
M, F, EN, ENDB = "#3b6fb6", "#c0473b", "#9e9e9e", "#000000"


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    es, es_w2i = load_vec(Z / "data/wiki.multi.es.vec.txt")
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    en_db, _ = hard_debias(en, en_w2i)
    seen, src, tgt = set(), [], []
    for line in open(DICT, encoding="utf-8"):
        a, b = line.split()
        if a in es_w2i and b in en_w2i and a not in seen:
            seen.add(a)
            src.append(es_w2i[a])
            tgt.append(en_w2i[b])
    order = np.argsort(src)[:5000]
    W = procrustes(es[np.array(src)[order]], en_db[np.array(tgt)[order]])  # es @ W.T lives in debiased-English frame

    # their directions (as published), in the Spanish frame
    defs = json.load(open(Z / "ES_definitional_pairs.json"))
    gp = list(zip(nouns("feminine_nouns_es", es_w2i), nouns("masculine_nouns_es", es_w2i)))[:3000]
    X = np.array([es[es_w2i[w]] for p in gp for w in p])
    g = unit(LinearSVC().fit(X, np.tile([0, 1], len(gp))).coef_.reshape(-1))
    d = unit(pca_dir(defs, es, es_w2i))
    d = d if es[es_w2i["hombre"]] @ d > es[es_w2i["mujer"]] @ d else -d
    ds = unit(d - (d @ g) * g)  # semantic direction, male positive
    ds_db = W @ ds  # the same direction in the debiased-English frame

    occ = [line.rstrip("\n").split("\t") for line in open(Z / "ES_occupation_words_with_EN_translations")]
    occ = [(m, f, e) for f, m, e in occ]
    pairs = [(m, f) for m, f, _ in occ]
    # Hybrid Ori: Shift, then align onto debiased English. Their Fig. 2/3 include doctor/doctora, which is not
    # among the 58 scored pairs (B12), so the five figure pairs are shifted too, as in their figure.
    fig_pairs = [(m, f) for m, f, _ in FIG2 if (m, f) not in pairs]
    es_hy = shift(es, es_w2i, pairs + fig_pairs, ds) @ W.T

    ref = pd.read_csv(ROOT / "data/zero_ref_es_frozen_v1.csv")
    epi = ref[(ref["set"] == "epicene") & (ref["tier"] == "strict") & ref["lemma"].isin(es_w2i)]
    pe = es[epi["lemma"].map(es_w2i)] @ ds
    zero = 0.5 * (pe[(epi["gender"] == "m").to_numpy()].mean() + pe[(epi["gender"] == "f").to_numpy()].mean())

    rows = []
    for m, f, e in occ + [x for x in FIG2 if (x[0], x[1]) in fig_pairs]:
        r = {"masc": m, "fem": f, "en": e,
             "orig_m": es[es_w2i[m]] @ ds, "orig_f": es[es_w2i[f]] @ ds,
             "hyb_m": es_hy[es_w2i[m]] @ ds_db, "hyb_f": es_hy[es_w2i[f]] @ ds_db}  # fmt: skip
        if e in en_w2i:
            v, vdb = en[en_w2i[e]], en_db[en_w2i[e]]
            r |= {"en_orig": v @ ds, "en_db": vdb @ ds_db,
                  "full_orig": v @ es[es_w2i[m]] - v @ es[es_w2i[f]],
                  "full_dealign": vdb @ (es[es_w2i[m]] @ W.T) - vdb @ (es[es_w2i[f]] @ W.T),
                  "full_hybrid": vdb @ es_hy[es_w2i[m]] - vdb @ es_hy[es_w2i[f]]}  # fmt: skip
        r["in_58"] = (m, f) in pairs
        rows.append(r)
    allP = pd.DataFrame(rows)
    P = allP[allP.in_58].reset_index(drop=True)
    allP.to_csv(OUT / "fig2_fig3_points.csv", index=False)
    full = P.dropna(subset=["full_orig"])
    stats = {"epicene_zero_point": zero, "n_pairs": len(P), "n_with_english": len(full),
             "fig2_pairs_fem_further_from_origin": int((P.orig_f.abs() > P.orig_m.abs()).sum()),
             "fig2_pairs_fem_further_from_epicene_zero": int(((P.orig_f - zero).abs() > (P.orig_m - zero).abs()).sum()),
             "fig3_max_abs_pair_sum_on_ds": float((P.hyb_m + P.hyb_f).abs().max()),
             "english_closer_to_masc_full_space": {k: int((full[k] > 0).sum()) for k in ("full_orig", "full_dealign", "full_hybrid")}}  # fmt: skip

    # ---- Fig. 2 and Fig. 3: replicated (5 pairs) + additions
    for tag, cm, cf, ce, title in (("fig2", "orig_m", "orig_f", "en_orig", "Figure 2"),
                                   ("fig3", "hyb_m", "hyb_f", "en_db", "Figure 3")):  # fmt: skip
        fig, ax = plt.subplots(1, 3, figsize=(18, 4.6), gridspec_kw={"width_ratios": [1, 1.1, 0.9]})
        a = ax[0]
        for i, (m, f, e) in enumerate(FIG2):
            r = allP[allP.masc == m].iloc[0]
            a.plot([r[cf], r[cm]], [i, i], c="#ccc", lw=0.8, zorder=0)
            a.scatter(r[cm], i, c=M, s=34)
            a.scatter(r[cf], i, c=F, s=34)
            if tag == "fig2":
                a.scatter(r.en_orig, i + 0.15, c=EN, s=30, marker="s")
                a.scatter(r.en_db, i - 0.15, c=ENDB, s=30, marker="*")
            else:
                a.scatter(r.en_db, i, c=ENDB, s=40, marker="*")
        a.axvline(0, c="grey", lw=0.8)
        a.set_yticks(range(len(FIG2)), [f"{m} / {f}" for m, f, _ in FIG2])
        a.set(title=f"A. {title} replicated (5 occupations, zero = origin)", xlabel="semantic gender direction (male →)")
        b = ax[1]
        o = P.sort_values(cm).reset_index(drop=True)
        for i, r in o.iterrows():
            b.plot([r[cf], r[cm]], [i, i], c="#ddd", lw=0.6, zorder=0)
        b.scatter(o[cm], o.index, c=M, s=10, label="masculine form")
        b.scatter(o[cf], o.index, c=F, s=10, label="feminine form")
        b.axvline(0, c="grey", lw=0.8, label="origin (their neutral point)")
        b.axvline(zero, c="black", ls="--", lw=1, label=f"epicene zero point ({zero:+.3f})")
        b.set_yticks([])
        lab = "B. All 58 pairs" + (" after Hybrid: symmetric about the origin by construction" if tag == "fig3" else "")
        b.set(title=lab, xlabel="semantic gender direction (male →)")
        b.legend(fontsize=8, loc="lower right")
        c = ax[2]
        col = "full_orig" if tag == "fig2" else "full_hybrid"
        v = full.sort_values(col)[col].to_numpy()
        c.barh(range(len(v)), v, color=np.where(v > 0, M, F), height=0.8)
        c.axvline(0, c="black", lw=0.8)
        c.set_yticks([])
        n = int((v > 0).sum())
        c.set(title=f"C. Full space: English word closer to masc. form\nin {n} of {len(v)} pairs"
                    + (" (after Hybrid)" if tag == "fig3" else ""),
              xlabel="cos(English, masc.) − cos(English, fem.)")  # fmt: skip
        fig.text(0.01, 0.01, "blue = Spanish masculine form · red = feminine · grey square = English word · black star = hard-debiased English (aligned)",
                 fontsize=8)  # fmt: skip
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(OUT / f"zhou_{tag}_redrawn.png", dpi=150)
        save_panels(fig, OUT, f"zhou_{tag}_redrawn")

    # ---- Fig. 4: replicated + all points
    per = pd.read_csv(ROOT / "results/zhou_sec43_44/per_occupation.csv")
    corr = pd.read_csv(ROOT / "results/zhou_sec43_44/correlation.csv")
    rho = corr.set_index(["list", "spanish_score"]).spearman
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    lab = per[per.en.isin(FIG4)]
    for a, pts, title in ((ax[0], lab, f"A. Figure 4 replicated (their 10 labels); ρ over 58 = {rho[('as_published (58)', 'mweat_es_i')]:.2f}"),
                          (ax[1], per, f"B. All 58 occupations; clean list (54) ρ = {rho[('clean (54)', 'mweat_es_i')]:.2f}")):  # fmt: skip
        if a is ax[1]:
            a.scatter(per[~per.clean].weat_en, per[~per.clean].mweat_es_i, facecolors="none", edgecolors="#888", s=30,
                      label="dropped from clean list")  # fmt: skip
            a.scatter(per[per.clean].weat_en, per[per.clean].mweat_es_i, c="#555", s=16, label="clean list")
            a.legend(fontsize=8)
        else:
            a.scatter(pts.weat_en, pts.mweat_es_i, c="#333", s=22)
            for _, r in pts.iterrows():
                a.annotate(r.en, (r.weat_en, r.mweat_es_i), xytext=(3, 3), textcoords="offset points", fontsize=8)
        a.axhline(0, c="#ccc", lw=0.6)
        a.axvline(0, c="#ccc", lw=0.6)
        a.set(title=title, xlabel="WEAT, English (← female)", ylabel="MWEAT, Spanish (← female)")
    fig.tight_layout()
    fig.savefig(OUT / "zhou_fig4_redrawn.png", dpi=150)
    save_panels(fig, OUT, "zhou_fig4_redrawn")
    stats["fig4_spearman"] = {f"{i[0]}|{i[1]}": float(v) for i, v in rho.items()}
    json.dump(stats, open(OUT / "stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
