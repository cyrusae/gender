"""Figure (exploratory, 2026-11-22, PI request): De-Align inherits its effect from the debiased English.

De-Align (Zhou et al. §3.3) rotates Spanish onto hard-debiased English. A rotation can't change anything
inside Spanish; all of its effect comes from the English vectors having been edited. Three panels, Spanish,
the 58 occupation pairs (rebuild as scripts/zhou_hybrid.py: orthogonal Procrustes on the 5,000 most
frequent MUSE es-en dictionary pairs):
  A  inside Spanish: each occupation form's MWEAT lean (native attribute words, corrected list),
     Original vs De-Align;
  B  inside English: each English gloss's position on English's gender direction (debiaswe's PCA direction),
     before vs after hard debiasing;
  C  across languages: for each occupation, how much closer its English gloss is to the masculine than to
     the feminine Spanish form (cos(en, masc) − cos(en, fem)), Original vs De-Align, and vs a control that
     runs the same Procrustes step onto the *undebiased* English.

    uv run --no-project --python 3.12 --with numpy,scikit-learn,matplotlib python embeddings/scripts/dealign_figure.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from zhou_hybrid import DICT, DW, hard_debias, procrustes
from zhou_shift import Z, attributes, load_vec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/dealign_figure"
EN_M = ["male", "he", "him", "his", "man", "boy", "son", "brother"]
EN_F = ["female", "she", "her", "hers", "woman", "girl", "daughter", "sister"]


def gender_dir(E, w2i):
    rows = []
    for a, b in json.load(open(DW / "definitional_pairs.json")):
        if a in w2i and b in w2i:
            c = (E[w2i[a]] + E[w2i[b]]) / 2
            rows += [E[w2i[a]] - c, E[w2i[b]] - c]
    return PCA(n_components=10).fit(np.array(rows)).components_[0]


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
    S, T = np.array(src)[order], np.array(tgt)[order]
    es_da = es @ procrustes(es[S], en_db[T]).T  # De-Align
    es_ctl = es @ procrustes(es[S], en[T]).T  # control: same step onto undebiased English

    occ = [line.rstrip("\n").split("\t") for line in open(Z / "ES_occupation_words_with_EN_translations")]
    occ = [(m, f, e) for f, m, e in occ if m in es_w2i and f in es_w2i and e in en_w2i]
    defs = json.load(open(Z / "ES_definitional_pairs.json"))
    am, af = attributes("es", defs)
    def lean_es(E, w):  # attribute vector built in the same space as the word
        a = np.mean([E[es_w2i[x]] for x in am], 0) - np.mean([E[es_w2i[x]] for x in af], 0)
        return E[es_w2i[w]] @ a
    a_en = np.mean([en[en_w2i[w]] for w in EN_M], 0) - np.mean([en[en_w2i[w]] for w in EN_F], 0)
    a_en_db = np.mean([en_db[en_w2i[w]] for w in EN_M], 0) - np.mean([en_db[en_w2i[w]] for w in EN_F], 0)
    g = gender_dir(en, en_w2i)

    rows = []
    for m, f, e in occ:
        for form, w in (("m", m), ("f", f)):
            rows.append({"form": form, "word": w, "en": e,
                         "A_orig": lean_es(es, w), "A_dealign": lean_es(es_da, w),
                         "C_orig": es[es_w2i[w]] @ en[en_w2i[e]], "C_dealign": es_da[es_w2i[w]] @ en_db[en_w2i[e]],
                         "C_control": es_ctl[es_w2i[w]] @ en[en_w2i[e]]})  # fmt: skip
    B = [(e, en[en_w2i[e]] @ g, en_db[en_w2i[e]] @ g) for e in dict.fromkeys(x[2] for x in occ)]
    import csv

    with open(OUT / "points.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    A0, A1 = np.array([r["A_orig"] for r in rows]), np.array([r["A_dealign"] for r in rows])
    C0, C1, Cc = (np.array([r[k] for r in rows]) for k in ("C_orig", "C_dealign", "C_control"))
    fm = np.array([r["form"] == "f" for r in rows])
    gap = lambda x: x[~fm].mean() - x[fm].mean()  # noqa: E731
    summ = {"n_pairs": len(occ),
            "A_max_abs_change": float(np.abs(A1 - A0).max()),
            "B_mean_abs_before": float(np.mean([abs(b) for _, b, _ in B])), "B_mean_abs_after": float(np.mean([abs(c) for _, _, c in B])),
            "C_mf_gap_original": float(gap(C0)), "C_mf_gap_control": float(gap(Cc)), "C_mf_gap_dealign": float(gap(C1)),
            "C_mean_abs_change_control": float(np.abs(Cc - C0).mean()), "C_mean_abs_change_dealign": float(np.abs(C1 - C0).mean()),
            "C_pairs_masc_closer_original": int(((C0[~fm] - C0[fm]) > 0).sum()), "C_pairs_masc_closer_dealign": int(((C1[~fm] - C1[fm]) > 0).sum()),
            "C_pairs_masc_closer_control": int(((Cc[~fm] - Cc[fm]) > 0).sum())}  # fmt: skip
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)

    col = np.where(fm, "#c2185b", "#1565c0")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    lim = max(abs(A0).max(), abs(A1).max()) * 1.1
    ax[0].scatter(A0, A1, c=col, s=18)
    ax[0].plot([-lim, lim], [-lim, lim], "k:", lw=0.8)
    ax[0].set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="Original", ylabel="De-Align",
              title="A. Inside Spanish: nothing moves\n(lean on Spanish attribute words)")  # fmt: skip
    bb = np.array([[b, c] for _, b, c in B])
    lim = abs(bb).max() * 1.1
    ax[1].scatter(bb[:, 0], bb[:, 1], c="#555", s=18)
    ax[1].plot([-lim, lim], [-lim, lim], "k:", lw=0.8)
    ax[1].set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="English, original", ylabel="English, hard-debiased",
              title="B. Inside English: the debiasing\n(English glosses on the gender direction)")  # fmt: skip
    D0, D1, Dc = (x[~fm] - x[fm] for x in (C0, C1, Cc))  # per occupation: masculine minus feminine
    lim = max(abs(D0).max(), abs(D1).max(), abs(Dc).max()) * 1.1
    ax[2].scatter(D0, Dc, facecolors="none", edgecolors="#555", s=22, label="control: aligned to undebiased English")
    ax[2].scatter(D0, D1, c="#e65100", s=18, label="De-Align: aligned to debiased English")
    ax[2].plot([-lim, lim], [-lim, lim], "k:", lw=0.8)
    ax[2].set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="Original", ylabel="after alignment",
              title="C. Across languages: inherited from English\n(English gloss: closeness to masc. minus fem. form)")  # fmt: skip
    ax[2].legend(fontsize=8, loc="upper left")
    for a in ax:
        a.axhline(0, c="#ccc", lw=0.5)
        a.axvline(0, c="#ccc", lw=0.5)
    fig.text(0.5, 0.005, "58 Spanish occupation pairs. A: blue = masculine, pink = feminine form. C: one point per occupation; above 0 = English word closer to the masculine form.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "dealign_inherits.png", dpi=150)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
