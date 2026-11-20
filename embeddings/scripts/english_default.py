"""Is the masculine default in Spanish and French inherited from English? (docs/english-default-prereg.md,
ADOPTED 2026-11-20). Committed before any score.

Lean of an English word w: s(w) = mean cos(w, male attributes) − mean cos(w, female attributes), Caliskan
et al. (2022)'s sixteen attribute words, MUSE English, first 200k words (unit vectors, so a mean of cosines
is the dot product with the mean attribute vector). Fair lean: s(w) − z, z = mean s over the zero-point
anchors (data/en_zero_frozen_v1.csv).

Q1  share of words with s > 0 and with s − z > 0 among the 100 / 1,000 / 10,000 / 100,000 most frequent
    words; bootstrap (10,000, seed 0) over attribute words (within gender) and anchors. (SC-WEAT's effect
    size d has the sign of s, so its share at Caliskan's zero equals the s share.)
Q2  per noun, plain-translation gap g = RR(masc) − RR(fem) for the query "English gloss" (cosine over the
    first 50k target words, misses 0, as wordpair_persons_extra.py); per language,
    z(g) ~ z(fair English lean) + z(log rank_f/rank_m) + occupation; bootstrap over nouns.
Q3  per noun, feminine-cue intrusion (results/wordpair_persons{,_fr}/per_noun.csv);
    z(intrusion) ~ occupation + z(fair English lean) + z(log rank ratio); bootstrap over nouns. Plus mean
    fair English lean of person vs occupation glosses. (Q2/Q3 coefficients don't depend on z: it shifts
    every noun's lean by the same amount.)
Sensitivity: epicene-gloss anchors; glosses-in anchors (primary + adult/human/friend/neighbour, with the
nouns they gloss dropped from Q2/Q3); Q2/Q3 without flagged nouns.

    uv run --no-project --python 3.12 --with numpy,pandas,scipy python embeddings/scripts/english_default.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from zhou_shift import Z, load_vec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/english_default"
N_BOOT, SEED, K = 10_000, 0, 1000
FEM = ["female", "she", "her", "hers", "woman", "girl", "daughter", "sister"]
MAS = ["male", "he", "him", "his", "man", "boy", "son", "brother"]
NS = [100, 1_000, 10_000, 100_000]
FLAGS = ["fem_also_noun", "masc_also_noun", "language_name", "masc_identical_en", "place_name"]
GLOSSES_IN = {"adult", "human", "friend", "neighbour"}


def zs(x):
    return (x - x.mean()) / x.std() if x.std() > 0 else x * 0


def q1(C, anchors_idx, rng, tag):
    """C: cosines of the top-100k words (rows) with the 16 attributes (cols: 8 male, then 8 female)."""
    s = C[:, :8].mean(1) - C[:, 8:].mean(1)
    z = s[anchors_idx].mean()
    rows = []
    for n in NS:
        rows.append({"anchors": tag, "N": n, "share_male_caliskan_zero": (s[:n] > 0).mean(),
                     "share_male_fair_zero": (s[:n] > z).mean(), "z": z})  # fmt: skip
    bm = rng.integers(0, 8, (N_BOOT, 8))
    bf = rng.integers(0, 8, (N_BOOT, 8)) + 8
    ba = rng.integers(0, len(anchors_idx), (N_BOOT, len(anchors_idx)))
    W = np.zeros((N_BOOT, 16))
    for b in range(N_BOOT):
        np.add.at(W[b], bm[b], 1 / 8)
        np.add.at(W[b], bf[b], -1 / 8)
    sh = {n: np.empty(N_BOOT) for n in NS}
    sh0 = {n: np.empty(N_BOOT) for n in NS}
    for i in range(0, N_BOOT, 500):
        S = C @ W[i:i + 500].T  # words × draws
        zb = np.array([S[anchors_idx[ba[i + j]], j].mean() for j in range(S.shape[1])])
        for n in NS:
            sh[n][i:i + 500] = (S[:n] > zb).mean(0)
            sh0[n][i:i + 500] = (S[:n] > 0).mean(0)
    for r in rows:
        n = r["N"]
        r["fair_lo"], r["fair_hi"] = np.percentile(sh[n], [2.5, 97.5])
        r["caliskan_lo"], r["caliskan_hi"] = np.percentile(sh0[n], [2.5, 97.5])
        r["verdict_fair"] = ("robust to the zero point" if r["fair_lo"] > 0.5 else
                             "reversed" if r["fair_hi"] < 0.5 else "explained by the zero point")  # fmt: skip
    rho = spearmanr(s - z, np.log(np.arange(1, len(s) + 1)))[0]
    return rows, s, z, rho


def plain_gap(en, en_w2i, T, w2i, i2w, nouns):
    Tn = T / np.linalg.norm(T, axis=1, keepdims=True)
    g = np.zeros(len(nouns))
    for k, (m, f, e) in enumerate(zip(nouns.masc, nouns.fem, nouns.en)):
        q = en[en_w2i[e]]
        sc = Tn @ (q / np.linalg.norm(q))
        top = [i2w[i] for i in np.argsort(-sc)[:K]][:999]
        pos = {w: r + 1 for r, w in enumerate(top)}
        g[k] = (1 / pos[m] if m in pos else 0) - (1 / pos[f] if f in pos else 0)
    return g


def reg(d, y, terms, rng):
    def fit(ix):
        x = d.iloc[ix]
        X = np.column_stack([np.ones(len(ix))] + [zs(x[t].to_numpy(float)) if t != "occupation" else x[t].to_numpy(float) for t in terms])
        return np.linalg.lstsq(X, zs(x[y].to_numpy(float)), rcond=None)[0][1:]

    n = len(d)
    b0 = fit(np.arange(n))
    B = np.array([fit(rng.integers(0, n, n)) for _ in range(N_BOOT)])
    return [{"term": t, "beta_std": b0[k], "lo": np.percentile(B[:, k], 2.5), "hi": np.percentile(B[:, k], 97.5), "n": n}
            for k, t in enumerate(terms)]  # fmt: skip


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    assert all(w in en_w2i for w in FEM + MAS)
    A = np.array([en[en_w2i[w]] for w in MAS + FEM])
    C = en[:100_000] @ A.T
    Call = en @ A.T
    sall = Call[:, :8].mean(1) - Call[:, 8:].mean(1)
    anc = pd.read_csv(ROOT / "data/en_zero_frozen_v1.csv")
    sets = {"primary": list(anc[anc["set"] == "primary"].word),
            "epicene_gloss": list(anc[anc["set"] == "epicene_gloss"].word),
            "glosses_in": list(anc[anc["set"].isin(["primary", "glosses_in_extra"])].word)}  # fmt: skip
    rows1, info = [], {}
    for tag, words in sets.items():
        idx = np.array([en_w2i[w] for w in words if w in en_w2i and en_w2i[w] < 100_000])
        r, s, z, rho = q1(C, idx, rng, tag)
        rows1 += r
        info[tag] = {"n_anchors": len(idx), "z": z, "anchor_rank_median": float(np.median(idx)),
                     "spearman_fair_lean_vs_log_rank": rho}  # fmt: skip
    q1df = pd.DataFrame(rows1)
    q1df.to_csv(OUT / "q1_shares.csv", index=False)
    pd.DataFrame(info).T.to_csv(OUT / "q1_anchor_info.csv")
    zp = info["primary"]["z"]
    rows23, lean_rows, per = [], [], []
    for lang in ("es", "fr"):
        T, w2i = load_vec(Z / f"data/wiki.multi.{lang}.vec.txt")
        i2w = {i: w for w, i in w2i.items()}
        nouns = pd.read_csv(ROOT / f"data/wp_persons_{lang}_frozen_v1.csv", keep_default_na=False, dtype=str)
        nouns = nouns[nouns.group.isin(["person", "occupation"]) & nouns.en.isin(en_w2i)
                      & nouns.masc.isin(w2i) & nouns.fem.isin(w2i)].reset_index(drop=True)  # fmt: skip
        pn = pd.read_csv(ROOT / f"results/wordpair_persons{'' if lang == 'es' else '_fr'}/per_noun.csv", keep_default_na=False)
        nouns = nouns.merge(pn[["masc", "intrusion"]], on="masc", how="left", validate="one_to_one")
        nouns["gap_plain"] = plain_gap(en, en_w2i, T, w2i, i2w, nouns)
        nouns["en_lean_fair"] = [sall[en_w2i[e]] - zp for e in nouns.en]
        nouns["log_rank_ratio"] = np.log(nouns.rank_f.astype(float) + 1) - np.log(nouns.rank_m.astype(float) + 1)
        nouns["occupation"] = (nouns.group == "occupation").astype(float)
        nouns["lang"] = lang
        per.append(nouns)
        clean = ~nouns[FLAGS].eq("True").any(axis=1)
        for sel, d in (("all", nouns), ("no flagged nouns", nouns[clean]),
                       ("glosses_in (nouns dropped)", nouns[~nouns.en.isin(GLOSSES_IN)])):  # fmt: skip
            d = d.reset_index(drop=True)
            for r in reg(d, "gap_plain", ["en_lean_fair", "log_rank_ratio", "occupation"], rng):
                rows23.append({"lang": lang, "set": sel, "question": "Q2 plain-translation gap", **r})
            dd = d.dropna(subset=["intrusion"]).reset_index(drop=True)
            for r in reg(dd, "intrusion", ["occupation", "en_lean_fair", "log_rank_ratio"], rng):
                rows23.append({"lang": lang, "set": sel, "question": "Q3 feminine-cue intrusion", **r})
            lp, lo_ = d[d.group == "person"].en_lean_fair.to_numpy(), d[d.group == "occupation"].en_lean_fair.to_numpy()
            B = [rng.choice(lo_, len(lo_)).mean() - rng.choice(lp, len(lp)).mean() for _ in range(N_BOOT)]
            lean_rows.append({"lang": lang, "set": sel, "mean_lean_person": lp.mean(), "mean_lean_occupation": lo_.mean(),
                              "occ_minus_person": lo_.mean() - lp.mean(), "lo": np.percentile(B, 2.5), "hi": np.percentile(B, 97.5),
                              "n_person": len(lp), "n_occ": len(lo_)})  # fmt: skip
    res = pd.DataFrame(rows23)

    def verdict(r):
        ok = r.lo > 0
        if r.question.startswith("Q2") and r.term == "en_lean_fair":
            return "English lean contributes" if ok else ("reversed" if r.hi < 0 else "not shown")
        if r.question.startswith("Q3") and r.term == "occupation":
            return "occupation-specific beyond English lean" if ok else "not shown"
        return ""

    res["verdict"] = res.apply(verdict, axis=1)
    res.to_csv(OUT / "q2_q3.csv", index=False)
    pd.DataFrame(lean_rows).to_csv(OUT / "lean_person_vs_occupation.csv", index=False)
    pd.concat(per).to_csv(OUT / "per_noun.csv", index=False)
    with pd.option_context("display.width", 220):
        print(q1df.round(3).to_string(index=False))
        print(pd.DataFrame(info).T.round(4))
        print(res.round(3).to_string(index=False))
        print(pd.DataFrame(lean_rows).round(4).to_string(index=False))
    print("=== ENDEFAULT done")


if __name__ == "__main__":
    main()
