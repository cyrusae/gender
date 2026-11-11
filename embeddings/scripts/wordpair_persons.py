"""Non-occupation person-noun baseline (docs/wordpair-persons-prereg.md, ADOPTED 2026-11-11).
Nouns frozen in data/wp_persons_es_frozen_v1.csv; cues in data/wp_cues_frozen_v1.csv (Spanish primary
gendered pairs). Committed before any score.

Query and scoring as wordpair_mechanism.rr_table (original Spanish space; reciprocal rank, 0 outside
the top 999). Per noun: intrusion = mean over feminine cues of (RR_masc − RR_fem).
P1: mean intrusion over person nouns. P2: person − occupation. Identity: descriptive.
Bootstrap 10,000 (seed 0): nouns resampled within group, cue pairs resampled once per draw (shared
by both groups). P3: per-noun intrusion ~ z(log rank_f/rank_m) + z(Glasgow GEND) + group indicators,
over nouns with a Glasgow rating; bootstrap over nouns and cues.
Sensitivity: P1–P2 without nouns carrying any of fem_also_noun, masc_also_noun, language_name,
masc_identical_en, place_name.

    uv run --no-project --python 3.12 --with numpy,pandas,scikit-learn python embeddings/scripts/wordpair_persons.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from wordpair_mechanism import glasgow, rr_table
from zhou_shift import Z, load_vec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/wordpair_persons"
N_BOOT, SEED = 10_000, 0
FLAGS = ["fem_also_noun", "masc_also_noun", "language_name", "masc_identical_en", "place_name"]


def zs(x):
    return (x - x.mean()) / x.std() if x.std() > 0 else x * 0


def p12(I, grp, rng, tag):
    """I [cue, noun] intrusion per cue × noun."""
    groups = {g: np.flatnonzero(grp == g) for g in ("person", "occupation", "identity")}
    pt = {g: I[:, ix].mean() for g, ix in groups.items()}
    pt["person-occupation"] = pt["person"] - pt["occupation"]
    B = []
    nc = I.shape[0]
    for _ in range(N_BOOT):
        ci = rng.integers(0, nc, nc)
        b = {g: I[np.ix_(ci, rng.choice(ix, len(ix)))].mean() for g, ix in groups.items()}
        b["person-occupation"] = b["person"] - b["occupation"]
        B.append(b)
    B = pd.DataFrame(B)
    rows = []
    for k, v in pt.items():
        lo, hi = np.percentile(B[k], [2.5, 97.5])
        if k == "person-occupation":
            verdict = "occupations carry extra intrusion" if hi < 0 else ("persons higher" if lo > 0 else "no difference shown")
        else:
            verdict = "present" if lo > 0 else ("reversed" if hi < 0 else "inconclusive")
        rows.append({"set": tag, "measure": k, "value": v, "lo": lo, "hi": hi, "verdict": verdict,
                     "n_nouns": len(groups.get(k, groups["person"])) if k in groups else
                     f"{len(groups['person'])}/{len(groups['occupation'])}", "n_cues": nc})  # fmt: skip
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nouns = pd.read_csv(ROOT / "data/wp_persons_es_frozen_v1.csv", keep_default_na=False, dtype=str)
    cues = pd.read_csv(ROOT / "data/wp_cues_frozen_v1.csv", keep_default_na=False, dtype=str)
    G = cues[(cues.lang == "es") & (cues.subset == "primary") & (cues.type == "gendered")].reset_index(drop=True)
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    T, w2i = load_vec(Z / "data/wiki.multi.es.vec.txt")
    i2w = {i: w for w, i in w2i.items()}
    ok = nouns.en.isin(en_w2i) & nouns.masc.isin(w2i) & nouns.fem.isin(w2i)
    nouns = nouns[ok].reset_index(drop=True)
    occ = [(f, m, e) for f, m, e in zip(nouns.fem, nouns.masc, nouns.en)]
    fc = [(e, x, "f") for e, x in zip(G.en, G.fem)]
    R = rr_table(en, T, en_w2i, w2i, i2w, fc, occ)  # [cue, noun, 2]
    I = R[..., 0] - R[..., 1]
    grp = nouns.group.to_numpy()
    rng = np.random.default_rng(SEED)
    rows = p12(I, grp, rng, "all")
    clean = ~nouns[FLAGS].eq("True").any(axis=1).to_numpy()
    rows += p12(I[:, clean], grp[clean], rng, "no flagged nouns")
    res = pd.DataFrame(rows)
    per = nouns.assign(intrusion=I.mean(0), rr_m=R[..., 0].mean(0), rr_f=R[..., 1].mean(0))
    per.to_csv(OUT / "per_noun.csv", index=False)
    # P3
    gl = glasgow()
    d = per.assign(freq=np.log(per.rank_f.astype(float) + 1) - np.log(per.rank_m.astype(float) + 1),
                   gend=[gl.get(e, np.nan) for e in per.en])  # fmt: skip
    has = d.gend.notna().to_numpy()
    d, Id = d[has].reset_index(drop=True), I[:, has]

    def fit(ci, ni):
        y = Id[np.ix_(ci, ni)].mean(0)
        X = np.column_stack([np.ones(len(ni)), zs(d.freq.to_numpy()[ni]), zs(d.gend.to_numpy()[ni]),
                             (d.group.to_numpy()[ni] == "occupation").astype(float),
                             (d.group.to_numpy()[ni] == "identity").astype(float)])  # fmt: skip
        return np.linalg.lstsq(X, zs(y), rcond=None)[0][1:3]

    nc, nn = Id.shape
    b0 = fit(np.arange(nc), np.arange(nn))
    bs = np.array([fit(rng.integers(0, nc, nc), rng.integers(0, nn, nn)) for _ in range(N_BOOT)])
    q3 = pd.DataFrame([{"term": t, "beta_std": b0[k], "lo": np.percentile(bs[:, k], 2.5), "hi": np.percentile(bs[:, k], 97.5),
                        "n_nouns": nn} for k, t in enumerate(["log_freq_ratio", "glasgow_gend"])])  # fmt: skip
    q3["verdict"] = np.where((q3.lo > 0) | (q3.hi < 0), "tracks", "not shown")
    res.to_csv(OUT / "p1_p2.csv", index=False)
    q3.to_csv(OUT / "p3.csv", index=False)
    with pd.option_context("display.width", 200):
        print(res.round(4).to_string(index=False))
        print(q3.round(3).to_string(index=False))
        print(per.groupby("group")[["intrusion", "rr_m", "rr_f"]].mean().round(3))
    print("=== WPPERSONS done")


if __name__ == "__main__":
    main()
