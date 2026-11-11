"""Generic masculine and Zhou's skew: singular vs plural shrink (docs/generic-masculine-prereg.md,
ADOPTED 2026-11-10; pairs frozen in data/gm_pairs_frozen_v1.csv). Committed before any score.

Axis d_s: PCA (true pair midpoints) of the language's 10 definitional pairs, with the grammatical
direction g (LinearSVC on unit vectors of the C1 Wiktionary nouns, first 50k words) removed; oriented
so the masculine word of the first pair is positive. Words are unit vectors from the first 200k MUSE
words. Zero point z = ½(mean m + mean f) over strict epicenes (data/zero_ref_{lang}_frozen_v1.csv).
Distances from z: Ms, Fs (singular), Mp, Fp (plural), per pair.
  ρ_m = mean(Mp)/mean(Ms), ρ_f = mean(Fp)/mean(Fs), Δρ = ρ_f − ρ_m           (primary)
  G = mean(Mp) − mean(common-gender reference nouns), margin = ½·mean(Ms)    (secondary)
Bootstrap 10,000 (seed 0) over pairs, epicene words within gender, common nouns, and the 10
definitional pairs (the PCA axis is re-estimated in every draw; g fixed). FR confirmatory, ES
replication of the exploratory observation. Floor: 15 usable pairs.

    uv run --no-project --python 3.12 --with numpy,pandas,scikit-learn python embeddings/scripts/generic_masculine.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from zhou_shift import Z

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/generic_masculine"
N_BOOT, SEED, FLOOR = 10_000, 0, 15


def load(path, nmax=200_000):
    vecs, w2i = [], {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        next(f)
        for i, line in enumerate(f):
            if i == nmax:
                break
            w, v = line.rstrip().split(" ", 1)
            if w not in w2i:
                w2i[w] = len(vecs)
                vecs.append(np.fromstring(v, sep=" "))
    U = np.vstack(vecs)
    return U / np.linalg.norm(U, axis=1, keepdims=True), w2i


def first_pc(D):
    """D: [pairs, 2, dim] unit vectors; top PC of the two half-differences (true midpoint)."""
    c = D.mean(1, keepdims=True)
    X = (D - c).reshape(-1, D.shape[-1])
    X = X - X.mean(0)
    return np.linalg.svd(X, full_matrices=False)[2][0]


def axis(Ddef, g, sign_pair):
    d = first_pc(Ddef)
    d = d - (d @ g) * g
    d /= np.linalg.norm(d)
    return d if (sign_pair[1] - sign_pair[0]) @ d > 0 else -d


def stats(P, Rm, Rf, C, d):
    """P [n, 4, dim] (Ms, Fs, Mp, Fp); epicene refs Rm, Rf; common C. Returns the measures."""
    z = 0.5 * ((Rm @ d).mean() + (Rf @ d).mean())
    x = P @ d - z
    Ms, Fs, Mp, Fp = (x[:, k].mean() for k in range(4))
    rm, rf = Mp / Ms, Fp / Fs
    G = Mp - ((C @ d) - z).mean()
    return {"Ms": Ms, "Fs": Fs, "Mp": Mp, "Fp": Fp, "rho_m": rm, "rho_f": rf, "delta_rho": rf - rm, "G": G,
            "margin": 0.5 * Ms, "G_over_margin_lo": G - 0.5 * Ms, "G_over_margin_hi": G + 0.5 * Ms}  # fmt: skip


def run(lang, pairs, U, w2i, tag):
    defs = json.load(open(Z / f"{lang.upper()}_definitional_pairs.json"))
    defs = [p for p in defs if all(w in w2i for w in p)]
    Ddef = np.array([[U[w2i[f]], U[w2i[m]]] for f, m in defs])  # fem, masc
    c1 = pd.read_csv(ROOT / f"data/c1_nouns_{lang}_v1.csv")
    c1 = c1[c1["in_50k"] & c1["lemma"].isin(w2i)]
    g = LinearSVC().fit(U[c1["lemma"].map(w2i)], (c1["gender"] == "m").astype(int)).coef_.reshape(-1)
    g /= np.linalg.norm(g)
    sign_pair = (U[w2i[defs[0][0]]], U[w2i[defs[0][1]]])
    ref = pd.read_csv(ROOT / f"data/zero_ref_{lang}_frozen_v1.csv")
    ref = ref[ref["lemma"].isin(w2i)]
    epi = ref[(ref["set"] == "epicene") & (ref["tier"] == "strict")]
    Rm, Rf = U[epi.loc[epi.gender == "m", "lemma"].map(w2i)], U[epi.loc[epi.gender == "f", "lemma"].map(w2i)]
    C = U[ref.loc[ref["set"] == "common", "lemma"].map(w2i)]
    cols = ["masc", "fem", "masc_pl", "fem_pl"]
    ok = pairs[cols].apply(lambda r: all(w in w2i for w in r), axis=1)
    use = pairs[ok]
    info = {"lang": lang, "list": tag, "n_listed": len(pairs), "n_usable": len(use), "floor_met": len(use) >= FLOOR,
            "dropped_missing_form": [f"{a}/{b}" for a, b in pairs.loc[~ok, ["masc", "fem"]].values]}  # fmt: skip
    if not info["floor_met"]:
        return info, []
    P = np.stack([U[use[c].map(w2i)] for c in cols], 1)
    pt = stats(P, Rm, Rf, C, axis(Ddef, g, sign_pair))
    rng = np.random.default_rng(SEED)
    B = []
    for _ in range(N_BOOT):
        di = rng.integers(0, len(Ddef), len(Ddef))
        d = axis(Ddef[di], g, sign_pair)
        B.append(stats(P[rng.integers(0, len(P), len(P))], Rm[rng.integers(0, len(Rm), len(Rm))],
                       Rf[rng.integers(0, len(Rf), len(Rf))], C[rng.integers(0, len(C), len(C))], d))  # fmt: skip
    B = pd.DataFrame(B)
    rows = []
    for k in pt:
        lo, hi = np.nanpercentile(B[k], [2.5, 97.5])
        rows.append({"lang": lang, "list": tag, "measure": k, "value": pt[k], "lo": lo, "hi": hi})
    # readings
    dlo, dhi = np.nanpercentile(B["delta_rho"], [2.5, 97.5])
    info["reading_1_generic_masculine"] = ("supported" if dlo > 0 else "against" if dhi < 0 else "inconclusive")
    # equivalence: G within ±margin in every draw's own scale → use CI of G − margin and G + margin
    upper = np.nanpercentile(B["G"] - B["margin"], 97.5)
    lower = np.nanpercentile(B["G"] + B["margin"], 2.5)
    info["reading_2_masc_plural_at_neutral"] = "at neutral" if (upper < 0 and lower > 0) else "not shown"
    info["G_minus_margin_97.5"], info["G_plus_margin_2.5"] = float(upper), float(lower)
    return info, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fr = pd.read_csv(ROOT / "data/gm_pairs_frozen_v1.csv")
    infos, rows = [], []
    for lang in ("fr", "es"):
        U, w2i = load(Z / f"data/wiki.multi.{lang}.vec.txt")
        p = fr[fr.lang == lang]
        for tag, sub in (("primary", p), ("no_homographs (descriptive)", p[p.H == 0])):
            i, r = run(lang, sub, U, w2i, tag)
            infos.append(i)
            rows += r
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
    json.dump(infos, open(OUT / "readings.json", "w"), indent=1, ensure_ascii=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(json.dumps(infos, indent=1, ensure_ascii=False))
    print("=== GENMASC done")


if __name__ == "__main__":
    main()
