"""Person-noun generic masculine (docs/persons-genmasc-prereg.md, ADOPTED 2026-11-11). Committed
before any score. Reuses generic_masculine (axis, stats, load). Spanish; definitional pairs without
the name pair maría/juan. Pairs need all four forms in the first 50k words
(data/wp_persons_es_plurals_v0.csv, frozen at adoption). Shared bootstrap draws (definitional pairs,
epicene references, common nouns) for the persons − occupations difference; pairs within group.

    uv run --no-project --python 3.12 --with numpy,pandas,scikit-learn python embeddings/scripts/persons_genmasc.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from generic_masculine import axis, load, stats
from sklearn.svm import LinearSVC
from zhou_shift import Z

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/persons_genmasc"
N_BOOT, SEED, FLOOR = 10_000, 0, 25
FLAGS = ["fem_also_noun", "masc_also_noun", "language_name", "masc_identical_en", "place_name"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    U, w2i = load(Z / "data/wiki.multi.es.vec.txt")
    pl = pd.read_csv(ROOT / "data/wp_persons_es_plurals_v0.csv", keep_default_na=False, dtype=str)
    fz = pd.read_csv(ROOT / "data/wp_persons_es_frozen_v1.csv", keep_default_na=False, dtype=str)
    pl = pl.merge(fz[["masc"] + FLAGS], on="masc", validate="one_to_one")
    cols = ["masc", "fem", "masc_pl", "fem_pl"]
    pl = pl[pl[cols].apply(lambda r: all(w in w2i for w in r), axis=1)]
    defs = [p for p in json.load(open(Z / "ES_definitional_pairs.json")) if p != ["maría", "juan"] and all(w in w2i for w in p)]
    Ddef = np.array([[U[w2i[f]], U[w2i[m]]] for f, m in defs])
    c1 = pd.read_csv(ROOT / "data/c1_nouns_es_v1.csv")
    c1 = c1[c1["in_50k"] & c1["lemma"].isin(w2i)]
    g = LinearSVC().fit(U[c1["lemma"].map(w2i)], (c1["gender"] == "m").astype(int)).coef_.reshape(-1)
    g /= np.linalg.norm(g)
    sign_pair = (U[w2i[defs[0][0]]], U[w2i[defs[0][1]]])
    ref = pd.read_csv(ROOT / "data/zero_ref_es_frozen_v1.csv")
    ref = ref[ref["lemma"].isin(w2i)]
    epi = ref[(ref["set"] == "epicene") & (ref["tier"] == "strict")]
    Rm, Rf = U[epi.loc[epi.gender == "m", "lemma"].map(w2i)], U[epi.loc[epi.gender == "f", "lemma"].map(w2i)]
    C = U[ref.loc[ref["set"] == "common", "lemma"].map(w2i)]
    rows, info = [], {}
    for sel, pp in (("all", pl), ("no flagged nouns", pl[~pl[FLAGS].eq("True").any(axis=1)])):
        P = {grp: np.stack([U[pp[pp.group == grp][c].map(w2i)] for c in cols], 1) for grp in ("person", "occupation")}
        n = {k: len(v) for k, v in P.items()}
        d0 = axis(Ddef, g, sign_pair)
        pt = {k: stats(v, Rm, Rf, C, d0) for k, v in P.items()}
        rng = np.random.default_rng(SEED)
        B = {k: [] for k in P}
        for _ in range(N_BOOT):
            d = axis(Ddef[rng.integers(0, len(Ddef), len(Ddef))], g, sign_pair)
            rm, rf = Rm[rng.integers(0, len(Rm), len(Rm))], Rf[rng.integers(0, len(Rf), len(Rf))]
            cc = C[rng.integers(0, len(C), len(C))]
            for k, v in P.items():
                B[k].append(stats(v[rng.integers(0, len(v), len(v))], rm, rf, cc, d))
        B = {k: pd.DataFrame(v) for k, v in B.items()}
        for k in P:
            for m in pt[k]:
                lo, hi = np.nanpercentile(B[k][m], [2.5, 97.5])
                rows.append({"set": sel, "group": k, "measure": m, "value": pt[k][m], "lo": lo, "hi": hi, "n_pairs": n[k]})
            dlo, dhi = np.nanpercentile(B[k]["delta_rho"], [2.5, 97.5])
            up = np.nanpercentile(B[k]["G"] - B[k]["margin"], 97.5)
            low = np.nanpercentile(B[k]["G"] + B[k]["margin"], 2.5)
            info[f"{sel}|{k}"] = {"n_pairs": n[k], "floor_met": n[k] >= FLOOR,
                                  "G1_generic_masculine": "supported" if dlo > 0 else "against" if dhi < 0 else "inconclusive",
                                  "G2_masc_plural_at_neutral": "at neutral" if (up < 0 and low > 0) else "not shown"}  # fmt: skip
        diff = B["person"]["delta_rho"] - B["occupation"]["delta_rho"]
        lo, hi = np.nanpercentile(diff, [2.5, 97.5])
        rows.append({"set": sel, "group": "person-occupation", "measure": "delta_rho",
                     "value": pt["person"]["delta_rho"] - pt["occupation"]["delta_rho"], "lo": lo, "hi": hi,
                     "n_pairs": f"{n['person']}/{n['occupation']}"})  # fmt: skip
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results.csv", index=False)
    json.dump(info, open(OUT / "readings.json", "w"), indent=1)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(res[res.measure.isin(["Ms", "Mp", "rho_m", "rho_f", "delta_rho", "G", "margin"])].round(4).to_string(index=False))
    print(json.dumps(info, indent=1))
    print("=== PGM done")


if __name__ == "__main__":
    main()
