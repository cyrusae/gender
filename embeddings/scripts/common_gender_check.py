"""Exploratory (2026-11-12, PI question): are attribute-word leans pure grammatical gender, or grammatical
plus semantic gender? Not pre-registered; script committed before running.

Lean s(w) = mean_x w·x − mean_y w·y over male / female attribute words (corrected native lists, and the
English lists in the aligned space).
1. Gap comparison (Spanish): mean lean of masculine minus feminine forms for Zhou's occupations, the
   rule-built occupations and the ordinary person nouns (wp_persons_es_frozen_v1), against the same gap
   for strict-tier epicenes (fixed grammatical gender, no sex reference = a pure grammatical-gender
   gap). Excess = group gap − epicene gap (the part grammar can't explain). Bootstrap 10,000 (seed 0)
   over words within group and attribute words (shared across groups in a draw).
2. Common-gender nouns (el/la periodista; one vector for both sexes, so no fixed grammatical gender),
   Spanish and French (frozen zero_ref lists, set "common"): correlation of lean with the Glasgow GEND
   rating of the English concept (human rating, 1 feminine – 7 masculine), where rated. Bootstrap over
   nouns and attribute words.

    uv run --no-project --python 3.12 --with numpy,pandas,scipy,scikit-learn python embeddings/scripts/common_gender_check.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from c1_run import EN_F, EN_M
from scipy.stats import pearsonr
from wordpair_mechanism import glasgow
from zhou_shift import Z, attributes, load_vec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/common_gender_check"
N, SEED = 10_000, 0


def leans(V, AM, AF):
    return (V @ AM.T).mean(1) - (V @ AF.T).mean(1), V @ AM.T, V @ AF.T


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    gl = glasgow()
    gap_rows, cg_rows = [], []
    for lang in ("es", "fr"):
        E, w2i = load_vec(Z / f"data/wiki.multi.{lang}.vec.txt")
        import json
        am, af = attributes(lang, json.load(open(Z / f"{lang.upper()}_definitional_pairs.json")))
        yard = {"native (corrected)": (np.array([E[w2i[w]] for w in am]), np.array([E[w2i[w]] for w in af])),
                "english": (np.array([en[en_w2i[w]] for w in EN_M]), np.array([en[en_w2i[w]] for w in EN_F]))}  # fmt: skip
        ref = pd.read_csv(ROOT / f"data/zero_ref_{lang}_frozen_v1.csv")
        ref = ref[ref.lemma.isin(w2i) & ~ref.lemma.isin(set(am) | set(af))]
        epi = ref[(ref["set"] == "epicene") & (ref.tier == "strict")]
        com = ref[ref["set"] == "common"].copy()
        com["gend"] = com.concept_en.str.lower().map(lambda e: gl.get(e))
        com = com.dropna(subset=["gend"]).reset_index(drop=True)
        groups = {"epicenes (grammar only)": (list(epi[epi.gender == "m"].lemma), list(epi[epi.gender == "f"].lemma))}
        if lang == "es":
            occ = [line.rstrip("\n").split("\t") for line in open(Z / "ES_occupation_words_with_EN_translations")]
            occ = [(m, f) for f, m, _ in occ if m in w2i and f in w2i]
            groups["Zhou occupations"] = ([m for m, _ in occ], [f for _, f in occ])
            pn = pd.read_csv(ROOT / "data/wp_persons_es_frozen_v1.csv", keep_default_na=False, dtype=str)
            pn = pn[pn.masc.isin(w2i) & pn.fem.isin(w2i)]
            for g in ("occupation", "person"):
                s = pn[pn.group == g]
                groups[f"rule-built {g}s"] = (list(s.masc), list(s.fem))
        for y, (AM, AF) in yard.items():
            S = {k: (leans(np.array([E[w2i[w]] for w in m]), AM, AF), leans(np.array([E[w2i[w]] for w in f]), AM, AF))
                 for k, (m, f) in groups.items()}  # fmt: skip

            def gap(k, ai, bi, mi, fi):
                (_, Pm_a, Pm_b), (_, Pf_a, Pf_b) = S[k]
                return (Pm_a[np.ix_(mi, ai)].mean() - Pm_b[np.ix_(mi, bi)].mean()) - (Pf_a[np.ix_(fi, ai)].mean() - Pf_b[np.ix_(fi, bi)].mean())

            na, nb = len(AM), len(AF)
            full = {k: (np.arange(len(S[k][0][0])), np.arange(len(S[k][1][0]))) for k in S}
            pt = {k: gap(k, np.arange(na), np.arange(nb), *full[k]) for k in S}
            B = []
            for _ in range(N):
                ai, bi = rng.integers(0, na, na), rng.integers(0, nb, nb)
                B.append({k: gap(k, ai, bi, rng.integers(0, len(full[k][0]), len(full[k][0])),
                                 rng.integers(0, len(full[k][1]), len(full[k][1]))) for k in S})  # fmt: skip
            B = pd.DataFrame(B)
            ek = "epicenes (grammar only)"
            for k in S:
                lo, hi = np.percentile(B[k], [2.5, 97.5])
                row = {"lang": lang, "yardstick": y, "group": k, "gap_m_minus_f": pt[k], "lo": lo, "hi": hi,
                       "n_m": len(full[k][0]), "n_f": len(full[k][1])}  # fmt: skip
                if k != ek:
                    ex = B[k] - B[ek]
                    row.update({"excess_over_epicene": pt[k] - pt[ek], "excess_lo": np.percentile(ex, 2.5),
                                "excess_hi": np.percentile(ex, 97.5)})  # fmt: skip
                gap_rows.append(row)
            # common-gender nouns
            V = np.array([E[w2i[w]] for w in com.lemma])
            Pa, Pb = V @ AM.T, V @ AF.T
            s = Pa.mean(1) - Pb.mean(1)
            r0 = pearsonr(s, com.gend)[0]
            rb = []
            for _ in range(N):
                ni = rng.integers(0, len(V), len(V))
                ai, bi = rng.integers(0, na, na), rng.integers(0, nb, nb)
                ss = Pa[np.ix_(ni, ai)].mean(1) - Pb[np.ix_(ni, bi)].mean(1)
                g = com.gend.to_numpy()[ni]
                rb.append(np.corrcoef(ss, g)[0, 1] if g.std() > 0 else np.nan)
            lo, hi = np.nanpercentile(rb, [2.5, 97.5])
            cg_rows.append({"lang": lang, "yardstick": y, "n_common_rated": len(V), "pearson_r_lean_vs_GEND": r0,
                            "lo": lo, "hi": hi, "reading": "tracks human rating" if lo > 0 else "not shown"})  # fmt: skip
            print("done", lang, y, flush=True)
    G, C = pd.DataFrame(gap_rows), pd.DataFrame(cg_rows)
    G.to_csv(OUT / "gaps.csv", index=False)
    C.to_csv(OUT / "common_gender.csv", index=False)
    with pd.option_context("display.width", 220):
        print(G.round(4).to_string(index=False))
        print(C.round(3).to_string(index=False))
    print("=== CGC done")


if __name__ == "__main__":
    main()
