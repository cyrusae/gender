"""Zhou et al. 2019 Table 2 "word pair translation by analogy", three ways (exploratory, PI 2026-11-10).

Committed before any score. Query (cell 347): q = EN[occupation] − EN[adj_en] + XX[adj_xx]; rank the
first 50k target-language words by cosine to q, skipping the cue word; keep the top 1,000.

(a) reconstruction, bugs kept (cells 345, 351, 352, 274): masculine/invariant cues only; a form
    outside the top 999 gets no rank (not 0), so F and M MRR average different query sets; ranks are
    recorded before the `continue` that drops one-form queries from ASD; FR Hybrid Ori offsets read
    the feminine form from the unaligned space (Z16). Compared with Table 2.
(b) fixed, same cues: misses score reciprocal rank 0; ASD and MRR over the same queries; FR offsets
    from one space. Split by cue type: masculine cue vs invariant cue.
(c) fixed, with feminine cues added (ES trabajadora, buena, mala; FR travailleuse, bonne,
    mauvaise, heureuse, meilleure; FR heureux/meilleur are masculine forms, not invariant).

Read-out for (b)/(c), set here before running:
  agreement  = MRR of the form the cue requires (masculine cue → masc. form; feminine → fem.),
               reported per cue gender; "agreement gap" = MRR(masc cue, masc form) −
               MRR(fem cue, fem form).
  bias       = on invariant cues only, M MRR − F MRR (no grammatically correct answer, so a gap is a
               preference), plus ASD.
Spaces (approximations, see zhou_hybrid.py): Original = MUSE as released; Shift Ori = our shift() on
the target language with the original English; Hybrid Ori = target aligned (Procrustes) onto hard-
debiased English, then shifted. Directions as c1_run.directions (their recipe).

    uv run --no-project --python 3.12 --with numpy,pandas,scikit-learn python embeddings/scripts/zhou_wordpair.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from c1_run import directions
from zhou_hybrid import DICT, hard_debias, procrustes
from zhou_shift import Z, load_vec, shift

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/zhou_wordpair"
K = 1000
CUES = {  # (english, form, cue gender)
    "es": [("hardworking", "trabajador", "m"), ("good", "bueno", "m"), ("kind", "amable", "inv"),
           ("happy", "feliz", "inv"), ("reliable", "fiable", "inv"), ("bad", "malo", "m"),
           ("better", "mejor", "inv")],
    "fr": [("hardworking", "travailleur", "m"), ("good", "bon", "m"), ("kind", "aimable", "inv"),
           ("happy", "heureux", "m"), ("reliable", "fiable", "inv"), ("bad", "mauvais", "m"),
           ("better", "meilleur", "m")],
}  # fmt: skip
FEM = {"es": [("hardworking", "trabajadora", "f"), ("good", "buena", "f"), ("bad", "mala", "f")],
       "fr": [("hardworking", "travailleuse", "f"), ("good", "bonne", "f"), ("happy", "heureuse", "f"),
              ("bad", "mauvaise", "f"), ("better", "meilleure", "f")]}  # fmt: skip
TABLE2 = {("es", "original"): (0.1082, 0.2073, 0.6940), ("es", "shift_ori"): (0.0961, 0.2507, 0.6766),
          ("es", "hybrid_ori"): (0.0755, 0.3450, 0.6398), ("fr", "original"): (0.1208, 0.1663, 0.6549),
          ("fr", "shift_ori"): (0.1048, 0.2101, 0.6313), ("fr", "hybrid_ori"): (0.0735, 0.3128, 0.6393)}  # fmt: skip


def spaces(lang, en, en_w2i):
    X, w2i = load_vec(Z / f"data/wiki.multi.{lang}.vec.txt")
    occ = [line.rstrip("\n").split("\t") for line in open(Z / f"{lang.upper()}_occupation_words_with_EN_translations")]
    pairs = [(m, f) for f, m, _ in occ]
    ds, _ = directions(X, w2i, lang)
    out = {"original": (en, X, X), "shift_ori": (en, shift(X, w2i, pairs, ds), shift(X, w2i, pairs, ds))}
    en_db, _ = hard_debias(en, en_w2i)
    seen, src, tgt = set(), [], []
    for line in open(Path(str(DICT).replace("es-en", f"{lang}-en")), encoding="utf-8"):
        a, b = line.split()
        if a in w2i and b in en_w2i and a not in seen:
            seen.add(a)
            src.append(w2i[a])
            tgt.append(en_w2i[b])
    o = np.argsort(src)[:5000]
    A = X @ procrustes(X[np.array(src)[o]], en_db[np.array(tgt)[o]]).T
    dsa, _ = directions(A, w2i, lang)
    fixed = shift(A, w2i, pairs, dsa)
    buggy = A.copy()
    for m, f in pairs:  # cell 274: FR feminine offset read from the unaligned space
        fem = (X if lang == "fr" else A)[w2i[f]] @ dsa
        off = -0.5 * (A[w2i[m]] @ dsa + fem)
        buggy[w2i[m]] += off * dsa
        buggy[w2i[f]] += off * dsa
    out["hybrid_ori"] = (en_db, buggy, fixed)
    return out, w2i, occ


def ranks(EN, T, en_w2i, w2i, i2w, Tn, cue_en, cue_xx, occ_en, forms):
    q = EN[en_w2i[occ_en]] - EN[en_w2i[cue_en]] + T[w2i[cue_xx]]
    s = Tn @ (q / np.linalg.norm(q))
    top = [i for i in np.argsort(-s)[:K] if i2w[i] != cue_xx]
    pos = {i2w[i]: (r + 1, s[i]) for r, i in enumerate(top[:999])}
    return {k: pos.get(w, (None, None)) for k, w in forms.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    rows = []
    for lang in ("es", "fr"):
        sp, w2i, occ = spaces(lang, en, en_w2i)
        i2w = {i: w for w, i in w2i.items()}
        for name, (EN, Tbug, Tfix) in sp.items():
            for version, T in (("bug", Tbug), ("fix", Tfix)):
                Tn = T / np.linalg.norm(T, axis=1, keepdims=True)
                for cue_en, cue_xx, cg in CUES[lang] + FEM[lang]:
                    if cue_xx not in w2i or cue_en not in en_w2i:
                        continue
                    for f, m, e in occ:
                        if e not in en_w2i:
                            continue
                        r = ranks(EN, T, en_w2i, w2i, i2w, Tn, cue_en, cue_xx, e, {"m": m, "f": f})
                        rows.append({"lang": lang, "space": name, "version": version, "cue": cue_xx, "cue_gender": cg,
                                     "occ": e, "rank_m": r["m"][0], "rank_f": r["f"][0], "sim_m": r["m"][1], "sim_f": r["f"][1]})  # fmt: skip
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "queries.csv", index=False)

    # (a) reconstruction: their cues, bug version of space, their MRR / ASD rules
    a = []
    for (lang, sp), g in d[(d.version == "bug") & (d.cue_gender != "f")].groupby(["lang", "space"]):
        both = g.dropna(subset=["rank_m", "rank_f"])
        a.append({"lang": lang, "space": sp, "ASD": (both.sim_m - both.sim_f).abs().mean(),
                  "F_MRR": (1 / g.rank_f.dropna()).mean(), "M_MRR": (1 / g.rank_m.dropna()).mean(),
                  "n_f_found": g.rank_f.notna().sum(), "n_m_found": g.rank_m.notna().sum(), "n_queries": len(g),
                  "paper_ASD": TABLE2[(lang, sp)][0], "paper_F_MRR": TABLE2[(lang, sp)][1], "paper_M_MRR": TABLE2[(lang, sp)][2]})  # fmt: skip
    a = pd.DataFrame(a)
    a["MRR_diff"], a["paper_MRR_diff"] = a.M_MRR - a.F_MRR, a.paper_M_MRR - a.paper_F_MRR

    # (b)/(c): fixed rules, by cue gender
    f = d[d.version == "fix"].copy()
    f["rr_m"], f["rr_f"] = (1 / f.rank_m).fillna(0), (1 / f.rank_f).fillna(0)
    bc = (f.groupby(["lang", "space", "cue_gender"])
            .agg(n=("occ", "size"), M_MRR=("rr_m", "mean"), F_MRR=("rr_f", "mean"),
                 ASD=("sim_m", lambda s: (s - f.loc[s.index, "sim_f"]).abs().mean()))
            .reset_index())  # fmt: skip
    b_pooled = (f[f.cue_gender != "f"].groupby(["lang", "space"])
                  .agg(M_MRR=("rr_m", "mean"), F_MRR=("rr_f", "mean")).reset_index())  # fmt: skip
    b_pooled["MRR_diff"] = b_pooled.M_MRR - b_pooled.F_MRR
    summ = []
    for (lang, sp), g in bc.groupby(["lang", "space"]):
        g = g.set_index("cue_gender")
        summ.append({"lang": lang, "space": sp,
                     "agree_masc_cue (M MRR)": g.loc["m", "M_MRR"], "agree_fem_cue (F MRR)": g.loc["f", "F_MRR"],
                     "agreement_gap": g.loc["m", "M_MRR"] - g.loc["f", "F_MRR"],
                     "wrong_form_masc_cue (F MRR)": g.loc["m", "F_MRR"], "wrong_form_fem_cue (M MRR)": g.loc["f", "M_MRR"],
                     "inv_M_MRR": g.loc["inv", "M_MRR"] if "inv" in g.index else np.nan,
                     "inv_F_MRR": g.loc["inv", "F_MRR"] if "inv" in g.index else np.nan})  # fmt: skip
    summ = pd.DataFrame(summ)
    summ["inv_bias_gap"] = summ.inv_M_MRR - summ.inv_F_MRR
    a.to_csv(OUT / "a_reconstruction.csv", index=False)
    b_pooled.to_csv(OUT / "b_fixed_their_cues.csv", index=False)
    bc.to_csv(OUT / "bc_by_cue_gender.csv", index=False)
    summ.to_csv(OUT / "c_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("(a) reconstruction vs Table 2\n", a.round(4).to_string(index=False))
        print("\n(b) fixed rules, their cues pooled\n", b_pooled.round(4).to_string(index=False))
        print("\n(b/c) by cue gender\n", bc.round(4).to_string(index=False))
        print("\n(c) summary\n", summ.round(4).to_string(index=False))
    print("=== WORDPAIR done")


if __name__ == "__main__":
    main()
