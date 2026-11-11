"""Word-pair test with the full adjective list (docs/wordpair-cues-prereg.md, ADOPTED 2026-11-11).
Cues frozen in data/wp_cues_frozen_v1.csv (sha in the adoption record). Committed before any score.

Queries, spaces and scoring as wordpair_mechanism.py (misses = reciprocal rank 0). Cue sets:
gendered pairs (one masculine and one feminine cue each) and invariant adjectives.
  intrusion       = mean over feminine cues of (RR_m − RR_f)
  agreement gap   = mean RR_m over masculine cues − mean RR_f over feminine cues
  invariant gap   = mean over invariant cues of (RR_m − RR_f)
Bootstrap 10,000 (seed 0): occupations, gendered pairs (masc and fem cue together) and invariant
cues resampled independently.
Q1 intrusion in ES original (sensitivity: without fem_also_noun pairs). Q2 change per mitigation.
Q3 per-pair intrusion ~ adjective stereotype (Williams & Best 1990 primary, Glasgow secondary;
standardised slope; bootstrap over pairs and occupations). Q4 per-occupation intrusion ~ D + R + H
+ S_BLS (as the mechanism study). Disability subset and French: descriptive.

    uv run --no-project --python 3.12 --with numpy,pandas,scikit-learn python embeddings/scripts/wordpair_cues.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from wordpair_mechanism import dict_masc_only, glasgow, ols_boot, rr_table, zs
from zhou_shift import Z, load_vec
from zhou_wordpair import spaces

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/wordpair_cues"
N_BOOT, SEED = 10_000, 0


def gaps(Rm, Rf, Ri, pi, ii, oi):
    """Rm/Rf: [pair, occ, 2] tables for masc/fem cues; Ri: [inv, occ, 2]."""
    M, F = Rm[np.ix_(pi, oi)], Rf[np.ix_(pi, oi)]
    out = {"intrusion": (F[..., 0] - F[..., 1]).mean(), "agreement_gap": M[..., 0].mean() - F[..., 1].mean()}
    out["invariant_gap"] = (Ri[np.ix_(ii, oi)][..., 0] - Ri[np.ix_(ii, oi)][..., 1]).mean() if len(ii) else np.nan
    return out


def interval(x):
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return lo, hi


def part_gaps(tabs, n_pairs, n_inv, n_occ, lang, tag, rng):
    full = (np.arange(n_pairs), np.arange(n_inv), np.arange(n_occ))
    pt = {sp: gaps(*t, *full) for sp, t in tabs.items()}
    draws = []
    for _ in range(N_BOOT):
        idx = (rng.integers(0, n_pairs, n_pairs), rng.integers(0, n_inv, n_inv), rng.integers(0, n_occ, n_occ))
        g = {sp: gaps(*t, *idx) for sp, t in tabs.items()}
        draws.append({(sp, k): v for sp in g for k, v in g[sp].items()}
                     | {(sp + "-original", k): g[sp][k] - g["original"][k] for sp in g if sp != "original" for k in g[sp]})  # fmt: skip
    B = pd.DataFrame(draws)
    rows = []
    for sp, k in B.columns:
        v = pt[sp][k] if "-" not in sp else pt[sp.split("-")[0]][k] - pt["original"][k]
        lo, hi = interval(B[(sp, k)])
        ch = "-" in sp
        verdict = (("reduces" if hi < 0 else "increases" if lo > 0 else "no clear change") if ch
                   else ("present" if lo > 0 else "reversed" if hi < 0 else "inconclusive"))  # fmt: skip
        rows.append({"lang": lang, "set": tag, "space": sp, "measure": k, "value": v, "lo": lo, "hi": hi, "verdict": verdict,
                     "n_pairs": n_pairs, "n_invariant": n_inv, "n_occ": n_occ})  # fmt: skip
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cues = pd.read_csv(ROOT / "data/wp_cues_frozen_v1.csv", keep_default_na=False, dtype=str)
    mp = pd.read_csv(ROOT / "data/occupation_mapping_v1.csv")
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    gl = glasgow()
    rng = np.random.default_rng(SEED)
    gap_rows, q3_rows, q4_rows, per_pair, dis_rows = [], [], [], [], []
    for lang in ("es", "fr"):
        sp, w2i, occ_all = spaces(lang, en, en_w2i)
        i2w = {i: w for w, i in w2i.items()}
        m = mp[mp.lang == lang].reset_index(drop=True)
        assert [tuple(x) for x in m[["fem", "masc", "en"]].values] == [tuple(o) for o in occ_all]
        keep = m.clean_keep.to_numpy(bool) & np.array([e in en_w2i for _, _, e in occ_all])
        occ = [o for o, k in zip(occ_all, keep) if k]
        c = cues[cues.lang == lang]
        for subset in sorted(c.subset.unique()):
            cs = c[c.subset == subset]
            G = cs[cs.type == "gendered"].reset_index(drop=True)
            Inv = cs[cs.type == "invariant"].reset_index(drop=True)
            mc = [(e, x, "m") for e, x in zip(G.en, G.masc)]
            fc = [(e, x, "f") for e, x in zip(G.en, G.fem)]
            ic = [(e, x, "inv") for e, x in zip(Inv.en, Inv.masc)]
            tabs = {}
            for name, (EN, _, T) in sp.items():
                if subset != "primary" and name != "original":
                    continue
                tabs[name] = (rr_table(EN, T, en_w2i, w2i, i2w, mc, occ), rr_table(EN, T, en_w2i, w2i, i2w, fc, occ),
                              rr_table(EN, T, en_w2i, w2i, i2w, ic, occ) if ic else np.zeros((0, len(occ), 2)))  # fmt: skip
            gap_rows += part_gaps(tabs, len(G), len(Inv), len(occ), lang, subset, rng)
            Rm, Rf, _ = tabs["original"]
            pp = pd.DataFrame({"lang": lang, "subset": subset, "masc": G.masc, "fem": G.fem, "en": G.en,
                               "intrusion": (Rf[..., 0] - Rf[..., 1]).mean(1),
                               "agree_m": Rm[..., 0].mean(1), "agree_f": Rf[..., 1].mean(1),
                               "wb": pd.to_numeric(G.wb_score_1990, errors="coerce"),
                               "glasgow": [gl.get(e, np.nan) for e in G.en], "fem_also_noun": G.fem_also_noun})  # fmt: skip
            per_pair.append(pp)
            if subset == "disability":
                dis_rows += pp[["lang", "masc", "fem", "en", "intrusion", "agree_m", "agree_f"]].to_dict("records")
                continue
            # Q1 sensitivity: without feminine forms that are also nouns
            nn = (G.fem_also_noun != "True").to_numpy()
            sub = {k: (v[0][nn], v[1][nn], v[2]) for k, v in tabs.items()}
            gap_rows += part_gaps(sub, int(nn.sum()), len(Inv), len(occ), lang, "primary, no fem_also_noun", rng)
            # Q3: per-pair intrusion ~ adjective stereotype
            for src in ("wb", "glasgow"):
                ok = pp[src].notna().to_numpy()
                x, Y = pp[src].to_numpy()[ok], (Rf[ok][..., 0] - Rf[ok][..., 1])  # [pair, occ]

                def slope(pi, oi):
                    y = Y[np.ix_(pi, oi)].mean(1)
                    xx = zs(x[pi])
                    return float(np.polyfit(xx, zs(y), 1)[0]) if xx.std() > 0 else np.nan

                b = slope(np.arange(len(x)), np.arange(Y.shape[1]))
                bs = [slope(rng.integers(0, len(x), len(x)), rng.integers(0, Y.shape[1], Y.shape[1])) for _ in range(N_BOOT)]
                lo, hi = interval(bs)
                q3_rows.append({"lang": lang, "score": src, "beta_std": b, "lo": lo, "hi": hi, "n_pairs": int(ok.sum()),
                                "verdict": "matters" if (lo > 0 or hi < 0) else "not shown"})  # fmt: skip
            # Q4: per-occupation intrusion ~ D + R + H + S_BLS (ES only; FR descriptive)
            if lang == "es":
                mk = m[keep].reset_index(drop=True)
                d = pd.DataFrame({"I": (Rf[..., 0] - Rf[..., 1]).mean(0),
                                  "D": [v for v, k in zip(dict_masc_only(lang, occ_all), keep) if k],
                                  "R": [np.log((w2i[f] + 1) / (w2i[mm] + 1)) for f, mm, _ in occ],
                                  "H": mk.fem_also_nonperson.notna().astype(float),
                                  "S_bls": 100 - pd.to_numeric(mk.bls_pct_women, errors="coerce")}).dropna()  # fmt: skip
                for res in ols_boot(d, "I", ["D", "R", "H", "S_bls"], rng):
                    q4_rows.append({"lang": lang, "n": len(d)} | res)
    G_, Q3, Q4 = pd.DataFrame(gap_rows), pd.DataFrame(q3_rows), pd.DataFrame(q4_rows)
    PP, DIS = pd.concat(per_pair), pd.DataFrame(dis_rows)
    G_.to_csv(OUT / "gaps.csv", index=False)
    Q3.to_csv(OUT / "q3_cue_stereotype.csv", index=False)
    Q4.to_csv(OUT / "q4_occupation.csv", index=False)
    PP.to_csv(OUT / "per_pair.csv", index=False)
    DIS.to_csv(OUT / "disability.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(G_.round(4).to_string(index=False))
        print(Q3.round(3).to_string(index=False))
        print(Q4.round(3).to_string(index=False))
        print(DIS.round(3).to_string(index=False))
    print("=== WPCUES done")


if __name__ == "__main__":
    main()
