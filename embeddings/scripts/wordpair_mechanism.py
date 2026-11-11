"""Word-pair intrusion: intervals (Part A) and mechanism (Part B).
docs/wordpair-mechanism-prereg.md, ADOPTED 2026-11-10; mapping data/occupation_mapping_v1.csv (sha in
the adoption record). Committed before any Part B score.

Queries and spaces as zhou_wordpair.py (fixed rules: a form outside the top 999 has reciprocal rank 0).
Per cue c and occupation o: rr_m, rr_f = reciprocal ranks of the masculine / feminine form.
  intrusion       = mean over feminine cues of (rr_m − rr_f)
  agreement gap   = mean rr_m over masculine cues − mean rr_f over feminine cues
  invariant gap   = mean over invariant cues of (rr_m − rr_f)
Part A: 95% bootstrap (10,000, seed 0) over occupations and, independently, cues within cue gender.
Part B (original MUSE space): I_o = per-occupation intrusion; D_o dictionary lists only the masculine
form; R_o = log(rank_f / rank_m) in frequency order; H_o feminine form also a non-person word;
S_o = stereotype (BLS primary, Glasgow, EU-27). Standardised OLS, bootstrap over occupations.
Alignment ablation: Procrustes onto original English, three dictionaries.

    uv run --no-project --python 3.12 --with numpy,pandas,scikit-learn python embeddings/scripts/wordpair_mechanism.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from zhou_hybrid import DICT, procrustes
from zhou_shift import Z, load_vec
from zhou_wordpair import CUES, FEM, K, spaces

ROOT = Path(__file__).resolve().parents[1]
GIT = ROOT.parent
OUT = ROOT / "results/wordpair_mechanism"
N_BOOT, SEED = 10_000, 0


def rr_table(EN, T, en_w2i, w2i, i2w, cues, occ):
    """Array [cue, occ, 2] of reciprocal ranks (masc, fem); 0 when outside the top 999."""
    Tn = T / np.linalg.norm(T, axis=1, keepdims=True)
    out = np.zeros((len(cues), len(occ), 2))
    for ci, (ce, cx, _) in enumerate(cues):
        for oi, (f, m, e) in enumerate(occ):
            q = EN[en_w2i[e]] - EN[en_w2i[ce]] + T[w2i[cx]]
            s = Tn @ (q / np.linalg.norm(q))
            top = [i2w[i] for i in np.argsort(-s)[:K] if i2w[i] != cx][:999]
            pos = {w: r + 1 for r, w in enumerate(top)}
            out[ci, oi] = [1 / pos[m] if m in pos else 0, 1 / pos[f] if f in pos else 0]
    return out


def measures(R, g, ci, oi):
    """R [cue, occ, 2]; g cue genders; ci, oi resampled indices (cue indices already within gender)."""
    X = R[np.ix_(ci, oi)]
    gg = g[ci]
    d = X[..., 0] - X[..., 1]
    return {"intrusion": d[gg == "f"].mean(), "agreement_gap": X[gg == "m"][..., 0].mean() - X[gg == "f"][..., 1].mean(),
            "invariant_gap": d[gg == "inv"].mean() if (gg == "inv").any() else np.nan}  # fmt: skip


def part_a(tables, g, n_occ, lang, tag):
    rng = np.random.default_rng(SEED)
    full = (np.arange(len(g)), np.arange(n_occ))
    pt = {sp: measures(R, g, *full) for sp, R in tables.items()}
    draws = []
    for _ in range(N_BOOT):
        ci = np.concatenate([rng.choice(np.flatnonzero(g == x), (g == x).sum()) for x in ("m", "f", "inv") if (g == x).any()])
        oi = rng.integers(0, n_occ, n_occ)
        m = {sp: measures(R, g, ci, oi) for sp, R in tables.items()}
        draws.append({f"{sp}|{k}": v for sp, mm in m.items() for k, v in mm.items()}
                     | {f"{sp}-original|{k}": m[sp][k] - m["original"][k] for sp in m if sp != "original" for k in m[sp]})  # fmt: skip
    B = pd.DataFrame(draws)
    rows = []
    for col in B.columns:
        sp, k = col.split("|")
        v = pt[sp][k] if "-" not in sp else pt[sp.split("-")[0]][k] - pt["original"][k]
        lo, hi = np.nanpercentile(B[col], [2.5, 97.5])
        is_change = "-" in sp
        verdict = (("reduces" if hi < 0 else "increases" if lo > 0 else "no clear change") if is_change
                   else ("present" if lo > 0 else "reversed" if hi < 0 else "inconclusive"))  # fmt: skip
        rows.append({"lang": lang, "list": tag, "space": sp, "measure": k, "value": v, "lo": lo, "hi": hi, "verdict": verdict})
    return rows


def zs(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std() if x.std() > 0 else x * 0


def ols_boot(df, y, xs, rng):
    def fit(d):
        X = np.column_stack([np.ones(len(d))] + [zs(d[c]) for c in xs])
        return np.linalg.lstsq(X, zs(d[y]), rcond=None)[0][1:]

    pt = fit(df)
    B = np.array([fit(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(N_BOOT)])
    lo, hi = np.nanpercentile(B, [2.5, 97.5], axis=0)
    return [{"term": c, "beta": b, "lo": a, "hi": h, "sig": "above 0" if a > 0 else "below 0" if h < 0 else "includes 0"}
            for c, b, a, h in zip(xs, pt, lo, hi)]  # fmt: skip


def glasgow():
    g = pd.read_csv(GIT / "data/raw/glasgow/glasgow_norms_scott2019.csv", header=[0, 1])
    words = g.iloc[:, 0].astype(str).str.lower()
    col = [i for i, c in enumerate(g.columns) if c[0] == "GEND"][0]
    return dict(zip(words, pd.to_numeric(g.iloc[:, col], errors="coerce")))


def dict_masc_only(lang, occ):
    tr = {}
    for line in open(Path(str(DICT).replace("es-en", f"en-{lang}")), encoding="utf-8"):
        a, b = line.split()
        tr.setdefault(a, set()).add(b)
    out = []
    for f, m, e in occ:
        t = tr.get(e, set())
        out.append(np.nan if not ({m, f} & t) else float(m in t and f not in t))
    return out


def realign(X, w2i, en, en_w2i, lang, occ, mode):
    pairs, seen = [], set()
    drop = {w for f, m, e in occ for w in (f, m)} | {e for _, _, e in occ}
    for line in open(Path(str(DICT).replace("es-en", f"{lang}-en")), encoding="utf-8"):
        a, b = line.split()
        if a in w2i and b in en_w2i and a not in seen:
            if mode != "as_is" and (a in drop or b in drop):
                continue
            seen.add(a)
            pairs.append((w2i[a], en_w2i[b]))
    pairs = sorted(pairs)[:5000]
    if mode == "fem_only":
        pairs += [(w2i[f], en_w2i[e]) for f, m, e in occ if f in w2i and e in en_w2i]
    s, t = map(np.array, zip(*pairs))
    return X @ procrustes(X[s], en[t]).T


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mp = pd.read_csv(ROOT / "data/occupation_mapping_v1.csv")
    gl = glasgow()
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    a_rows, b_rows, abl_rows, per_occ = [], [], [], []
    for lang in ("es", "fr"):
        sp, w2i, occ_all = spaces(lang, en, en_w2i)
        i2w = {i: w for w, i in w2i.items()}
        cues = [c for c in CUES[lang] + FEM[lang] if c[1] in w2i and c[0] in en_w2i]
        g = np.array([c[2] for c in cues])
        m = mp[mp.lang == lang].reset_index(drop=True)
        assert [tuple(x) for x in m[["fem", "masc", "en"]].values] == [tuple(o) for o in occ_all]
        ok = np.array([e in en_w2i for _, _, e in occ_all])
        tables_all = {name: rr_table(EN, Tfix, en_w2i, w2i, i2w, cues, occ_all) for name, (EN, _, Tfix) in sp.items()}
        for tag, keep in (("clean", m.clean_keep.to_numpy(bool) & ok), ("as_published", ok)):
            a_rows += part_a({k: v[:, keep] for k, v in tables_all.items()}, g, int(keep.sum()), lang, tag)
        # Part B, original space, clean list
        keep = m.clean_keep.to_numpy(bool) & ok
        R = tables_all["original"]
        I = (R[g == "f"][..., 0] - R[g == "f"][..., 1]).mean(0)
        Iinv = (R[g == "inv"][..., 0] - R[g == "inv"][..., 1]).mean(0) if (g == "inv").any() else np.full(len(occ_all), np.nan)
        d = pd.DataFrame({"lang": lang, "fem": m.fem, "masc": m.masc, "en": m.en, "I": I, "I_inv": Iinv,
                          "D": dict_masc_only(lang, occ_all),
                          "R": [np.log((w2i[f] + 1) / (w2i[mm] + 1)) for f, mm, _ in occ_all],
                          "H": m.fem_also_nonperson.notna().astype(float),
                          "S_bls": 100 - pd.to_numeric(m.bls_pct_women, errors="coerce"),
                          "S_glasgow": [gl.get(e, np.nan) for e in m.en],
                          "S_eu27": 100 - pd.to_numeric(m.eu27_pct_women, errors="coerce"),
                          "S_fr2d": 100 - pd.to_numeric(m.fr_pct_women, errors="coerce"),
                          "keep": keep})  # fmt: skip
        per_occ.append(d)
        dk = d[d.keep]
        rng = np.random.default_rng(SEED)
        desc = lang == "fr"
        for src in ["S_bls", "S_glasgow", "S_eu27"] + (["S_fr2d"] if desc else []):
            sub = dk.dropna(subset=["D", src])
            r = np.corrcoef(sub.R, sub[src])[0, 1]
            b_rows.append({"lang": lang, "model": f"R~{src} correlation", "term": "r", "beta": r,
                           "sig": "not identifiable (|r|>0.6)" if abs(r) > 0.6 else "", "n": len(sub)})  # fmt: skip
            if not desc:
                for res in ols_boot(sub, "I", ["D", "R", "H", src], rng):
                    b_rows.append({"lang": lang, "model": f"I~D+R+H+{src}", "n": len(sub)} | res)
                s0 = sub[sub.H == 0]
                for res in ols_boot(s0, "I", ["D", "R", src], rng):
                    b_rows.append({"lang": lang, "model": f"sens H=0: I~D+R+{src}", "n": len(s0)} | res)
            fem_set = sub[(sub[src] <= 40) if src != "S_glasgow" else (sub[src] <= 3.5)]
            if len(fem_set):
                bt = [fem_set.I.to_numpy()[rng.integers(0, len(fem_set), len(fem_set))].mean() for _ in range(N_BOOT)]
                lo, hi = np.percentile(bt, [2.5, 97.5])
                b_rows.append({"lang": lang, "model": f"female-stereotyped ({src}) mean I", "term": "mean_I",
                               "beta": fem_set.I.mean(), "lo": lo, "hi": hi, "n": len(fem_set),
                               "sig": "above 0" if lo > 0 else "below 0" if hi < 0 else "includes 0",
                               "words": ", ".join(fem_set.masc)})  # fmt: skip
        if not desc:
            sub = dk.dropna(subset=["D"])
            for res in ols_boot(sub, "I", ["D", "R", "H"], rng):
                b_rows.append({"lang": lang, "model": "I~D+R+H", "n": len(sub)} | res)
        # alignment ablation (original English), all clean occupations
        X = sp["original"][1]
        Is = {}
        for mode in ("as_is", "no_occupations", "fem_only"):
            A = realign(X, w2i, en, en_w2i, lang, occ_all, mode)
            RA = rr_table(en, A, en_w2i, w2i, i2w, cues, occ_all)[:, keep]
            Is[mode] = (RA[g == "f"][..., 0] - RA[g == "f"][..., 1]).mean(0)
        for mode in ("no_occupations", "fem_only"):
            dlt = Is[mode] - Is["as_is"]
            bt = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(N_BOOT)]
            lo, hi = np.percentile(bt, [2.5, 97.5])
            abl_rows.append({"lang": lang, "mode": mode, "mean_I_as_is": Is["as_is"].mean(), "mean_I": Is[mode].mean(),
                             "change": dlt.mean(), "lo": lo, "hi": hi,
                             "verdict": "lowers I" if hi < 0 else "raises I" if lo > 0 else
                                        ("no change" if abs(dlt.mean()) < 0.05 else "inconclusive")})  # fmt: skip
    A_, B_, AB = pd.DataFrame(a_rows), pd.DataFrame(b_rows), pd.DataFrame(abl_rows)
    P = pd.concat(per_occ)
    A_.to_csv(OUT / "part_a.csv", index=False)
    B_.to_csv(OUT / "part_b_regression.csv", index=False)
    AB.to_csv(OUT / "part_b_ablation.csv", index=False)
    P.to_csv(OUT / "per_occupation.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 70):
        print(A_[A_.list == "clean"].round(4).to_string(index=False))
        print(B_.round(3).to_string(index=False))
        print(AB.round(4).to_string(index=False))
    print("=== WPMECH done")


if __name__ == "__main__":
    main()
