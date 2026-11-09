"""Gender ValNorm on inanimate nouns, MUSE arms (embeddings/docs/gvalnorm-prereg.md, adopted 2026-11-08;
French: Addendum A). The Omrani Sabbaghi & Caliskan erasure arm is a separate script.

Per-noun scores s(w) come from C1 (`results/c1_<lang>/c1_per_noun.csv`: arms × attribute sets).
Nouns: frozen C1 list ∩ Glasgow (concept_en, homonym senses averaged). Model per arm × attributes ×
vocab, on z-scored variables: s_z = β_G·masc + β_H·GEND_z + β_F·zipf_z. 95% bootstrap CIs over nouns
(2,000, seed 0). English anchor: the same concepts' English words in English MUSE, pilot attributes.

    uv run --no-project --python 3.12 --with numpy,pandas,scipy python embeddings/scripts/gvalnorm_run.py [es|fr]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gvalnorm_en import A, B, glasgow, load
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
MUSE_EN = Path("/Volumes/Lexar/gbleed/shared/embeddings/muse/wiki.multi.en.vec")


def z(x):
    return (x - x.mean()) / x.std(ddof=1)


def fit(df, col):
    X = np.column_stack([np.ones(len(df)), (df["gender"] == "m").astype(float), z(df["gend"]), z(df["zipf"])])
    y = z(df[col]).to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        i = rng.integers(0, len(df), len(df))
        Xi, yi = X[i], y[i]
        # re-standardise within the resample, as the point estimate does
        Xi = np.column_stack([np.ones(len(i)), Xi[:, 1], z(pd.Series(df["gend"].to_numpy()[i])), z(pd.Series(df["zipf"].to_numpy()[i]))])
        yi = z(pd.Series(df[col].to_numpy()[i])).to_numpy()
        boot.append(np.linalg.lstsq(Xi, yi, rcond=None)[0])
    lo, hi = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
    out = {}
    for k, name in [(1, "beta_G"), (2, "beta_H"), (3, "beta_F")]:
        out[name], out[f"{name}_lo"], out[f"{name}_hi"] = beta[k], lo[k], hi[k]
    out["r_gend"] = pearsonr(df[col], df["gend"])[0]
    for g in "mf":
        sub = df[df["gender"] == g]
        out[f"r_gend_{g}"] = pearsonr(sub[col], sub["gend"])[0]
    return out


def verdict(arm, base, key):
    excl = arm[f"{key}_lo"] > 0 or arm[f"{key}_hi"] < 0
    if excl and arm[key] >= 0.5 * base[key]:
        return "survives"
    if not excl and abs(arm[key]) < 0.25 * abs(base[key]):
        return "removed"
    return "reduced"


def main(lang="es"):
    out = ROOT / f"results/gvalnorm_{lang}"
    out.mkdir(parents=True, exist_ok=True)
    pn = pd.read_csv(ROOT / f"results/c1_{lang}/c1_per_noun.csv")
    gl = glasgow()
    pn["concept_l"] = pn["concept_en"].astype(str).str.lower()
    d = pn.merge(gl.rename(columns={"word": "concept_l"}), on="concept_l", how="inner")
    # English anchor: the concept words themselves in English MUSE
    V = load(MUSE_EN, set(d["concept_l"]) | set(A) | set(B), nmax=200000)
    un = lambda v: v / np.linalg.norm(v)  # noqa: E731
    Am, Bm = np.array([un(V[w]) for w in A]), np.array([un(V[w]) for w in B])
    d["s_en_anchor"] = [
        (un(V[w]) @ Bm.T).mean() - (un(V[w]) @ Am.T).mean() if w in V and w not in A + B else np.nan
        for w in d["concept_l"]
    ]
    cols = [c for c in pn.columns if "|" in c]
    rows, info = [], {}
    for vocab in ("50k", "200k"):
        dv = d[d["in_50k"]] if vocab == "50k" else d
        a = dv.dropna(subset=["s_en_anchor"])
        info[vocab] = {"n": len(dv), "n_m": int((dv.gender == "m").sum()), "n_f": int((dv.gender == "f").sum()),
                       "english_anchor_r": pearsonr(a["s_en_anchor"], a["gend"])[0], "anchor_n": len(a)}  # fmt: skip
        for c in cols:
            arm, att = c.split("|")
            rows.append({"vocab": vocab, "arm": arm, "attributes": att, **fit(dv, c)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "gvalnorm_fits.csv", index=False)
    d.to_csv(out / "gvalnorm_nouns.csv", index=False)
    # pre-registered reading, primary = 50k
    ver = []
    valid = info["50k"]["english_anchor_r"] >= 0.2
    for att in res["attributes"].unique():
        p = res.query("vocab=='50k' and attributes==@att").set_index("arm")
        base = p.loc["1_original"]
        tracks = base.beta_H_lo > 0 if valid else None
        for arm in p.index:
            if arm == "1_original":
                continue
            ver.append({"attributes": att, "arm": arm,
                        "tracks_human_ratings_original": tracks if valid else "no valid test",
                        "beta_H_orig": base.beta_H, "beta_H_arm": p.loc[arm].beta_H,
                        "beta_H_verdict": verdict(p.loc[arm], base, "beta_H") if valid else "no valid test",
                        "beta_G_orig": base.beta_G, "beta_G_arm": p.loc[arm].beta_G,
                        "beta_G_verdict": verdict(p.loc[arm], base, "beta_G")})  # fmt: skip
    pd.DataFrame(ver).to_csv(out / "gvalnorm_verdicts.csv", index=False)
    json.dump(info, open(out / "info.json", "w"), indent=1)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(json.dumps(info, indent=1))
        print(res.query("vocab=='50k'")[["arm", "attributes", "beta_G", "beta_G_lo", "beta_G_hi", "beta_H",
                                         "beta_H_lo", "beta_H_hi", "r_gend", "r_gend_m", "r_gend_f"]].round(3).to_string(index=False))  # fmt: skip
        print(pd.DataFrame(ver).round(3).to_string(index=False))
    print("=== GVALNORM done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "es")
