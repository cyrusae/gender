"""Confound check on the 208-noun pool (no models). Rule fixed before computing (PI 2026-10-25):
use all nouns (option A) if every |r| < 0.2 between a language's gender indicator and each
covariate (concreteness, English frequency, Glasgow valence, Glasgow English gender rating);
otherwise trim (option B). Indicators: feminine-vs-masculine (nouns with f or m only) and, for
de/ru, neuter-vs-rest. Glasgow covariates use the nouns that have ratings (n reported)."""

from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
d = pd.read_csv(R / "feasibility/roundtrip.csv", keep_default_na=False)
d = d[d.final.astype(str) == "True"].copy()
for c in ("conc", "zipf_en", "val", "gend_en"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
cov = {"conc": "concreteness", "zipf_en": "English frequency", "val": "valence (Glasgow)",
       "gend_en": "English gender rating (Glasgow)"}  # fmt: skip
rows = []
for lg in ["de", "ru", "es", "hi", "ar"]:
    g = d[f"{lg}_g"]
    tests = {"f vs m": (g.isin(["f", "m"]), (g == "f").astype(float))}
    if (g == "n").any():
        tests["n vs rest"] = (g != "", (g == "n").astype(float))
    for tn, (mask, ind) in tests.items():
        for c, cname in cov.items():
            ok = mask & d[c].notna()
            r = ind[ok].corr(d.loc[ok, c])
            rows.append({"lang": lg, "contrast": tn, "covariate": cname, "n": int(ok.sum()),
                         "r": round(r, 3), "flag": abs(r) >= 0.2})  # fmt: skip
t = pd.DataFrame(rows)
t.to_csv(R / "confound/correlations.csv", index=False)
print(f"nouns: {len(d)}")
print(
    t.pivot_table(index=["lang", "contrast"], columns="covariate", values="r").round(2).to_string()
)
print("\nflags (|r| >= 0.2):")
print(t[t.flag].to_string(index=False) if t.flag.any() else "none")
