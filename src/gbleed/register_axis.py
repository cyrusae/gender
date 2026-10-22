"""Is the German "frequency" axis partly register? (`docs/design/register-axis-prereg.md`)

Register proxy: age of acquisition (Kuperman et al. 2012) of each noun's English gloss.
E1 AoA by gender (no model); E2 (primary) gender-frequency overlap with vs without an AoA
control, same rated nouns; E3 cross-fitted cosines of the frequency, AoA and gender
coefficients; E4 (exploratory) thread-3 axis positions vs AoA.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import frequency_axis as fa
from .estimators import _halves, _strata
from .models import model_slug

AOA = "data/raw/norms/kuperman_aoa/kuperman_aoa.csv"
OUT = Path("results/register_axis")
SEED = 23


def aoa_table() -> dict[str, float]:
    a = pd.read_csv(AOA, keep_default_na=False)
    a = a[pd.to_numeric(a["Rating.Mean"], errors="coerce").notna()]
    return dict(zip(a.Word.astype(str).str.lower(), a["Rating.Mean"].astype(float), strict=True))


def gloss_aoa(lemmas, lang: str, table: dict | None = None) -> tuple[np.ndarray, list[str]]:
    """AoA of each lemma's `concept_en` gloss: exact match, else the last word of a two-word
    gloss; NaN otherwise. Also returns which rule matched ("exact" / "head" / "")."""
    table = table or aoa_table()
    lex = pd.read_csv(f"data/lexicon/{lang}_nouns.csv", keep_default_na=False)
    gl = lex.drop_duplicates("lemma").set_index("lemma").concept_en
    out, how = [], []
    for w in lemmas:
        g = str(gl.get(w, "")).lower().strip()
        parts = g.split()
        if g in table:
            out.append(table[g])
            how.append("exact")
        elif len(parts) == 2 and parts[1] in table:
            out.append(table[parts[1]])
            how.append("head")
        else:
            out.append(np.nan)
            how.append("")
    return np.array(out, float), how


def e1(d: dict, aoa: np.ndarray, rng, n_boot: int = 2000) -> dict:
    """Feminine - masculine AoA at equal ending cell (least squares), bootstrap within
    cell x gender strata."""
    from .estimators import nuisance

    k = np.isfinite(aoa)
    y, cells, a = d["y"][k].astype(float), d["cells"][k], aoa[k]

    def coef(idx):
        M = np.column_stack([y[idx], nuisance(cells[idx])])
        return float(np.linalg.lstsq(M, a[idx], rcond=None)[0][0])

    strata = _strata(cells, y)
    bs = [coef(np.concatenate([rng.choice(s, len(s)) for s in strata])) for _ in range(n_boot)]
    return {"n_rated": int(k.sum()), "n_f": int(y.sum()), "n_m": int((1 - y).sum()),
            "f_minus_m": coef(np.arange(k.sum())), "lo95": float(np.quantile(bs, 0.025)),
            "hi95": float(np.quantile(bs, 0.975))}  # fmt: skip


def e3(d: dict, X: np.ndarray, aoa: np.ndarray, rng, n_splits: int = fa.N_SPLITS) -> dict:
    """Cross-fitted cosines of the frequency (Zipf), AoA and gender coefficients of the fit
    [gender, cells, covariates, AoA] on the rated nouns."""
    k = np.isfinite(aoa)
    sub = {kk: (v[k] if isinstance(v, np.ndarray) else [c[k] for c in v])
           for kk, v in d.items() if kk in ("y", "cells", "covs", "lemma")}  # fmt: skip
    M, zi = fa._design(sub, np.asarray(sub["covs"][fa.ZIPF], float), "lin", aoa[k])
    ai = M.shape[1] - 1
    Xk = X[k]
    strata = _strata(sub["cells"], sub["y"])
    halves = [_halves(sub["cells"], sub["y"], rng, strata) for _ in range(n_splits)]

    def coef(j):
        return lambda h: fa._coef(M, Xk, h, j)

    return {f"cos_{a}_{b}": fa._cross(halves, coef(i), coef(j))["cos"]
            for (a, i), (b, j) in ((("freq", zi), ("aoa", ai)), (("gender", 0), ("aoa", ai)),
                                   (("gender", 0), ("freq", zi)))}  # fmt: skip


def run(model_id: str, n_splits: int = fa.N_SPLITS, n_perm: int = fa.N_PERM) -> tuple:
    from .phase5 import LAYER_FRACS

    rng = np.random.default_rng(SEED)
    table = aoa_table()
    e1_rows, rows = [], []
    for lang in ("es", "de"):
        d = fa._data(model_id, lang)
        aoa, how = gloss_aoa(d["lemma"], lang, table)
        rated = np.isfinite(aoa)
        e1_rows.append({"model": model_id, "lang": lang, "n_total": len(aoa),
                        "n_exact": how.count("exact"), "n_head": how.count("head"),
                        **e1(d, aoa, rng)})  # fmt: skip
        n_layers = d["X"].shape[1] - 1
        for k in sorted({max(1, round(f * n_layers)) for f in LAYER_FRACS}):
            X = d["X"][:, k].astype(np.float64)
            base = {"model": model_id, "lang": lang, "layer": k}
            res = {}
            for tag, extra in (("subset_baseline", None), ("aoa_control", np.nan_to_num(aoa))):
                r = fa.regression_variant(d, X, "lin", rng, keep=rated, extra=extra,
                                          n_splits=n_splits, n_perm=n_perm)  # fmt: skip
                null = r.pop("null")
                res[tag] = fa._judge(r, null, None)
            rows.append(base | {f"{t}_{c}": res[t][c] for t in res
                                for c in ("cos", "overlap", "null_abs_cross_p95", "cross", "n")}
                        | e3(d, X, aoa, rng, n_splits))  # fmt: skip
            print(rows[-1], flush=True)
    return pd.DataFrame(e1_rows), pd.DataFrame(rows)


def reading(e2: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (m, lang), g in e2.groupby(["model", "lang"]):
        b = g[g.subset_baseline_overlap.astype(str) == "True"]
        gone = int((b.aoa_control_overlap.astype(str) != "True").sum())
        out.append({"model": m, "lang": lang, "layers_with_subset_overlap": len(b),
                    "gone_with_aoa": gone,
                    "register_explains": bool(len(b) and gone > len(b) // 2)})  # fmt: skip
    return pd.DataFrame(out)


def e4(model_id: str) -> dict:
    """Exploratory: Spearman of the thread-3 residual axis position with AoA (German)."""
    from scipy.stats import spearmanr

    t = pd.read_csv(fa.OUT / "threads" / f"thread3_nouns_{model_slug(model_id)}.csv",
                    keep_default_na=False)  # fmt: skip
    aoa, _ = gloss_aoa(t.lemma, "de")
    k = np.isfinite(aoa)
    r = spearmanr(t.resid_mean_z[k], aoa[k])
    return {"model": model_id, "n": int(k.sum()), "spearman": float(r.statistic),
            "p": float(r.pvalue)}  # fmt: skip


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    e1s, e2s = zip(*(run(m) for m in fa.MODELS), strict=True)
    e1df, e2df = pd.concat(e1s), pd.concat(e2s)
    e1df.drop_duplicates(["lang"]).to_csv(OUT / "e1_aoa_by_gender.csv", index=False)
    e2df.to_csv(OUT / "e2_e3_layers.csv", index=False)
    reading(e2df).to_csv(OUT / "e2_reading.csv", index=False)
    (OUT / "e4_exploratory.json").write_text(json.dumps([e4(m) for m in fa.MODELS], indent=1))
