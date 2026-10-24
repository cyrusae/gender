"""Amendment 2026-10-24: R2 and Phase 4 T2 with Gemma's outlier dimension zeroed + validity gate.

    uv run python scripts/run_outlier_fix.py c        # R2 cosine, all four models
    uv run python scripts/run_outlier_fix.py t2       # Phase 4 T2 values, Gemma only
    uv run python scripts/run_outlier_fix.py acd      # frequency-axis A/C/D (only if c found overlap)
Outputs: results/outlier_fix/. Run from the Gemma worktree for Gemma activations; Qwen rows use
GBLEED_ACT_DIR when set.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gbleed import confidence_ctl as cc
from gbleed import frequency_axis as fa
from gbleed import phase4
from gbleed.outlier_dims import dropped

OUT = Path("results/outlier_fix")
GEMMA = ["google/gemma-3-4b-pt", "google/gemma-3-12b-pt"]


def valid(c) -> bool:
    return bool(np.isfinite(c) and abs(c) <= 1)


def verdict(g: pd.DataFrame, col: str = "cos_gender_zipf") -> str:
    """Per model x language over the four Phase 5 layers (prereg amendment)."""
    ok = g[col].map(valid)
    ov = ok & g.overlap.astype(bool)
    no = ok & ~g.overlap.astype(bool)
    if (~ok).sum() >= 2:
        return "no valid test"
    if ov.sum() >= 3:
        return "overlap"
    if no.sum() >= 3:
        return "no overlap"
    return "mixed"


def run_c(models) -> pd.DataFrame:
    rows = []
    for m in models:
        with dropped(m) as dims:
            r = cc.run_c(m)
        r["dims_zeroed"] = [len(dims[int(L)]) for L in r.layer]
        rows.append(r)
    d = pd.concat(rows)
    d["valid"] = d.cos_gender_zipf.map(valid)
    d["overlap"] = d.overlap.astype(bool) & d.valid  # an invalid layer shows no overlap
    d.to_csv(OUT / "c_frequency_cosine.csv", index=False)
    v = d.groupby(["model", "lang"]).apply(verdict).rename("verdict").reset_index()
    v.to_csv(OUT / "c_verdicts.csv", index=False)
    print(d.to_string(), "\n", v.to_string())
    return d


def run_t2() -> None:
    out = {}
    for m in GEMMA:
        with dropped(m):
            s = phase4.analyze(m, out_root=str(OUT / "phase4"))
        lay = pd.read_csv(OUT / "phase4" / m.replace("/", "__") / "layers.csv")
        out[m] = {"t2_cos_mean": s.get("t2_cos_mean"),
                  "t2_layers_cos_above_cos_null": s.get("t2_layers_cos_above_cos_null"),
                  "t2_layers_valid": int(lay.t2_cos.map(valid).sum()), "layers": len(lay)}  # fmt: skip
    (OUT / "t2.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


def run_acd() -> None:
    v = pd.read_csv(OUT / "c_verdicts.csv")
    todo = v[(v.lang == "de") & (v.verdict == "overlap") & v.model.isin(GEMMA)].model.tolist()
    if not todo:
        print("no Gemma size has a German overlap verdict: A/C/D not run (prereg amendment)")
        return
    rows = []
    for m in todo:
        with dropped(m):
            rows.append(fa.run_acd(m, c_ref_path=OUT / "c_frequency_cosine.csv"))
    acd = pd.concat(rows)
    acd["valid"] = acd.cos.map(valid)
    acd.loc[~acd.valid, "overlap"] = True  # an invalid layer can't show the overlap gone
    acd.to_csv(OUT / "acd_layers.csv", index=False)
    fa.readings(acd).to_csv(OUT / "acd_readings.csv", index=False)
    print(fa.readings(acd).to_string())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    part = sys.argv[1]
    if part == "c":
        run_c(sys.argv[2:] or GEMMA)
    elif part == "t2":
        run_t2()
    elif part == "acd":
        run_acd()
