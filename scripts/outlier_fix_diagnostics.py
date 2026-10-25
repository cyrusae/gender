"""Outcome-blind diagnostics for candidate fixes to Gemma's dominant residual dimensions.

Run before choosing a fix for the cross-fitted cosines (R2, Phase 4 T2). Nothing here touches
frequency (Zipf) or any cross-language quantity: only the f−m training directions themselves.

Per model × language × Phase 5 layer × method, on the training nouns (German strat3 m/f/n, Spanish
strat; Gemma: known-check filtered); halves split within cell x gender, each half
residualised on its own spelling cells (as the production estimator):
  top1_share   variance share of the largest dimension after the transform
  len2_neg     share of 200 split-halves where vA·vB ≤ 0 for the f−m direction (true length > 0)
  reliab       mean cos(vA, vB): agreement of the two halves' f−m directions
  auc          half-B nouns scored on half-A's direction (gender kept?)
  cos_fm_n     German only: cross-fitted cos(f−m, (f+m)/2 − n), the estimator R2 uses (cross and
               squared lengths averaged over splits, then the ratio), on a quantity unrelated to
               frequency; cos_valid = finite and in [−1, 1]
Methods: raw; A standardise each dimension; B drop dimensions with >10% of the variance;
C remove the top principal component; D0 per-token RMS normalisation (Gemma's RMSNorm without its
learned weights, which can't be downloaded without a token).

    uv run python scripts/outlier_fix_diagnostics.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MODELS = {
    "google/gemma-3-4b-pt": ("/Volumes/Lexar/gbleed/gemma-pod/extract/activations", [8, 14, 19, 24]),
    "google/gemma-3-12b-pt": ("/Volumes/Lexar/gbleed/gemma-pod-12b/extract/activations", [12, 19, 26, 34]),
    "Qwen/Qwen3-8B-Base": ("/Volumes/Lexar/gbleed/gender-s2/activations", [9, 14, 20, 25]),
    "Qwen/Qwen3-14B-Base": ("/Volumes/Lexar/gbleed/gender-s2/activations", [10, 16, 22, 28]),
}
ROOT = "."  # Gemma known-check results (in results/)
N_SPLIT = 200


def _load(actdir: str, model_id: str, name: str):
    import json
    from pathlib import Path

    from gbleed.models import model_slug

    d = Path(actdir) / model_slug(model_id)
    return np.load(d / f"{name}.npy"), json.loads((d / f"{name}.json").read_text())


def nouns(actdir: str, model_id: str, lang: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from gbleed import known_filter as kf

    kf.ROOTS = {k: f"{ROOT}/{v}" for k, v in {"es": "results/phase2_known",
                                              "de": "results/phase3_known"}.items()}  # fmt: skip
    if lang == "de":
        df = pd.read_csv("data/stimuli/phase3_final_v3.csv", keep_default_na=False)
        X, meta = _load(actdir, model_id, "phase3_bare")
        d = df[df.set == "strat3"]
        cells = (d.ending + "|" + d.loan).to_numpy()
    else:
        df = pd.read_csv("data/stimuli/phase2_final_v4.csv", keep_default_na=False)
        X, meta = _load(actdir, model_id, "phase2_bare")
        d = df[df.set.isin(["strat", "matched"])]
        cells = d.ending.to_numpy()
    row = {w: i for i, w in enumerate(meta["words"])}
    k = kf.known_lemmas(model_id, lang)
    keep = np.ones(len(d), bool) if k is None else d.lemma.isin(k).to_numpy()
    idx = np.array([row[w] for w in d.lemma])[keep]
    return X[idx], d.gender.to_numpy()[keep], cells[keep]


def resid(x: np.ndarray, cells: np.ndarray) -> np.ndarray:
    Z = pd.get_dummies(cells).to_numpy(float)
    return x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]


def transform(x: np.ndarray, method: str) -> np.ndarray:
    if method == "raw":
        return x
    if method == "A":
        sd = x.std(0)
        return (x - x.mean(0)) / np.where(sd > 0, sd, 1)
    if method == "B":
        v = x.var(0)
        return x[:, v / v.sum() <= 0.10]
    if method == "C":
        xc = x - x.mean(0)
        u = np.linalg.svd(xc, full_matrices=False)[2][0]
        return xc - np.outer(xc @ u, u)
    if method == "D0":
        return x / np.sqrt((x**2).mean(1, keepdims=True))
    raise ValueError(method)


def diag(x: np.ndarray, g: np.ndarray, cells: np.ndarray, rng) -> dict:
    v = x.var(0)
    out = {"top1_share": float(v.max() / v.sum()), "dims": x.shape[1]}
    fm = np.isin(g, ["m", "f"])
    neg, rel, aucs, cos_n = [], [], [], []
    for _ in range(N_SPLIT):
        # halves within cell x gender strata, each half residualised on its own cells (as the
        # production estimator, which refits the cell design per half): residualising all nouns
        # first couples the halves and biases every cross-half product
        a = np.zeros(len(g), bool)
        key = np.char.add(np.char.add(cells.astype(str), "|"), g.astype(str))
        for k in np.unique(key):
            i = np.flatnonzero(key == k)
            a[rng.choice(i, len(i) // 2, replace=False)] = True
        b = ~a
        xr = np.empty_like(x)
        xr[a], xr[b] = resid(x[a], cells[a]), resid(x[b], cells[b])

        def fmdir(h, xr=xr):
            return xr[h & (g == "f")].mean(0) - xr[h & (g == "m")].mean(0)

        va, vb = fmdir(a), fmdir(b)
        neg.append(va @ vb <= 0)
        rel.append(va @ vb / np.linalg.norm(va) / np.linalg.norm(vb))
        s = xr[b & fm] @ va
        aucs.append(roc_auc_score(g[b & fm] == "f", s))
        if "n" in g:

            def ndir(h, xr=xr):
                return (xr[h & (g == "f")].mean(0) + xr[h & (g == "m")].mean(0)) / 2 - xr[
                    h & (g == "n")
                ].mean(0)

            na, nb = ndir(a), ndir(b)
            cos_n.append(((va @ nb + vb @ na) / 2, va @ vb, na @ nb))
    out |= {"len2_neg": float(np.mean(neg)), "reliab": float(np.mean(rel)),
            "auc": float(np.mean(aucs))}  # fmt: skip
    if cos_n:  # as R2: average cross and squared lengths over splits, then the ratio
        cross, l_fm, l_n = np.array(cos_n).mean(0)
        c = cross / np.sqrt(l_fm * l_n) if l_fm > 0 and l_n > 0 else np.nan
        out |= {"cos_fm_n": float(c), "cos_valid": bool(np.isfinite(c) and abs(c) <= 1)}
    return out


def main() -> None:
    rows = []
    for m, (actdir, layers) in MODELS.items():
        for lang in ("de", "es"):
            X, g, cells = nouns(actdir, m, lang)
            for L in layers:
                x0 = X[:, L, :].astype(np.float64)
                for meth in ("raw", "A", "B", "C", "D0"):
                    rng = np.random.default_rng(0)
                    x = transform(x0, meth)
                    rows.append({"model": m.split("/")[1], "lang": lang, "layer": L,
                                 "method": meth, "n": len(g), **diag(x, g, cells, rng)})  # fmt: skip
                print(m, lang, L, flush=True)
    d = pd.DataFrame(rows)
    os.makedirs("results/outlier_fix", exist_ok=True)
    d.to_csv("results/outlier_fix/diagnostics.csv", index=False)
    print(d.groupby(["model", "method"])[["top1_share", "len2_neg", "reliab", "auc",
                                           "cos_valid"]].mean().round(3).to_string())  # fmt: skip


if __name__ == "__main__":
    main()
