"""Bootstrap intervals for the cross-fitted cos(f-m, Zipf) (instruct prereg, amendment 1).

Resamples nouns within cell x gender; each resample's cosine averages `n_splits` item-aware half
splits (copies of a noun stay in one half), so duplicated rows never share noise across halves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_BOOT = 200
N_SPLITS = 50
SEED = 27


def _cos_items(M, X, gi, fi, items, strata, rng, n_splits) -> float:
    from .confidence_ctl import _coefs
    from .estimators import _halves_items, _item_groups

    groups = _item_groups(items, strata)
    acc = np.zeros(3)
    for _ in range(n_splits):
        A = _halves_items(len(items), groups, rng)
        gA, fA = _coefs(M, X, A, [gi, fi])
        gB, fB = _coefs(M, X, ~A, [gi, fi])
        acc += [(gA @ fB + gB @ fA) / 2, gA @ gB, fA @ fB]
    acc /= n_splits
    return float(acc[0] / np.sqrt(acc[1] * acc[2])) if acc[1] > 0 and acc[2] > 0 else np.nan


def run(model_id: str, n_boot: int = N_BOOT, n_splits: int = N_SPLITS) -> pd.DataFrame:
    from .confidence_ctl import ZIPF_COV
    from .estimators import _strata, boot_within, nuisance
    from .phase4 import de_train, es_train
    from .phase5 import LAYER_FRACS

    rng = np.random.default_rng(SEED)
    rows = []
    for lang in ("es", "de"):
        d = es_train(model_id, "last") if lang == "es" else de_train(model_id, "last")
        n_layers = d["X"].shape[1] - 1
        for k in sorted({max(1, round(f * n_layers)) for f in LAYER_FRACS}):
            X0 = d["X"][:, k].astype(np.float64)
            boots = []
            for _ in range(n_boot):
                idx = boot_within(d["cells"], d["y"], rng)
                cells, y = d["cells"][idx], d["y"][idx]
                Z = nuisance(cells, [c[idx] for c in d["covs"]])
                M = np.column_stack([y.astype(float), Z])
                fi = 1 + len(set(cells)) + ZIPF_COV
                boots.append(_cos_items(M, X0[idx], 0, fi, idx, _strata(cells, y), rng, n_splits))
            b = np.array(boots)
            lo, hi = np.nanquantile(b, [0.025, 0.975])
            rows.append({"model": model_id, "lang": lang, "layer": k, "boot_lo": lo,
                         "boot_hi": hi, "boot_median": float(np.nanmedian(b)),
                         "boot_nan": int(np.isnan(b).sum()),
                         "nearly_parallel": bool(lo > 0.8 or hi < -0.8)})  # fmt: skip
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def label(cos: float) -> str:
    if not np.isfinite(cos):
        return "undefined"
    if cos > 1:
        return "> 1 (estimator limit)"
    if cos < -1:
        return "< -1 (estimator limit)"
    return f"{cos:+.2f}"
