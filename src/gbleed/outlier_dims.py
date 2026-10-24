"""Outlier-dimension fix for the cross-fitted cosines on Gemma (amendment 2026-10-24).

Gemma's residual stream has one dimension carrying most of the variance across nouns (59–92% at
the Phase 5 layers; Qwen3 ≤ 1.6%), and the split-half cosine estimator breaks on it. Fix
(`docs/design/gemma-slim-prereg.md`, "Amendment: outlier-dimension fix"): per model and layer,
zero every dimension with more than OUTLIER_SHARE of the variance across all Phase 3 bare nouns
(`phase3_bare`, the reference file; outcome-blind). Zeroing is the same as dropping for every
dot product, length and regression the cosines use. No dimension qualifies on Qwen3, so Qwen
results are unchanged.
"""

from contextlib import contextmanager

import numpy as np

from . import activations as acts

OUTLIER_SHARE = 0.10
REFERENCE = "phase3_bare"


def outlier_dims(model_id: str, loader=None) -> dict[int, np.ndarray]:
    """{layer: dimensions to zero} from the reference file."""
    X, _ = (loader or acts.load)(model_id, REFERENCE)
    out = {}
    for L in range(X.shape[1]):
        v = X[:, L, :].astype(np.float64).var(0)
        out[L] = np.flatnonzero(v / v.sum() > OUTLIER_SHARE)
    return out


@contextmanager
def dropped(model_id: str):
    """Within the block, acts.load(model_id, ...) returns arrays with outlier dims zeroed."""
    prev = acts.load  # restored on exit; composes with other acts.load patches
    dims = outlier_dims(model_id, prev)

    def load(mid, name):
        X, meta = prev(mid, name)
        if mid == model_id:
            X = X.copy()
            for L, d in dims.items():
                if len(d):
                    X[:, L, d] = 0.0
        return X, meta

    acts.load = load
    try:
        yield dims
    finally:
        acts.load = prev
