"""Shared estimators for the stratified Phase 2 (v4) and Phase 3 analyses.

Pre-registration: docs/design/phases-3-5-plan.md ("Phase 2 (v4) and Phase 3 analysis").

- `nuisance(cells, covs)`: design matrix of cell indicators (one per stratification cell; they
  also act as the intercept) plus centred covariates (frequency, token count, concreteness...).
- `residualise(X, Z)`: activations minus their least-squares fit on Z (Frisch-Waugh), so only
  within-cell variation, net of the covariates, is left.
- `probe_dir`: the single estimator (standardised L2 logistic regression, C = 1), as a direction
  in raw activation space.
- `split_half_geometry`: cross-fitted squared lengths, cosine and reliabilities of class vectors
  relative to a reference class. A plain squared length of a mean difference is biased upward by
  noise, by about tr(Sigma) (1/n_g + 1/n_ref), more for smaller groups; and two vectors sharing
  the reference mean share its noise, which biases their cosine upward. The dot product of
  estimates from two disjoint halves has neither bias (independent noise, zero expected
  cross-product).
- `shuffle_within(labels, cells)`: permute labels within cells, keeping every cell's counts (the
  noise reference: no real class differences, the same group sizes).
- `boot_within(cells, labels)`: bootstrap resample within cell x label strata.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def nuisance(cells, covs=()) -> np.ndarray:
    cells = np.asarray(cells)
    levels = sorted(set(cells))
    D = (cells[:, None] == np.array(levels)[None, :]).astype(float)
    C = [np.asarray(c, dtype=float) - np.mean(c) for c in covs]
    return np.column_stack([D, *C]) if C else D


def residualise(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """X minus its least-squares fit on Z: X - U U^T X, U = Z's left singular vectors (rank cut as
    np.linalg.lstsq's default). The same fitted values as lstsq, ~30x faster for wide X (lstsq's
    solver is slow with thousands of right-hand sides)."""
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    U = U[:, s > s[0] * max(Z.shape) * np.finfo(float).eps]
    return X - U @ (U.T @ X)


def auc(y, s) -> float:
    """ROC AUC of scores s for binary labels y (nan unless both classes are present).

    Exact Mann-Whitney form, P(positive scores above negative), ties counting half: the same
    value as sklearn's roc_auc_score, ~50x faster (it runs inside every bootstrap round)."""
    y = np.asarray(y).astype(bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _newton_logistic(C: np.ndarray, y1: np.ndarray, tol: float = 1e-9, max_iter: int = 100):
    """Binary L2 logistic regression, sklearn's objective with C = 1 (log-loss sum + ||a||^2 / 2,
    intercept unpenalised), by Newton's method with backtracking. Returns (a, b), or None if it
    doesn't converge (the caller falls back to sklearn)."""
    n, r = C.shape
    A = np.column_stack([C, np.ones(n)])
    pen = np.r_[np.ones(r), 0.0]
    w = np.zeros(r + 1)

    def f(w):
        eta = A @ w
        return np.logaddexp(0, eta).sum() - y1 @ eta + 0.5 * (pen * w) @ w

    fw = f(w)
    for _ in range(max_iter):
        p = 0.5 * (1 + np.tanh(0.5 * (A @ w)))  # sigmoid, overflow-free
        g = A.T @ (p - y1) + pen * w
        if np.abs(g).max() < tol:
            return w[:r], w[r]
        H = (A.T * (p * (1 - p))) @ A + np.diag(pen)
        step = np.linalg.solve(H, g)
        t = 1.0
        while True:  # Armijo backtracking
            fn = f(w - t * step)
            if fn <= fw - 1e-4 * t * (g @ step) or t < 1e-10:
                break
            t *= 0.5
        w, fw = w - t * step, fn
    return None


def fit_logistic(Z: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """L2 logistic regression (C = 1) on Z, fitted in Z's row space when n < d.

    Same optimum as fitting on Z directly: the penalty sets any weight component outside the row
    space to zero, so the problem is solved on the n coordinates U*s (Z = U s V^T, from the n x n
    Gram matrix) and the weights mapped back with V. Several times faster for d in the thousands
    (bootstrap refits at 8B/14B); checked against the direct fit at tight tolerance (cosine 1.0)."""
    n, d = Z.shape
    if n >= d:
        return LogisticRegression(C=1.0, max_iter=5000).fit(Z, y)
    w, U = np.linalg.eigh(Z @ Z.T)
    keep = w > w.max() * 1e-10
    s, U = np.sqrt(w[keep]), U[:, keep]
    classes = np.unique(y)
    sol = _newton_logistic(U * s, (y == classes[-1]).astype(float)) if len(classes) == 2 else None
    if sol is None:  # multiclass (or no convergence): sklearn's lbfgs on the same coordinates
        lr = LogisticRegression(C=1.0, max_iter=5000).fit(U * s, y)
    else:  # an sklearn object carrying the Newton solution, so predict() etc. work unchanged
        lr = LogisticRegression(C=1.0)
        lr.classes_, lr.coef_, lr.intercept_ = classes, sol[0][None], np.array([sol[1]])
        lr.n_iter_ = np.array([0])
    lr.coef_ = lr.coef_ @ ((U / s).T @ Z)
    lr.n_features_in_ = d
    return lr


def probe_dir(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    sc = StandardScaler().fit(X)
    lr = fit_logistic(sc.transform(X), y)
    return lr.coef_[0] / sc.scale_


def probe_strat(X: np.ndarray, y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Probe on residualised activations: the direction can only use within-cell differences."""
    return probe_dir(residualise(X, Z), y)


def class_betas(X: np.ndarray, G: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Coefficients of the class indicators G (n x k; reference class = all zeros) in a least-
    squares fit of X on [G, Z]: per-class mean differences from the reference at equal cell and
    covariates (the stratified adjusted difference of means)."""
    M = np.column_stack([G, Z])
    return (np.linalg.pinv(M) @ X)[: G.shape[1]]


def _strata(cells, labels) -> list[np.ndarray]:
    """Row indices of every cell x label stratum, in sorted key order."""
    keys = np.char.add(np.asarray(cells).astype(str), "|" + np.asarray(labels).astype(str))
    return [np.where(keys == k)[0] for k in np.unique(keys)]


def _halves(cells, labels, rng, strata: list[np.ndarray] | None = None) -> np.ndarray:
    """Boolean mask of half A: within every cell x label stratum, half the items (odd counts:
    the extra item goes to a random half). `strata` (from `_strata`) skips recomputing them."""
    a = np.zeros(len(cells), dtype=bool)
    for idx in _strata(cells, labels) if strata is None else strata:
        idx = rng.permutation(idx)
        n = len(idx) // 2 + (rng.random() < 0.5 if len(idx) % 2 else 0)
        a[idx[:n]] = True
    return a


def _pinv_rows(Mh: np.ndarray, k: int) -> np.ndarray:
    """First k rows of pinv(Mh) for a stack of (n x p) matrices, via the p x p normal matrix."""
    MtM = Mh.transpose(0, 2, 1) @ Mh
    return np.linalg.pinv(MtM, hermitian=True)[:, :k] @ Mh.transpose(0, 2, 1)


def split_half_geometry(
    X, labels, cells, covs, classes, ref, rng, n_splits: int = 200, gram: np.ndarray | None = None
) -> dict:
    """Cross-fitted geometry of class vectors v_c = mean(c) - mean(ref) at equal cell/covariates.

    Returns, averaged over n_splits random half splits within cell x class strata:
      len2_<c>   squared length estimate v_c^A . v_c^B (unbiased; can be < 0 when ~0)
      cos        (v_1^A . v_2^B + v_1^B . v_2^A) / 2 / sqrt(len2_1 len2_2)   (two classes only)
      rel_<c>    cos(v_c^A, v_c^B): is the class vector's direction reproducible?
    Exact shortcut: every v is P @ X for a small matrix P (the class rows of the half's least-
    squares pseudo-inverse), so all dot products are P_A K P_B^T with the Gram matrix K = X X^T
    (n x n) instead of d-dimensional vectors. Covariates need no re-centring per half because
    the cell indicators of a half always sum to an intercept.
    """
    labels = np.asarray(labels)
    G = np.column_stack([(labels == c).astype(float) for c in classes])
    M = np.column_stack([G, nuisance(cells, covs)])
    K = X @ X.T if gram is None else gram
    k = len(classes)
    strata = _strata(cells, labels)
    A = np.stack([_halves(cells, labels, rng, strata) for _ in range(n_splits)])  # (S, n)
    # All splits at once: zeroing the other half's rows leaves the singular values unchanged and
    # gives pinv(M[half]) in the half's columns and zeros elsewhere (the per-split P_h).
    # pinv(Mh) = pinv(Mh^T Mh) Mh^T: the same matrix, from p x p instead of n x p decompositions.
    PA, PB = (_pinv_rows(M[None] * h[..., None], k) for h in (A, ~A))  # (S, k, n) each
    KA, KB = PA @ K, PB @ K
    AB = KA @ PB.transpose(0, 2, 1)  # (S, k, k): v_i^A . v_j^B
    AA = (KA * PA).sum(-1)  # (S, k): v_i^A . v_i^A
    BB = (KB * PB).sum(-1)
    diag = np.diagonal(AB, axis1=1, axis2=2)  # (S, k)
    rel = diag / (np.sqrt(np.maximum(AA, 0) * np.maximum(BB, 0)) + 1e-12)
    out = {f"len2_{c}": float(diag[:, i].mean()) for i, c in enumerate(classes)}
    out |= {f"rel_{c}": float(rel[:, i].mean()) for i, c in enumerate(classes)}
    if k == 2:
        cross = (AB[:, 0, 1] + AB[:, 1, 0]) / 2
        l1, l2 = out[f"len2_{classes[0]}"], out[f"len2_{classes[1]}"]
        out["cos"] = float(cross.mean() / np.sqrt(l1 * l2)) if l1 > 0 and l2 > 0 else float("nan")
    return out


def plain_geometry(X, labels, cells, covs, classes) -> dict:
    """The biased plug-in version (all nouns at once), reported for comparison only."""
    labels = np.asarray(labels)
    G = np.column_stack([(labels == c).astype(float) for c in classes])
    B = class_betas(X, G, nuisance(cells, covs))
    out = {f"len2_{c}": float(B[i] @ B[i]) for i, c in enumerate(classes)}
    if len(classes) == 2:
        out["cos"] = float(B[0] @ B[1] / (np.linalg.norm(B[0]) * np.linalg.norm(B[1]) + 1e-12))
    return out


def shuffle_within(labels, cells, rng) -> np.ndarray:
    labels = np.asarray(labels).copy()
    cells = np.asarray(cells)
    for c in np.unique(cells):
        idx = np.where(cells == c)[0]
        labels[idx] = labels[rng.permutation(idx)]
    return labels


def boot_within(cells, labels, rng) -> np.ndarray:
    """Bootstrap row indices, resampling within every cell x label stratum."""
    keys = np.char.add(np.asarray(cells).astype(str), "|" + np.asarray(labels).astype(str))
    return np.concatenate(
        [rng.choice(idx, len(idx)) for k in np.unique(keys) for idx in [np.where(keys == k)[0]]]
    )
