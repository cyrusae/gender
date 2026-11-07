"""Compare confirmatory structures for the writer test across four model x thinking cells
(Luna none / medium, Qwen3-14B off / on). No models; 2026-11-07.

Generative model as power_sim.py, per cell, with two additions:
- noun effects are CORRELATED across cells (rho): a noun that shifts a lot in one model tends to
  shift in the others (same nouns, same languages);
- Luna cells are 95% gendered, Qwen cells 60% (pilot-like).
True effect = 5 points (SESOI) in every cell (power), or 0 in every cell (false-alarm check).
N = 30 draws per arm (cap).

Structures compared (all one-sided in effect, two-sided p, overall false-alarm rate 5%):
  A. all four cells confirmatory, Holm over 4
  B. gatekeeping: one test on the four-cell average at .05; only if it passes, Holm over 4 cells
  C. one cell per model (Luna medium, Qwen off), Holm over 2; the other two descriptive
Writes character/results/power/power_structures.csv.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from power_sim import AB, G_CODE, IS_FM, MU, N_NOUN, R

A_, B_ = MU * AB, (1 - MU) * AB
G_CELLS = [0.95, 0.95, 0.60, 0.60]  # Luna none, Luna medium, Qwen off, Qwen on


def cell_shifts(rng, beta, tau, rho, n):
    shared = rng.normal(0, tau / np.sqrt(2), (N_NOUN, 1))
    ds, oks = [], []
    for g in G_CELLS:
        own = rng.normal(0, tau / np.sqrt(2), (N_NOUN, 1))
        u_noun = np.sqrt(rho) * shared + np.sqrt(1 - rho) * own
        u = u_noun + rng.normal(0, tau / np.sqrt(2), G_CODE.shape)
        pe = rng.beta(A_, B_, G_CODE.shape)
        pf = np.clip(pe + G_CODE * beta / 2 + u, 0.005, 0.995)
        ne, nf = rng.binomial(n, g, G_CODE.shape), rng.binomial(n, g, G_CODE.shape)
        ok = IS_FM & (ne > 0) & (nf > 0)
        d = rng.binomial(nf, pf) / np.maximum(nf, 1) - rng.binomial(ne, pe) / np.maximum(ne, 1)
        ds.append(np.where(ok, d, np.nan))
        oks.append(ok)
    return ds


def test(d):
    """Noun-clustered regression of cell shifts on gender code (as power_sim.py)."""
    ok = ~np.isnan(d)
    c, y = G_CODE[ok].astype(float), d[ok]
    X = np.column_stack([np.ones_like(c), c])
    XtX = np.linalg.inv(X.T @ X)
    coef = XtX @ X.T @ y
    e = y - X @ coef
    idx = np.nonzero(ok)[0]
    meat = np.zeros((2, 2))
    for i in np.unique(idx):
        s = X[idx == i].T @ e[idx == i]
        meat += np.outer(s, s)
    k = len(np.unique(idx))
    V = XtX @ meat @ XtX * k / (k - 1) * (len(y) - 1) / (len(y) - 2)
    b, se = 2 * coef[1], 2 * np.sqrt(V[1, 1])
    return b, 2 * stats.t.sf(abs(b / se), k - 1)


def holm(ps, bs, alpha):
    order = np.argsort(ps)
    rej = np.zeros(len(ps), bool)
    for k, i in enumerate(order):
        if ps[i] < alpha / (len(ps) - k):
            rej[i] = True
        else:
            break
    return rej & (np.array(bs) > 0)


def main(reps=500):
    rng = np.random.default_rng(20261107)
    rows = []
    for beta, tau, rho in itertools.chain(
        itertools.product([0.05], [0.10, 0.15, 0.20], [0.3, 0.6, 0.9]),
        itertools.product([0.0], [0.15], [0.3, 0.9]),
    ):
        acc = {k: [] for k in ("A_cells", "B_gate", "B_cells", "C_cells", "A_any", "B_any")}
        for _ in range(reps):
            ds = cell_shifts(rng, beta, tau, rho, 30)
            res = [test(d) for d in ds]
            bs, ps = [r[0] for r in res], np.array([r[1] for r in res])
            b_avg, p_avg = test(np.nanmean(np.stack(ds), 0))
            a = holm(ps, bs, 0.05)
            gate = p_avg < 0.05 and b_avg > 0
            bcell = a & gate
            c = holm(ps[[1, 2]], [bs[1], bs[2]], 0.05)
            acc["A_cells"].append(a)
            acc["B_gate"].append(gate)
            acc["B_cells"].append(bcell)
            acc["C_cells"].append(c)
            acc["A_any"].append(a.any())
            acc["B_any"].append(gate)
        A = np.array(acc["A_cells"])
        B = np.array(acc["B_cells"])
        C = np.array(acc["C_cells"])
        rows.append({
            "true_effect": beta, "tau": tau, "rho": rho,
            "A_any_cell": np.mean(acc["A_any"]), "A_luna_med": A[:, 1].mean(), "A_qwen_off": A[:, 2].mean(),
            "A_all4": A.all(1).mean(),
            "B_headline": np.mean(acc["B_gate"]), "B_luna_med": B[:, 1].mean(), "B_qwen_off": B[:, 2].mean(),
            "B_all4": B.all(1).mean(),
            "C_luna_med": C[:, 0].mean(), "C_qwen_off": C[:, 1].mean(),
        })  # fmt: skip
        print(rows[-1], flush=True)
    pd.DataFrame(rows).round(3).to_csv(R / "power/power_structures.csv", index=False)
    print("=== POWER STRUCTURES DONE")


if __name__ == "__main__":
    main()
