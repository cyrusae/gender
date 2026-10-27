"""Power simulation for the writer-prompt study (no models; 2026-10-27).

Question: with the 191-noun pool, how many draws per arm does the writer study need to detect the
pre-registered smallest effects of interest (SESOI: 5 points pooled, 10 points per language)?

Effect definition used here (needs PI sign-off): for each noun x gendered language, the SHIFT is
she-share(foreign-word arm) - she-share(English-word arm), she-share = she / (she + he). The effect
is the feminine-minus-masculine difference in that shift (neuter excluded), i.e. beta = 5 points
means feminine nouns move +2.5 and masculine nouns -2.5 relative to their English-word arm.

Generative model, per model:
- genders: the real Wiktionary genders of the 191 pool nouns in de, ru, es, hi, ar;
- baseline she-share in the English-word arm: Beta with mean 0.55, SD 0.33 per noun x language
  (pilot: English-arm she-share means 0.36-0.88, SDs 0.18-0.38 across four models);
- shift = c * beta / 2 + u_noun + u_cell, c = +1 feminine / -1 masculine / 0 neuter; tau^2 =
  var(u_noun) + var(u_cell), split half/half (pilot tau estimates 0.00-0.23 from 20 cells each);
- foreign-arm share = baseline + shift, clipped to [0.005, 0.995] (ceiling effects are real);
- per arm, N draws; each is gendered (he or she) with probability G, then she with the arm's share.
Analysis (simple stand-in for the pre-registered mixed model): cell shift estimates regressed on c
with noun-clustered (CR1) standard errors; pooled test two-sided alpha .05; per-language tests
two-sample t on the cell shifts (feminine vs masculine), alpha .01 (Bonferroni over 5 languages).
Outputs power and the SD of the pooled estimate (-> confidence-interval half-width for the
precision-based stopping rule). Writes instruct/results/power/power_grid.csv.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path(__file__).resolve().parents[1] / "results"
LANGS = ["de", "ru", "es", "hi", "ar"]
pool = pd.read_csv(R / "confound/pool_191_DRAFT.csv", keep_default_na=False)
G_CODE = np.array([[{"f": 1, "m": -1}.get(g, 0) for g in pool[f"{lg}_g"]] for lg in LANGS]).T
IS_FM = G_CODE != 0
N_NOUN = len(pool)
MU, SD = 0.55, 0.33
AB = MU * (1 - MU) / SD**2 - 1
A, B = MU * AB, (1 - MU) * AB


def one(rng, beta: float, tau: float, n: int, g: float) -> tuple[float, float, list[float]]:
    pe = rng.beta(A, B, size=G_CODE.shape)
    u = rng.normal(0, tau / np.sqrt(2), size=(N_NOUN, 1)) + rng.normal(
        0, tau / np.sqrt(2), size=G_CODE.shape
    )
    pf = np.clip(pe + G_CODE * beta / 2 + u, 0.005, 0.995)
    ne, nf = rng.binomial(n, g, G_CODE.shape), rng.binomial(n, g, G_CODE.shape)
    ok = IS_FM & (ne > 0) & (nf > 0)
    d = np.where(
        ok, rng.binomial(nf, pf) / np.maximum(nf, 1) - rng.binomial(ne, pe) / np.maximum(ne, 1), 0
    )
    # pooled: OLS d ~ 1 + c over ok cells, noun-clustered SE
    c = G_CODE[ok].astype(float)
    y = d[ok]
    X = np.column_stack([np.ones_like(c), c])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    e = y - X @ coef
    noun_idx = np.nonzero(ok)[0]
    meat = np.zeros((2, 2))
    for i in np.unique(noun_idx):
        s = X[noun_idx == i].T @ e[noun_idx == i]
        meat += np.outer(s, s)
    k = len(np.unique(noun_idx))
    V = XtX_inv @ meat @ XtX_inv * k / (k - 1) * (len(y) - 1) / (len(y) - 2)
    b_hat, se = 2 * coef[1], 2 * np.sqrt(V[1, 1])
    p_pool = 2 * stats.t.sf(abs(b_hat / se), k - 1)
    p_lang = []
    for j in range(len(LANGS)):
        f = d[ok[:, j] & (G_CODE[:, j] == 1), j]
        m = d[ok[:, j] & (G_CODE[:, j] == -1), j]
        p_lang.append(
            stats.ttest_ind(f, m, equal_var=False).pvalue if len(f) > 2 and len(m) > 2 else 1.0
        )
    return b_hat, p_pool, p_lang


def main(reps: int = 1000) -> None:
    rng = np.random.default_rng(20261027)
    rows = []
    grid = itertools.product([10, 20, 30, 50, 100], [0.05, 0.10, 0.15, 0.20, 0.25], [0.95, 0.60])
    for n, tau, g in grid:
        res = {}
        for beta, tag in [(0.05, "pool"), (0.10, "lang"), (0.0, "null")]:
            out = [one(rng, beta, tau, n, g) for _ in range(reps)]
            b = np.array([o[0] for o in out])
            pp = np.array([o[1] for o in out])
            pl = np.array([o[2] for o in out])
            if tag == "pool":
                res["power_pooled_5pt"] = np.mean((pp < 0.05) & (b > 0))
                res["sd_pooled_est"] = b.std()
            elif tag == "lang":
                pw = (pl < 0.01).mean(0)
                res |= {f"power_{lg}_10pt": v for lg, v in zip(LANGS, pw)}
            else:
                res["type1_pooled"] = np.mean(pp < 0.05)
        rows.append({"draws_per_arm": n, "tau": tau, "gendered_share": g} | res)
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out["ci_halfwidth_pooled"] = 1.96 * out.sd_pooled_est
    out["completions_per_model"] = N_NOUN * 17 * out.draws_per_arm
    (R / "power").mkdir(exist_ok=True)
    out.round(4).to_csv(R / "power/power_grid.csv", index=False)


if __name__ == "__main__":
    main()
