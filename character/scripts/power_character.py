"""Precision simulation for the character + modifiers run (no models; 2026-10-27).

Target (PI 2026-10-27): 95% CI half-width <= 5 points for each modifier's effect, where the effect
is the mean over nouns of she-share(modifier) - she-share(plain a/an), she-share = she/(she+he).
Per noun: plain baseline ~ Beta(mean 0.55, SD 0.33) (pilot-like); modifier shift ~ N(0, tau)
around it (tau = noun-to-noun spread of the modifier effect); draws are gendered with probability
G. Plain gets 10 draws per noun. Half-width = t(.975, k-1) x SD(noun diffs) / sqrt(k), k = nouns
with at least one gendered draw in both. Writes character/results/power/power_character.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

N_NOUN, N_PLAIN, REPS = 191, 10, 400
MU, SD = 0.55, 0.33
AB = MU * (1 - MU) / SD**2 - 1
rng = np.random.default_rng(20261027)
rows = []
for g in (0.95, 0.75, 0.55):
    for tau in (0.05, 0.10, 0.15, 0.20):
        for n in (2, 3, 4, 6, 8):
            hw = []
            for _ in range(REPS):
                pb = rng.beta(MU * AB, (1 - MU) * AB, N_NOUN)
                pm = np.clip(pb + rng.normal(0, tau, N_NOUN), 0.005, 0.995)
                np_, nm = rng.binomial(N_PLAIN, g, N_NOUN), rng.binomial(n, g, N_NOUN)
                ok = (np_ > 0) & (nm > 0)
                d = (
                    rng.binomial(nm, pm) / np.maximum(nm, 1)
                    - rng.binomial(np_, pb) / np.maximum(np_, 1)
                )[ok]
                k = ok.sum()
                hw.append(stats.t.ppf(0.975, k - 1) * d.std(ddof=1) / np.sqrt(k))
            rows.append({"gendered_share": g, "tau": tau, "draws_per_modifier": n,
                         "ci_halfwidth_pts": round(100 * np.mean(hw), 2),
                         "share_meeting_5pt": round(np.mean(np.array(hw) <= 0.05), 3)})  # fmt: skip
out = pd.DataFrame(rows)
p = Path(__file__).resolve().parents[1] / "results/power"
p.mkdir(exist_ok=True)
out.to_csv(p / "power_character.csv", index=False)
for g, s in out.groupby("gendered_share", sort=False):
    print(f"gendered share {g}: CI half-width (points)")
    print(s.pivot(index="draws_per_modifier", columns="tau", values="ci_halfwidth_pts").to_string())
