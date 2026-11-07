"""Power of the pooled 5-point writer test with a correction across the four confirmatory
model x thinking cells (Luna none/medium, Qwen3-14B off/on). Same generative model and analysis as
power_sim.py; reports power at alpha .05 (uncorrected) and .0125 (Bonferroni over 4 = the
strictest step of Holm; Holm's per-cell power lies between the two). Same reps reused for both.
Writes character/results/power/power_4cells.csv."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from power_sim import R, one

rng = np.random.default_rng(20261030)
rows = []
for n, tau, g in itertools.product([10, 20, 30], [0.05, 0.10, 0.15, 0.20, 0.25], [0.95, 0.60]):
    out = [one(rng, 0.05, tau, n, g) for _ in range(1000)]
    b = np.array([o[0] for o in out])
    p = np.array([o[1] for o in out])
    rows.append({"draws_per_arm": n, "tau": tau, "gendered_share": g,
                 "power_alpha05": np.mean((p < 0.05) & (b > 0)),
                 "power_alpha0125": np.mean((p < 0.0125) & (b > 0))})  # fmt: skip
    print(rows[-1], flush=True)
pd.DataFrame(rows).round(3).to_csv(R / "power/power_4cells.csv", index=False)
print("=== POWER 4CELLS DONE")
