"""They-masking check (PI 2026-10-26). Compares embedding scores of he/she paragraphs with plural
they/them/their masked (production) vs left intact (--keep-they). Reports, per embedder, the mean
and SD of the shift, and the shift relative to the she-vs-he score difference.
Usage: python they_mask_check.py PRODUCTION.csv KEPT.csv"""

import sys

import pandas as pd

keys = ["model", "name_condition", "noun", "modifier", "thinking", "sample", "fixed_sample"]
a = pd.read_csv(sys.argv[1])
b = pd.read_csv(sys.argv[2])
m = a.merge(b, on=keys, suffixes=("", "_kept"))
m = m[m.pron.isin(["she", "he"])]
for c in [c for c in a.columns if c.startswith("emb_") and not c.endswith("_sd")]:
    shift = m[c + "_kept"] - m[c]
    gap = m[m.pron == "she"][c].mean() - m[m.pron == "he"][c].mean()
    print(f"{c}: n {len(m)}, shift mean {shift.mean():+.4f} sd {shift.std():.4f}, "
          f"she-he gap {gap:+.4f}, |shift sd / gap| {abs(shift.std() / gap) if gap else float('nan'):.2f}")  # fmt: skip
