"""Run every session step for one model in a single process, loading the model once.

For models whose loading dominates (Qwen3-30B-A3B: ~25 min of MoE weight conversion per load,
~10 loads in session.sh). Same pipeline functions and arguments as session.sh; only
`load_model` is memoised, so measurements are unchanged.
Usage:  uv run python runpod/one_process.py Qwen/Qwen3-30B-A3B-Base
"""

import sys
import time

from gbleed import models

_cache: dict = {}
_orig = models.load_model


def _cached(model_id, device=None, dtype=None):
    key = (model_id, str(device), str(dtype))
    if key not in _cache:
        _cache[key] = _orig(model_id, device, dtype)
    return _cache[key]


import gbleed.framecheck
import gbleed.multigender
import gbleed.phase0
import gbleed.phase1
import gbleed.phase2
import gbleed.phase3
import gbleed.phase3_known
import gbleed.readout
from gbleed.cli import main

for mod in (models, gbleed.framecheck, gbleed.multigender, gbleed.phase0, gbleed.phase1,
            gbleed.phase2, gbleed.phase3, gbleed.phase3_known, gbleed.readout):  # fmt: skip
    if hasattr(mod, "load_model"):
        mod.load_model = _cached

M = sys.argv[1]
STEPS = [
    ["phase0", M, "--stimuli", "data/stimuli/phase0_v3.csv"],
    ["multi-check", M],
    ["phase0", M, "--stimuli", "data/stimuli/phase2_pool_v4.csv", "--out", "results/phase2_known"],
    ["phase3-known", M],
    ["frame-check", M],
    ["readout", M, "--extract-only"],
    ["phase1", M, "--extract-only"],
    ["phase2", M, "--extract-only"],
    ["phase2", M, "--position", "after", "--extract-only"],
    ["phase3", M, "--extract-only"],
]
for args in STEPS:
    t = time.time()
    print(f"=== {time.strftime('%H:%M:%S')} start: {' '.join(args)}", flush=True)
    sys.argv = ["gbleed", *args]
    main()
    print(
        f"=== {time.strftime('%H:%M:%S')} done:  {' '.join(args)} ({time.time() - t:.0f} s)",
        flush=True,
    )
