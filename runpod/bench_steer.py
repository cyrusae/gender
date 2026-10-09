"""Phase 5 steering throughput on one GPU (chip choice and session-2 cost; nothing is kept).

    uv run python runpod/bench_steer.py Qwen/Qwen3-8B-Base [--k-frac 0.4] [--per-readout 24]

Times `phase5.run` on a fixed, seeded sample of real readout prompts (both steering languages,
all four readouts) under the trimmed condition set, for several batch sizes, with random unit
vectors in place of the fitted direction (the cost doesn't depend on the vector). Prints rows/s
per readout and the hours for one full layer-language sweep at the best batch setting.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from gbleed import phase5
from gbleed.models import load_model, pick_device, pick_dtype
from gbleed.phase5_stimuli import NOUNS

BATCHES = [(16, 64), (16, 128), (32, 256), (64, 256), (32, 512)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--k-frac", type=float, default=0.40)
    ap.add_argument("--per-readout", type=int, default=24)
    ap.add_argument("--out", default="logs/bench_steer.json")
    a = ap.parse_args()
    dev = pick_device(None)
    model, tok = load_model(a.model, dev, pick_dtype(None, dev))
    k = round(a.k_frac * model.config.num_hidden_layers)
    nouns = pd.read_csv(NOUNS, keep_default_na=False)
    d = model.config.hidden_size
    V = torch.tensor(phase5.random_vectors(d, 1 + phase5.N_RANDOM, seed=7), dtype=torch.float32)
    rng = np.random.default_rng(0)
    res = {"model": a.model, "k": k, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev,
           "readouts": {}}  # fmt: skip
    for lang in ("es", "de"):
        sub = phase5.dose_subset(nouns[nouns.lang == lang])
        for name, prompts, read in phase5.readout_sets(tok, nouns, lang, phase5.family(a.model)):
            full = phase5.conditions(len(prompts), phase5.N_RANDOM, phase5.N_RANDOM_GATE,
                                     in_subset=[p.key["lemma"] in sub for p in prompts])  # fmt: skip
            idx = sorted(rng.choice(len(prompts), a.per_readout, replace=False))
            ps = [prompts[i] for i in idx]
            conds = phase5.conditions(len(ps), phase5.N_RANDOM, phase5.N_RANDOM_GATE,
                                      in_subset=[p.key["lemma"] in sub for p in ps])  # fmt: skip
            unit = phase5.noun_norm(model, tok, ps, k)
            phase5.run(model, tok, ps[:2], k, [c for c in conds if c[0] < 2][:64], V, read,
                       unit=unit)  # warm-up  # fmt: skip
            times = {}
            for pb, rb in BATCHES:
                if dev == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t = time.time()
                try:
                    phase5.run(model, tok, ps, k, conds, V, read, prompt_batch=pb, row_batch=rb,
                               unit=unit)  # fmt: skip
                except torch.OutOfMemoryError:
                    times[f"{pb}x{rb}"] = None
                    torch.cuda.empty_cache()
                    continue
                if dev == "cuda":
                    torch.cuda.synchronize()
                sec = time.time() - t
                mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else None
                times[f"{pb}x{rb}"] = {"rows_per_s": len(conds) / sec, "peak_gb": mem}
                print(f"{lang} {name} {pb}x{rb}: {len(conds) / sec:.0f} rows/s, peak {mem} GB",
                      flush=True)  # fmt: skip
            res["readouts"][f"{lang}_{name}"] = {"prompts_full": len(prompts),
                                                 "rows_full": len(full), "times": times}  # fmt: skip
    # one full layer-language sweep at the best single batch setting (same for every readout)
    keys = [f"{pb}x{rb}" for pb, rb in BATCHES]
    best, best_h = None, None
    for key in keys:
        r = [v for v in res["readouts"].values() if v["times"].get(key)]
        if len(r) < len(res["readouts"]):
            continue
        hours = {lang: sum(v["rows_full"] / v["times"][key]["rows_per_s"]
                           for n, v in res["readouts"].items() if n.startswith(lang)) / 3600
                 for lang in ("es", "de")}  # fmt: skip
        if best_h is None or sum(hours.values()) < sum(best_h.values()):
            best, best_h = key, hours
    res["best_batch"], res["hours_per_layer"] = best, best_h
    print(json.dumps({"gpu": res["gpu"], "best": best, "hours_per_layer": best_h}), flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
