"""Poll RunPod for a cheap >=80 GB GPU at host CUDA >= 12.9 (vLLM 0.31 needs torch 2.13, cu129/cu130).

Read-only key from ~/.config/runpod/price-check-key. Every INTERVAL s, queries both clouds; logs
one line per check to LOG; exits 0 printing "HIT ..." when any card is in stock at <= MAX $/h.
Usage: python character/scripts/price_watch.py [MAX] [INTERVAL] [LOG]
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

MAX = float(sys.argv[1]) if len(sys.argv) > 1 else 2.00
INTERVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 1800
LOG = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("price_watch.log")
KEY = (Path.home() / ".config/runpod/price-check-key").read_text().strip()
# PW_IGNORE="GPU_ID:Cloud@UNTIL_EPOCH,..." skips listings that failed to book (stale stock)
IGNORE = {}
for part in filter(None, os.environ.get("PW_IGNORE", "").split(",")):
    k, _, until = part.rpartition("@")
    IGNORE[k] = float(until)
CUDA = '["12.9","13.0","13.1","13.2"]'


def check() -> list[tuple]:
    out = []
    for secure in ("true", "false"):
        q = ("query { gpuTypes { id memoryInGb lowestPrice(input:{gpuCount:1, "
             f"secureCloud:{secure}, allowedCudaVersions:{CUDA}}}) "
             "{ uninterruptablePrice stockStatus } } }")  # fmt: skip
        req = urllib.request.Request(
            "https://api.runpod.io/graphql",
            data=json.dumps({"query": q}).encode(),
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            for g in json.loads(r.read())["data"]["gpuTypes"]:
                lp = g.get("lowestPrice") or {}
                p, st = lp.get("uninterruptablePrice"), lp.get("stockStatus")
                if (g.get("memoryInGb") or 0) >= 80 and p and st:
                    out.append(
                        (
                            p,
                            g["id"],
                            g["memoryInGb"],
                            "Secure" if secure == "true" else "Community",
                            st,
                        )
                    )
    return sorted(out)


while True:
    stamp = time.strftime("%F %T")
    try:
        rows = check()
    except Exception as e:  # noqa: BLE001 - keep polling through transient API errors
        LOG.open("a").write(f"{stamp} error {e!r}\n")
        time.sleep(INTERVAL)
        continue
    best = (
        ", ".join(f"{i} {m}GB {c} ${p:.2f} ({s})" for p, i, m, c, s in rows[:4]) or "none in stock"
    )
    LOG.open("a").write(f"{stamp} {best}\n")
    hits = [r for r in rows if r[0] <= MAX and IGNORE.get(f"{r[1]}:{r[3]}", 0) < time.time()]
    if hits:
        p, i, m, c, s = hits[0]
        print(f"HIT {stamp}: {i} ({m} GB, {c}) ${p:.2f}/h, stock {s}", flush=True)
        sys.exit(0)
    time.sleep(INTERVAL)
