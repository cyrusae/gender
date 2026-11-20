"""Kaggle runner for the content embedders (character study, exploratory). One GPU, fp32.
MODE (edit before each push): "check" = environment only (GPU, dataset, HF secret);
"test" = time 2,000 writer rows on bge-m3 + embeddinggemma; "full" = SCORERS below, every output,
chunked (10,000 rows), stops starting new chunks after BUDGET_H hours. Outputs: /kaggle/working/full/.
"""

import glob
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

MODE = "test"
SCORERS = ["bge-m3"]
BUDGET_H = 11.0
CH = 10000

T0 = time.time()
W = Path("/kaggle/working")
anchors = glob.glob("/kaggle/input/**/anchors.json", recursive=True)
assert anchors, "dataset not attached"
D = Path(anchors[0]).parent
code = D / "code"
if not (code / "content_embed.py").exists():
    with zipfile.ZipFile(D / "code.zip") as z:
        z.extractall(W / "code")
    code = W / "code" if (W / "code/content_embed.py").exists() else W / "code/code"
print("dataset", D, "| code", code, "| files", sorted(p.name for p in D.iterdir()), flush=True)

try:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN secret: attached", flush=True)
except Exception as e:  # noqa: BLE001
    print("HF_TOKEN secret: MISSING", type(e).__name__, flush=True)

import torch

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, flush=True)  # fmt: skip
try:
    import sentence_transformers

    print("sentence-transformers", sentence_transformers.__version__, flush=True)
except ImportError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=True
    )
if MODE == "check":
    print("=== CHECK DONE")
    sys.exit(0)

env = os.environ | {"ANCHORS_JSON": str(D / "anchors.json"), "CONTENT_DEVICE": "cuda",
                    "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(code)}  # fmt: skip


def inp(stem: str) -> Path:
    """Kaggle unpacks uploaded .gz files, so accept either name."""
    return next(p for p in (D / f"{stem}.jsonl", D / f"{stem}.jsonl.gz") if p.exists())


OUT = W / "full"
OUT.mkdir(exist_ok=True)


def emb(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(code / "content_embed.py"), *args], env=env,
                          capture_output=True, text=True)  # fmt: skip


if MODE == "test":
    f = inp("writer_gpt-6-luna")
    for e in ("bge-m3", "embeddinggemma"):
        t = time.time()
        p = emb([str(f), "--only", e, "--take", "2000", "--out", str(OUT / f"test_{e}.csv")])
        print(
            e,
            "rc",
            p.returncode,
            f"{time.time() - t:.0f} s for 2000 rows",
            p.stdout[-300:],
            p.stderr[-800:] if p.returncode else "",
            flush=True,
        )
    print("=== TEST DONE")
    sys.exit(0)

log = []
for e in SCORERS:
    for kind in ("writer", "character"):
        for fn in ("gpt-6-luna", "qwen3-14b_off", "qwen3-14b_on"):
            f = str(inp(f"{kind}_{fn}"))
            for ps, flags in (("all", []), ("kept", ["--keep-they", "--they-only"])):
                n = int(
                    emb([f, "--only", e, *flags, "--count", "--out", "/dev/null"]).stdout.split()[
                        -1
                    ]
                )
                for k in range((n + CH - 1) // CH):
                    out = OUT / f"{e}_{kind}_{fn}_{ps}_{k:03d}.csv"
                    if out.exists():
                        continue
                    if time.time() - T0 > BUDGET_H * 3600:
                        print("=== BUDGET REACHED (resume in a new version)", flush=True)
                        sys.exit(0)
                    p = emb(
                        [
                            f,
                            "--only",
                            e,
                            *flags,
                            "--skip",
                            str(k * CH),
                            "--take",
                            str(CH),
                            "--out",
                            str(out),
                        ]
                    )
                    msg = f"{time.strftime('%H:%M:%S')} {out.name} rc={p.returncode}"
                    print(msg, p.stderr[-500:] if p.returncode else "", flush=True)
                    log.append(msg)
(W / "chunks.json").write_text(json.dumps(log))
print("=== FULL DONE")
