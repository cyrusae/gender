"""Build the private Kaggle dataset for the content embedders (2026-11-20): slim gzipped inputs
(key fields + answer + precomputed pron from the frozen coder) and the frozen anchor lists, plus
the code. Usage: uv run python character/scripts/kaggle_prep.py OUTDIR"""

import gzip
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from content_embed import anchors

R = HERE / "results/main_v1"
KEEP = ("model", "name_condition", "noun", "modifier", "thinking", "sample", "fixed_sample",
        "given_pron", "lang", "given_name", "answer", "finish_reason")  # fmt: skip


def main() -> None:
    out = Path(sys.argv[1])
    (out / "code").mkdir(parents=True, exist_ok=True)
    (out / "anchors.json").write_text(json.dumps({k: anchors(k) for k in ("properties", "people")}))
    for kind in ("writer", "character"):
        for f in ("gpt-6-luna", "qwen3-14b_off", "qwen3-14b_on"):
            s = pd.read_csv(R / f"{kind}_{f}_summary_coded.csv", keep_default_na=False,
                            usecols=["noun", "modifier", "thinking", "sample", "pron"])  # fmt: skip
            pron = {
                (r.noun, r.modifier, str(r.thinking), int(r.sample)): r.pron for r in s.itertuples()
            }
            n = miss = 0
            with (
                open(R / f"{kind}_{f}.jsonl") as fh,
                gzip.open(out / f"{kind}_{f}.jsonl.gz", "wt") as o,
            ):
                for ln in fh:
                    r = json.loads(ln)
                    if r.get("finish_reason") == "error" or not r.get("answer"):
                        continue
                    p = pron.get((r["noun"], r["modifier"], str(r["thinking"]), int(r["sample"])))
                    if p is None:
                        miss += 1
                        continue
                    o.write(json.dumps({k: r.get(k) for k in KEEP} | {"pron": p}) + "\n")
                    n += 1
            print(f"{kind} {f}: {n} rows, {miss} without a coded pron", flush=True)
    for c in ("content_embed.py", "code_outputs.py", "content_glasgow.py"):
        shutil.copy(HERE / c, out / "code" / c)
    meta = {"title": "gbleed character outputs", "id": "cyruseosphoros/gbleed-character-outputs",
            "licenses": [{"name": "other"}]}  # fmt: skip
    (out / "dataset-metadata.json").write_text(json.dumps(meta))
    print("=== KAGGLE PREP DONE")


if __name__ == "__main__":
    main()
