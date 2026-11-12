"""Exploratory (amendment 3): style measures and thinking-leakage flag on the main-run outputs.
Style per output on the paragraph (Name line removed), averaged per noun, then on - off per family
as a paired mean over nouns (t CI). Leakage rate per cell. Writes results/exploratory/style_leak.csv.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import analysis_main as am
from code_outputs import split_name
from style_measures import leaked, style

R, OUT = HERE / "results/main_v1", HERE / "results/exploratory"
FILES = {("Luna", True): "gpt-6-luna", ("Luna", False): "gpt-6-luna",
         ("Qwen", False): "qwen3-14b_off", ("Qwen", True): "qwen3-14b_on"}  # fmt: skip
MET = ("n_words", "mean_sent_len", "ttr_100", "formal_markers", "list_like")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per = {}  # (fam, think, kind) -> DataFrame noun x metric means
    leak = []
    for (fam, th), f in FILES.items():
        for kind in ("writer", "character"):
            acc, n, nl = defaultdict(lambda: defaultdict(list)), 0, 0
            with open(R / f"{kind}_{f}.jsonl") as fh:
                for ln in fh:
                    r = json.loads(ln)
                    if bool(r["thinking"]) != th or r.get("finish_reason") != "stop":
                        continue
                    n += 1
                    nl += leaked(r.get("answer") or "")
                    for k, v in style(split_name(r.get("answer") or "")[0]).items():
                        acc[r["noun"]][k].append(v)
            per[(fam, th, kind)] = pd.DataFrame(
                {nn: {k: np.nanmean(v) for k, v in m.items()} for nn, m in acc.items()}
            ).T
            leak.append({"family": fam, "thinking": th, "experiment": kind, "n": n, "leak_flags": nl,
                         "leak_rate_%": round(100 * nl / n, 3)})  # fmt: skip
            print(f"{fam} thinking={th} {kind}: {n} outputs, leak flags {nl}", flush=True)
    rows = []
    for fam in ("Luna", "Qwen"):
        for kind in ("writer", "character"):
            on, off = per[(fam, True, kind)], per[(fam, False, kind)]
            for m in MET:
                r = am.mean_fit((on[m] - off[m]) / 100)  # mean_fit reports x100; undo
                rows.append({"family": fam, "experiment": kind, "metric": m,
                             "off_mean": round(off[m].mean(), 3), "on_mean": round(on[m].mean(), 3),
                             "on_minus_off": round(r["est"], 3), "lo95": round(r["lo95"], 3),
                             "hi95": round(r["hi95"], 3)})  # fmt: skip
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "style.csv", index=False)
    pd.DataFrame(leak).to_csv(OUT / "leakage.csv", index=False)
    with pd.option_context("display.width", 200):
        print(df.to_string(index=False))
        print(pd.DataFrame(leak).to_string(index=False))
    print("=== STYLE LEAK DONE")


if __name__ == "__main__":
    main()
