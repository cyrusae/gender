"""Content measures on the main run (EXPLORATORY: the prereg froze content measures for Q3, which
is deferred, so nothing here is confirmatory). 2026-11-19.

Step 1 (this script, fast): Glasgow word average (human ratings, model-free) on EVERY valid output,
  writer and character studies. Masking as content_glasgow.py; for character outputs the modifier's
  own words are also masked (so "a malevolent door" can't score through the word "malevolent").
  -> results/content/glasgow_<file>.csv
Step 2 (this script): fixed sample for the transformer embedders, which run ~3.5 paragraphs/s on
  CPU: writer study, draw 0 only (draws are already random), Q1 arms only (foreign/citation arm
  of de/ru/es/hi/ar, the matching English-word arms, consider|english), all four cells.
  -> results/content/sample_<cell>.jsonl  (~12k outputs)
Step 3 (shell, see content_main.sh): content_embed.py on the samples, mask-all and --keep-they
  (character-only masking = primary, as in the pilot); fastText on all writer outputs.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from code_outputs import split_name
from content_glasgow import glasgow_score

R, OUT = HERE / "results/main_v1", HERE / "results/content"
FILES = ["gpt-6-luna", "qwen3-14b_off", "qwen3-14b_on"]
CELLS = {"luna_medium": ("gpt-6-luna", True), "luna_none": ("gpt-6-luna", False),
         "qwen_off": ("qwen3-14b_off", False), "qwen_on": ("qwen3-14b_on", True)}  # fmt: skip
Q1 = re.compile(
    r"^(writer|consider)\|((de|es):citation|(ru|hi|ar):foreign|(de|ru|es|hi|ar):english|english)$"
)


def modifier_words(mod: str) -> str:
    p = mod.split("|", 1)[-1]
    if p.startswith(("plain_", "wug_")):
        return p.removeprefix("wug_") if p.startswith("wug_") else ""
    return " ".join(
        w
        for w in re.findall(r"[a-z-]+", p)
        if w not in ("a", "an", "the", "your", "my", "to", "you")
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for kind in ("writer", "character"):
        for f in FILES:
            rows = []
            with open(R / f"{kind}_{f}.jsonl") as fh:
                for ln in fh:
                    r = json.loads(ln)
                    if r.get("finish_reason") != "stop":
                        continue
                    para, name = split_name(r.get("answer") or "")
                    extra = modifier_words(r["modifier"]) if kind == "character" else ""
                    g = glasgow_score(para, r["noun"], " ".join(filter(None, [name or "", extra])))
                    rows.append({"noun": r["noun"], "modifier": r["modifier"], "thinking": r["thinking"],
                                 "sample": r["sample"]} | g)  # fmt: skip
            pd.DataFrame(rows).to_csv(OUT / f"glasgow_{kind}_{f}.csv", index=False)
            print(f"glasgow {kind} {f}: {len(rows)}", flush=True)
    for cell, (f, th) in CELLS.items():
        n = 0
        with open(R / f"writer_{f}.jsonl") as fh, open(OUT / f"sample_{cell}.jsonl", "w") as o:
            for ln in fh:
                r = json.loads(ln)
                if (bool(r["thinking"]) == th and r["sample"] == 0 and r.get("finish_reason") == "stop"
                        and Q1.match(r["modifier"])):  # fmt: skip
                    o.write(json.dumps({k: r.get(k) for k in ("model", "name_condition", "noun",
                            "modifier", "thinking", "sample", "fixed_sample", "given_pron", "lang",
                            "given_name", "answer", "finish_reason")}) + "\n")  # fmt: skip
                    n += 1
        print(f"sample {cell}: {n}", flush=True)
    print("=== CONTENT STEP 1-2 DONE")


if __name__ == "__main__":
    main()
