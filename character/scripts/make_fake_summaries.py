"""Synthetic coded summaries with the main run's nouns, arms and phrases (no model outputs), for a
dry run of run_confirmatory.py. Planted: writer f-minus-m shift +10 points in every cell; modifier
"a rough" -10 vs plain a/an; pair rough - smooth -10; frame writer - consider +4; all else 0.
Usage: uv run python character/scripts/make_fake_summaries.py OUTDIR"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
G = pd.read_csv(HERE / "stimuli/main_writer_v1.csv", keep_default_na=False)
gend = {(r.en, r.lang): r.gender for r in G.itertuples()}
nouns = pd.read_csv(HERE / "stimuli/main_nouns_v1.csv").iloc[:, 0].tolist()
C = pd.read_csv(HERE / "stimuli/main_character_v1.csv", keep_default_na=False)
phr = sorted(set(C.modifier)) + ["plain_aan", "plain_the", "plain_my", "plain_your"]
phr = sorted(set(phr))
LG = ("de", "ru", "es", "hi", "ar", "fi", "ja", "tr")
for f, th in (
    ("gpt-6-luna", True),
    ("gpt-6-luna", False),
    ("qwen3-14b_off", False),
    ("qwen3-14b_on", True),
):
    for kind in ("writer", "character"):
        rows = []
        for n in nouns:
            base = rng.uniform(0.3, 0.8)
            if kind == "writer":
                arms = [("writer|plain", base)]
                for lg in LG:
                    g = gend.get((n, lg))
                    eff = 0.05 * (1 if g == "f" else -1 if g == "m" else 0) + rng.normal(0, 0.1)
                    fa = f"{lg}:citation" if lg in ("de", "es") else f"{lg}:foreign"
                    arms += [(f"writer|{fa}", base + eff), (f"consider|{fa}", base + eff),
                             (f"writer|{lg}:english", base)]  # fmt: skip
                arms.append(("consider|english", base))
                draws = 10
            else:
                arms = []
                for p in phr:
                    m = p
                    d = -0.10 if p == "a rough" else 0.0
                    arms += [(f"writer|{m}", base + d), (f"consider|{m}", base + d)]
                draws = 2
            for arm, p in arms:
                p = np.clip(p + (0.04 if arm.startswith("writer|") else 0), 0.01, 0.99)
                for s in rng.random(draws) < p:
                    rows.append({"noun": n, "modifier": arm, "thinking": th, "valid": True,
                                 "pron": "she" if s else "he"})  # fmt: skip
        fn = (
            out / f"{kind}_{f}_summary_coded.csv"
        )  # Luna's two cells share one file, as in the real run
        pd.DataFrame(rows).to_csv(fn, index=False, mode="a", header=not fn.exists())
print("fake summaries written")
