"""Phase 3 "known" check: three-way article frames for German (m/f/n).

The Phase 0 German frames can't separate masculine from neuter (dem and ein serve both), which
didn't matter while neuter was excluded. Phase 3 needs neuter as a reference class, so every
candidate is re-checked with frames where all three articles differ and none has another reading:
  frame 1  "Hier ist der/die/das X."       nominative definite; singular "ist" rules out reading
                                           "die" as plural
  frame 2  "Dort gibt es einen/eine/ein X." accusative indefinite ("es gibt" + accusative): all
                                           three forms differ, and indefinites have no plural
As in Phase 0 the noun (+ rest) is scored given the article, not the article itself. Per frame,
margin = log P(correct) - max over the two wrong articles; right if >= 1 nat, wrong if <= -1,
else unsure. Known = right in both frames. Chosen on grammatical grounds; checked on Qwen3 and
EuroLLM (pre-registered in decisions.md).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .models import load_model, model_slug, pick_device, pick_dtype, run_metadata, stage
from .phase0 import DEFAULT_MIN_MARGIN
from .phase3_stimuli import POOL
from .scoring import continuation_logprobs_batch

FRAMES = [
    ("Hier ist {art} {noun}.", {"m": "der", "f": "die", "n": "das"}),
    ("Dort gibt es {art} {noun}.", {"m": "einen", "f": "eine", "n": "ein"}),
]
GENDERS = ("m", "f", "n")
KNOWN_MODELS = ["Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base"]
FINAL = "data/stimuli/phase3_final_v1.csv"


def _items() -> pd.DataFrame:
    p = pd.read_csv(POOL, keep_default_na=False)
    return p[p.gender.isin(GENDERS)].reset_index(drop=True)


def score(model_id: str, device=None, dtype=None, out_root: str = "results/phase3_known"):
    df = _items()
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = load_model(model_id, dev, dt)
    meta = run_metadata(model, model_id, dev, dt)
    meta["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    reqs = []
    for w in df.lemma:
        for frame, arts in FRAMES:
            prefix, rest = frame.split("{noun}")
            reqs += [(prefix.format(art=arts[g]).rstrip(), " " + w + rest) for g in GENDERS]
    stage(f"{model_id}: three-way known check, {len(df)} nouns, {len(reqs)} scores")
    lp = np.array(continuation_logprobs_batch(model, tok, reqs, desc="phase3-known"))
    lp = lp.reshape(len(df), len(FRAMES), len(GENDERS))
    gi = df.gender.map({g: i for i, g in enumerate(GENDERS)}).to_numpy()
    statuses = []
    for k in range(len(FRAMES)):
        for j, g in enumerate(GENDERS):
            df[f"ctx{k + 1}_lp_{g}"] = lp[:, k, j]
        correct = lp[np.arange(len(df)), k, gi]
        wrong = np.where(np.eye(3, dtype=bool)[gi], -np.inf, lp[:, k, :])
        df[f"ctx{k + 1}_margin"] = correct - wrong.max(1)
        df[f"ctx{k + 1}_rival"] = [GENDERS[i] for i in wrong.argmax(1)]
        m = df[f"ctx{k + 1}_margin"]
        statuses.append(np.where(m >= DEFAULT_MIN_MARGIN, "right",
                                 np.where(m <= -DEFAULT_MIN_MARGIN, "wrong", "unsure")))  # fmt: skip
    s1, s2 = statuses
    df["status"] = np.select(
        [(s1 == "right") & (s2 == "right"), (s1 == "wrong") & (s2 == "wrong"),
         ((s1 == "right") & (s2 == "wrong")) | ((s1 == "wrong") & (s2 == "right"))],
        ["known", "wrong", "conflict"], "unsure",
    )  # fmt: skip
    out = Path(out_root) / model_slug(model_id)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "scores.csv", index=False)
    summ = {"meta": meta, "known_by_gender": df.groupby("gender").status.apply(
        lambda s: round(float((s == "known").mean()), 3)).to_dict(),
        "known_by_gender_zipf4": df[df.zipf >= 4].groupby("gender").status.apply(
        lambda s: round(float((s == "known").mean()), 3)).to_dict()}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(
        f"{model_id}: known by gender {summ['known_by_gender']}; Zipf>=4 {summ['known_by_gender_zipf4']}"
    )
    del model
    return df


def frame_check(model_ids: list[str], out_root: str = "results/phase3_known") -> str:
    """Pre-registered: every model x gender has >= 70% known among Zipf >= 4 nouns."""
    cells = []
    for m in model_ids:
        s = json.loads((Path(out_root) / model_slug(m) / "summary.json").read_text())
        cells += [(m, g, v) for g, v in s["known_by_gender_zipf4"].items()]
    bad = [c for c in cells if c[2] < 0.7]
    return "PASS" if not bad else "FAIL: " + ", ".join(f"{m} {g} {v:.2f}" for m, g, v in bad)
