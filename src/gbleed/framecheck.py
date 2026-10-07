"""Behavioural check of the Phase 2 homograph verb frame (`siempre {w}`), on two model families.

Question: does the frame select the verb reading? For each word, the frame effect is
    s(w) = log P(w | "Bueno, siempre") - log P(w | "Bueno, mi")
i.e. how much more likely the word is after "always" than after "my". A frame that licenses verbs
and not bare nouns should raise s for homographs (also verb forms) relative to regular nouns that
have no verb reading (Phase 2 regular set; `eligible()` excludes any other part of speech).
Compared within ending (-o = 1sg, -a = 3sg), so the frame must work for both persons; a frame
that worked for only one would bring back a weaker form of the yo/usted confound.

Pass (pre-registered, decisions.md): for every model and both endings, mean s(homograph) minus
mean s(regular) > 0 with the bootstrap 95% CI lower bound > 0. Gender labels are not read; the
ending split is the person split by construction.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .models import load_model, model_slug, pick_device, pick_dtype, run_metadata, stage
from .phase2 import FINAL
from .scoring import continuation_logprobs_batch

LEAD = "Bueno,"
FRAMES = {"verb": "siempre", "noun": "mi"}
N_BOOT = 2000


def items() -> pd.DataFrame:
    df = pd.read_csv(FINAL, keep_default_na=False)
    df = df[df.set.isin(["homograph", "regular"])][["lemma", "set"]].copy()
    df["ending"] = df.lemma.str[-1]
    return df[df.ending.isin(["o", "a"])].reset_index(drop=True)


def score(model_id: str, device=None, dtype=None) -> pd.DataFrame:
    df = items()
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = load_model(model_id, dev, dt)
    meta = run_metadata(model, model_id, dev, dt)
    meta["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    for name, frame in FRAMES.items():
        stage(f"{model_id}: scoring {len(df)} words after '{LEAD} {frame}'")
        reqs = [(f"{LEAD} {frame}", " " + w) for w in df.lemma]
        df[f"lp_{name}"] = continuation_logprobs_batch(model, tok, reqs, desc=name)
    df["s"] = df.lp_verb - df.lp_noun
    del model
    return df, meta


def _boot_diff(a, b, rng) -> tuple[float, float, float]:
    d = a.mean() - b.mean()
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d), float(lo), float(hi)


def run(model_ids: list[str], out_root: str = "results/framecheck", device=None, dtype=None):
    rng = np.random.default_rng(0)
    verdicts = []
    for m in model_ids:
        df, meta = score(m, device, dtype)
        out = Path(out_root) / model_slug(m)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "scores.csv", index=False)
        res = {}
        for e, g in df.groupby("ending"):
            h, r = g[g.set == "homograph"].s.to_numpy(), g[g.set == "regular"].s.to_numpy()
            d, lo, hi = _boot_diff(h, r, rng)
            res[e] = {"n_homograph": len(h), "n_regular": len(r), "diff": round(d, 3),
                      "ci": [round(lo, 3), round(hi, 3)], "pass": bool(lo > 0)}  # fmt: skip
            verdicts.append(lo > 0)
            print(
                f"{m} -{e}: homograph - regular frame effect {d:+.2f} nats [{lo:+.2f}, {hi:+.2f}]"
            )
        (out / "summary.json").write_text(json.dumps({"meta": meta, "by_ending": res}, indent=2))
    verdict = "PASS" if all(verdicts) else "FAIL"
    print(f"frame check: {sum(verdicts)}/{len(verdicts)} model x ending cells pass -> {verdict}")
    return verdict
