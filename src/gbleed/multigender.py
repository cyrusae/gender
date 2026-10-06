"""Acceptance check for same-spelling, two-gender nouns (der/die See, el/la mar).

Both articles are grammatical for these nouns, so Phase 0's "right article" test doesn't
apply. Instead: does the model accept both genders, i.e. stay close to indifferent between
the articles? Pre-registered (docs/decisions.md): both accepted = |margin| < 2.3 nats (10x)
in both Phase 0 frames. Typical margins for known single-gender nouns are 3-8 nats.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .models import load_model, model_slug, pick_device, pick_dtype, progress, run_metadata, stage
from .phase0 import score_noun

SPEC = "data/stimuli/multi_gender_spec.csv"
ACCEPT = 2.3  # nats


def run(
    model_id: str, out_root: str = "results/multigender", device=None, dtype=None
) -> pd.DataFrame:
    spec = pd.read_csv(SPEC, keep_default_na=False)
    spec = spec[spec.decision == "keep"].reset_index(drop=True)
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = load_model(model_id, dev, dt)
    meta = run_metadata(model, model_id, dev, dt)
    meta["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    rows = []
    stage(f"{model_id}: scoring {len(spec)} two-gender items")
    for r in progress(spec.itertuples(), total=len(spec), desc=model_id, unit="item"):
        s = score_noun(model, tok, r.lang, r.lemma)
        m1, m2 = s["ctx1_margin"], s["ctx2_margin"]
        both = abs(m1) < ACCEPT and abs(m2) < ACCEPT
        lean = "m" if (m1 + m2) > 0 else "f"
        rows.append({"lang": r.lang, "lemma": r.lemma, "type": r.type, "ctx1_margin": m1,
                     "ctx2_margin": m2, "both_accepted": both,
                     "prefers": "" if both else lean, "tokens": s["tokens"]})  # fmt: skip
    df = pd.DataFrame(rows)
    out = Path(out_root) / model_slug(model_id)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "items.csv", index=False)
    acc = df.groupby(["lang", "type"]).both_accepted.mean()
    summary = {
        "meta": meta,
        "threshold_nats": ACCEPT,
        "both_accepted": {f"{lang}/{t}": round(float(v), 3) for (lang, t), v in acc.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return df
