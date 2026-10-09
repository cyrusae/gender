"""Gender wug test on the R-NONCE words (Phase 5 P19, adopted 2026-10-08; exploratory).

Do models assign Spanish gender to brand-new words by their ending? Phase 0's two article
frames, unchanged (margin = log P(word + rest | masculine) - log P(... | feminine), so > 0 =
masculine), on the Wuggy stems x -a/-o/-e/-iz. Stems beginning with a-/ha- are dropped:
feminine nouns with stressed initial a take el (el aula), so the frames are ambiguous there.

Prediction (lexicon rates, fixed before data): feminine share -a > -iz > -e > -o.
Pass rule (fixed before data): in both frames, >= 70% of -a forms prefer feminine and >= 70% of
-o forms prefer masculine.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .estimators import auc
from .nonce5 import ENDINGS, OUT_ES

PREDICTED = ["a", "iz", "e", "o"]  # most to least feminine
PASS_SHARE = 0.70
FRAMES = ("ctx1_margin", "ctx2_margin")


def items() -> pd.DataFrame:
    d = pd.read_csv(OUT_ES, keep_default_na=False)
    d = d[~d.stem.str.match(r"h?[aá]")]
    rows = [
        {"stem": r.stem, "ending": e, "lang": "es", "lemma": getattr(r, f"form_{e}")}
        for r in d.itertuples()
        for e in ENDINGS
    ]
    return pd.DataFrame(rows)


def run(model_id: str, out_root: str = "results/nonce_wug", device=None, dtype=None,
        batch_size: int = 32) -> Path:  # fmt: skip
    from .models import load_model, model_slug, pick_device, pick_dtype, run_metadata, stage
    from .phase0 import score_all

    df = items()
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    stage(f"Loading {model_id} on {dev} ({dt})")
    model, tok = load_model(model_id, dev, dt)
    meta = run_metadata(model, model_id, dev, dt)
    meta["stimuli"] = OUT_ES
    meta["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    stage(f"{model_id}: scoring {len(df)} nonce words")
    scored = pd.DataFrame(score_all(model, tok, df, batch_size))
    res = pd.concat([df.reset_index(drop=True), scored[[*FRAMES, "n_tokens", "tokens"]]], axis=1)
    out = Path(out_root) / model_slug(model_id)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "items.csv", index=False)
    summary = {"meta": meta, "results": analyze(res)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary["results"], indent=2, ensure_ascii=False))
    return out


def _ending_is_token(r) -> bool:
    return r.tokens.split("|")[-1].strip() == r.ending


def analyze(res: pd.DataFrame) -> dict:
    from scipy.stats import spearmanr

    out = {"n_stems": int(res.stem.nunique())}
    for f in FRAMES:
        share_f = {e: float((res[res.ending == e][f] < 0).mean()) for e in ENDINGS}
        mean_m = {e: float(res[res.ending == e][f].mean()) for e in ENDINGS}
        ao = res[res.ending.isin(["a", "o"])]
        rank_model = [share_f[e] for e in PREDICTED]
        rho = spearmanr(rank_model, [4, 3, 2, 1]).statistic
        out[f] = {
            "share_feminine": share_f,
            "mean_margin": mean_m,
            "auc_a_vs_o": float(auc((ao.ending == "o").to_numpy(int), ao[f].to_numpy())),
            "spearman_vs_predicted_order": float(rho),
        }
    out["passes"] = bool(
        all(
            out[f]["share_feminine"]["a"] >= PASS_SHARE
            and 1 - out[f]["share_feminine"]["o"] >= PASS_SHARE
            for f in FRAMES
        )
    )
    tok_sep = res.apply(_ending_is_token, axis=1)
    out["ending_own_token"] = {e: float(tok_sep[res.ending == e].mean()) for e in ENDINGS}
    out["mean_tokens"] = {e: float(res[res.ending == e].n_tokens.mean()) for e in ENDINGS}
    both = np.sign(res[FRAMES[0]]) == np.sign(res[FRAMES[1]])
    out["frames_agree"] = float(both.mean())
    return out
