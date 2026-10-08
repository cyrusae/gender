"""Russian "known" check: three-way agreement frames (m/f/n), pre-registered 2026-10-08.

Russian has no articles, so a gender-agreeing word comes *before* the noun, and (as in the German
and Spanish checks) the noun + rest is scored given each form:
  frame 1  "Вот наш/наша/наше X."          "Here is our X" (possessive; no other reading)
  frame 2  "Здесь есть новый/новая/новое X." "Here there is a new X" (adjective; present-tense
                                            есть has no gender)
Per frame, margin = log P(correct) - max over the two wrong forms; right if >= 1 nat, wrong if
<= -1, else unsure. Known = right in both frames. Neuter stays a wrong option even though the
-ь pool is m/f only.

Items: the -ь pool (phase4_ru) plus, for the frame check, common Russian nouns of all endings
(Zipf >= 4; single gender; inanimate by grammatical animacy and the gloss judge; unmarked;
declinable; not another part of speech). Frame check (as German): every model x gender >= 70%
known among the Zipf >= 4 nouns.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .lexicon import load_lexicon
from .models import load_model, model_slug, pick_device, pick_dtype, run_metadata, stage
from .phase0 import DEFAULT_MIN_MARGIN
from .phase4_ru import POOL
from .scoring import continuation_logprobs_batch

FRAMES = [
    ("Вот {art} {noun}.", {"m": "наш", "f": "наша", "n": "наше"}),
    ("Здесь есть {art} {noun}.", {"m": "новый", "f": "новая", "n": "новое"}),
]
GENDERS = ("m", "f", "n")
OUT_ROOT = "results/ru_known"


def items() -> pd.DataFrame:
    lex = load_lexicon("ru")
    common = lex[
        lex.gender.isin(GENDERS)
        & (lex.zipf >= 4)
        & (lex.ru_animacy == "inan")
        & (lex.animacy == "inanimate")
        & (lex.marked == "")
        & (lex.also_pos == "")
        & ~lex.ru_indecl.astype(str).eq("True")
    ][["lemma", "gender", "concept_en", "zipf", "ru_ending"]].assign(set="framecheck")
    pool = pd.read_csv(POOL, keep_default_na=False)
    pool = pool[pool.set == "strat_ru"][["lemma", "gender", "concept_en", "zipf", "cell"]]
    df = pd.concat([common, pool.assign(set="strat_ru")]).drop_duplicates("lemma", keep="last")
    df["zipf"] = pd.to_numeric(df.zipf)
    return df.reset_index(drop=True)


def score(model_id: str, device=None, dtype=None, out_root: str = OUT_ROOT) -> pd.DataFrame:
    df = items()
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
    stage(f"{model_id}: Russian three-way known check, {len(df)} nouns, {len(reqs)} scores")
    lp = np.array(continuation_logprobs_batch(model, tok, reqs, desc="ru-known"))
    lp = lp.reshape(len(df), len(FRAMES), len(GENDERS))
    gi = df.gender.map({g: i for i, g in enumerate(GENDERS)}).to_numpy()
    statuses = []
    for k in range(len(FRAMES)):
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
    fc = df[(df.set == "framecheck") & (df.zipf >= 4)]
    pool = df[df.set == "strat_ru"]
    summ = {
        "meta": meta,
        "framecheck_known_by_gender": fc.groupby("gender").status.apply(
            lambda s: round(float((s == "known").mean()), 3)).to_dict(),
        "framecheck_n_by_gender": fc.gender.value_counts().to_dict(),
        "pool_known_by_gender": pool.groupby("gender").status.apply(
            lambda s: round(float((s == "known").mean()), 3)).to_dict(),
    }  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(f"{model_id}: frame check {summ['framecheck_known_by_gender']} "
          f"(n {summ['framecheck_n_by_gender']}); -ь pool {summ['pool_known_by_gender']}")  # fmt: skip
    del model
    return df


def frame_check(model_ids: list[str], out_root: str = OUT_ROOT) -> str:
    """Pre-registered: every model x gender >= 70% known among the Zipf >= 4 frame-check nouns."""
    cells = []
    for m in model_ids:
        s = json.loads((Path(out_root) / model_slug(m) / "summary.json").read_text())
        cells += [(m, g, v) for g, v in s["framecheck_known_by_gender"].items()]
    bad = [c for c in cells if c[2] < 0.7]
    return "PASS" if not bad else "FAIL: " + ", ".join(f"{m} {g} {v:.2f}" for m, g, v in bad)
