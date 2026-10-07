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
from .phase2_stimuli import _clean
from .phase3_stimuli import POOL, SEED
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
    p = p[p.gender.isin(GENDERS)].reset_index(drop=True)
    p["zipf"] = pd.to_numeric(p.zipf)  # the multi rows (dropped here) leave the column as text
    return p


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


def _match(df: pd.DataFrame, genders, keys, name: str, nested: dict | None = None) -> pd.DataFrame:
    """Per cell of `keys`, the same number of nouns of each gender (seeded shuffle, so a smaller
    set built from the same shuffle is a prefix of a larger one)."""
    out = []
    for _, g in df.groupby(keys):
        by = {x: g[g.gender == x].sample(frac=1, random_state=SEED) for x in genders}
        k = min(len(v) for v in by.values())
        out += [v.iloc[:k] for v in by.values()]
    return pd.concat(out).assign(set=name) if out else df.iloc[:0].assign(set=name)


def _strata(df: pd.DataFrame, genders, keys, name: str) -> pd.DataFrame:
    """Every noun in the cells of `keys` that contain all `genders` (stratified comparison:
    gender is compared only within cells; cells missing a gender carry no information)."""
    ok = df.groupby(keys).gender.transform(lambda g: set(genders) <= set(g))
    return df[ok].assign(set=name)


def finalize(known_root: str = "results/phase3_known") -> pd.DataFrame:
    """Keep nouns known by every KNOWN_MODELS model; sample the training sets; freeze splits.

    strat3        primary: every noun in ending x loan cells that contain all of m, f and n;
                  analysed with the cells as fixed effects (within-cell comparison only)
    matched3      per ending x loan status: equal m, f, n (strict secondary; a subset of strat3)
    matched3_end  per ending only: equal m, f, n (secondary; loan status as a covariate)
    matched2      per ending x loan status: equal m, f; the same seeded shuffle as matched3, so it
                  contains matched3's m/f nouns
    compound_train  compounds (heads disjoint from compound_test), no gender-predicting suffix,
                  stratified like strat3 (by head ending x loan status)
    Test sets: suffix, multi, compound_test (known nouns only; multi has no single gender).
    No compound (train or test) has a head that is a simplex training noun.
    """
    pool = pd.read_csv(POOL, keep_default_na=False)
    pool["en_overlap"] = pool.en_overlap.astype(str).eq("True")
    known = None
    for m in KNOWN_MODELS:
        s = pd.read_csv(Path(known_root) / model_slug(m) / "scores.csv", keep_default_na=False)
        k = set(s[s.status == "known"].lemma)
        known = k if known is None else known & k
    ok = pool.lemma.isin(known)
    simplex = _clean(pool[(pool.set == "simplex") & ok])
    train = [
        _strata(simplex, GENDERS, ["ending", "loan"], "strat3"),
        _match(simplex, GENDERS, ["ending", "loan"], "matched3"),
        _match(simplex, GENDERS, ["ending"], "matched3_end"),
        _match(simplex, ("m", "f"), ["ending", "loan"], "matched2"),
    ]
    # A compound is read at its last token, which is roughly its head: a compound whose head is a
    # simplex training noun would partly re-measure that noun (test) or double-count it (train).
    trained = {w.lower() for t in train for w in t.lemma}
    head_ok = ~pool["head"].str.lower().isin(trained)
    n_drop = ((pool.set.str.startswith("compound")) & ok & ~head_ok).groupby(pool.set).sum()
    print("compounds dropped (head is a simplex training noun):", n_drop[n_drop > 0].to_dict())
    comp = _clean(pool[(pool.set == "compound_train") & ok & head_ok & (pool.de_suffix == "")])
    train.append(_strata(comp, GENDERS, ["ending", "loan"], "compound_train"))
    test = pool[(pool.set == "suffix") & ok | (pool.set == "compound_test") & ok & head_ok]
    test = pd.concat([test, pool[pool.set == "multi"]])
    df = pd.concat([*train, test], ignore_index=True)
    df["split"] = np.where(df.set.isin(["suffix", "compound_test", "multi"]), "test", "train")
    tr, te = set(df[df.split == "train"].lemma), set(df[df.split == "test"].lemma)
    assert not tr & te, tr & te
    heads_tr = set(df[df.set == "compound_train"]["head"])
    heads_te = set(df[df.set == "compound_test"]["head"])
    assert not heads_tr & heads_te
    simplex_tr = {
        w.lower() for w in df[df.set.isin(["strat3", "matched3", "matched3_end", "matched2"])].lemma
    }
    assert not {h.lower() for h in heads_tr | heads_te} & simplex_tr
    df.to_csv(FINAL, index=False)
    print(df.groupby(["set", "gender"]).size().unstack(fill_value=0).to_string())
    return df
