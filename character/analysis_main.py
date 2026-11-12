"""Confirmatory analysis of the character-study main run (prereg v1 + amendments 1-4).

Written and tested on synthetic data BEFORE it is run on the main outputs. Every function takes
coded-summary DataFrames, so the tests (`tests/test_analysis_main.py`) feed planted effects.

Units: percentage points of she-share among she + he outputs (they / it reported separately).
Cells (amendment 2): Luna medium, Luna none, Qwen3-14B off, Qwen3-14B on.
Frames (Q5): the primary Q1/Q2 estimate averages the two frames per noun (writer, consider); each
frame is also reported on its own.

Q1 writer: per noun x gendered language (de, ru, es, hi, ar) shift = she(foreign arm) -
  she(English-word arm); foreign arm = citation form for de/es (amendment 1), bare word for ru/hi/ar.
  Writer frame English arm = writer|<lg>:english; Consider frame English arm = consider|english.
  Effect = feminine-minus-masculine difference in shift = 2 x coef of code (+1 f, -1 m) in
  shift ~ code + language, noun-clustered (CR1) SEs, t with (nouns - 1) df. Neuter separate.
  Robustness: permutation test shuffling gender codes within language (2,000 draws).
Q2 character: per noun d = she(modifier) - she(baseline); baseline = plain_the / plain_your /
  plain_my for "the / your / my ..." phrases, else plain a/an. Mean over nouns, t CI. Antonym
  pairs: d = she(m1) - she(m2). Wugs: (a) mean of the six wugs vs plain a/an, (b) miryal - sputul
  (estimate only). frindle never analysed.
Q5 frame: per noun, mean over arms/phrases of she(writer frame) - she(consider frame); writer
  experiment over the foreign + English arms present in both frames, character over all phrases.
Gatekeeping (amendment 2): headline = four-cell average (per noun, or noun x language, mean of the
  cells' values; rows present in all four), alpha .05 two-sided; only if it passes, each cell
  Holm-corrected over the four. Q2 families (modifiers, pairs) are Holm-corrected across members
  at the headline level first. Decision: "effect" = 95% CI excludes 0 (sign reported);
  "no effect of interest" = 90% CI inside +-5 points; else "inconclusive".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

GLANGS = ("de", "ru", "es", "hi", "ar")
SESOI = 5.0
WUGS = ("miryal", "tisbal", "gundle", "sloke", "distock", "sputul")


# ---------- basic tables ----------
def she_table(d: pd.DataFrame) -> pd.Series:
    """she / (she + he) per (noun, modifier) among valid outputs."""
    d = d[(d.valid.astype(str) == "True") & d.pron.isin(["she", "he"])]
    return d.groupby(["noun", "modifier"]).pron.apply(lambda s: (s == "she").mean())


def foreign_arm(frame: str, lg: str) -> str:
    return f"{frame}|{lg}:{'citation' if lg in ('de', 'es') else 'foreign'}"


def english_arm(frame: str, lg: str) -> str:
    return f"{frame}|{lg}:english" if frame == "writer" else "consider|english"


def writer_shifts(d: pd.DataFrame, genders: dict, frames=("writer", "consider")) -> pd.DataFrame:
    """Rows noun, lang, g (f/m/n), shift (proportion), averaged over the frames present."""
    ss = she_table(d)
    rows = []
    for noun in sorted(d.noun.unique()):
        for lg in GLANGS:
            g = genders.get((noun, lg))
            if g not in ("f", "m", "n"):
                continue
            vals = [ss[(noun, foreign_arm(fr, lg))] - ss[(noun, english_arm(fr, lg))]
                    for fr in frames
                    if (noun, foreign_arm(fr, lg)) in ss and (noun, english_arm(fr, lg)) in ss]  # fmt: skip
            if vals:
                rows.append({"noun": noun, "lang": lg, "g": g, "shift": float(np.mean(vals))})
    return pd.DataFrame(rows)


# ---------- estimators ----------
def _design(df: pd.DataFrame) -> np.ndarray:
    X = pd.get_dummies(df.lang, drop_first=True, dtype=float)
    X.insert(0, "code", np.where(df.g == "f", 1.0, -1.0))
    X.insert(0, "const", 1.0)
    return X.to_numpy()


def _coef_code(X: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def q1_fit(sh: pd.DataFrame, perm: int = 0, seed: int = 0) -> dict:
    """Feminine-minus-masculine shift difference (points) with noun-clustered CR1 SE."""
    df = sh[sh.g.isin(["f", "m"])].reset_index(drop=True)
    X, y = _design(df), df["shift"].to_numpy()
    XtX = np.linalg.inv(X.T @ X)
    b = XtX @ X.T @ y
    e = y - X @ b
    meat = np.zeros((X.shape[1],) * 2)
    for n in df.noun.unique():
        m = (df.noun == n).to_numpy()
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    k, N, p = df.noun.nunique(), len(y), X.shape[1]
    V = XtX @ meat @ XtX * k / (k - 1) * (N - 1) / (N - p)
    out = summarise(100 * 2 * b[1], 100 * 2 * np.sqrt(V[1, 1]), k - 1)
    out["n_rows"], out["n_nouns"] = N, k
    if perm:
        rng = np.random.default_rng(seed)
        obs, hits = b[1], 0
        codes = df.g.to_numpy()
        for _ in range(perm):
            sh_g = codes.copy()
            for lg in df.lang.unique():
                idx = np.nonzero((df.lang == lg).to_numpy())[0]
                sh_g[idx] = rng.permutation(sh_g[idx])
            Xp = _design(df.assign(g=sh_g))
            hits += abs(_coef_code(Xp, y)) >= abs(obs) - 1e-12
        out["p_perm"] = (hits + 1) / (perm + 1)
    return out


def mean_fit(x: pd.Series) -> dict:
    """Mean over nouns (points), t CI."""
    x = x.dropna()
    out = summarise(100 * x.mean(), 100 * x.std(ddof=1) / np.sqrt(len(x)), len(x) - 1)
    out["n_nouns"] = len(x)
    return out


def summarise(est: float, se: float, dfree: int) -> dict:
    t95, t90 = stats.t.ppf(0.975, dfree), stats.t.ppf(0.95, dfree)
    p = 2 * stats.t.sf(abs(est / se), dfree) if se > 0 else float("nan")
    return {"est": est, "se": se, "lo95": est - t95 * se, "hi95": est + t95 * se,
            "lo90": est - t90 * se, "hi90": est + t90 * se, "p": p}  # fmt: skip


def decide(r: dict, sig: bool) -> str:
    if sig:
        return "effect (+)" if r["est"] > 0 else "effect (-)"
    if -SESOI < r["lo90"] and r["hi90"] < SESOI:
        return "no effect of interest"
    return "inconclusive"


def holm(ps: list[float], alpha: float = 0.05) -> list[bool]:
    order = np.argsort(ps)
    rej = [False] * len(ps)
    for k, i in enumerate(order):
        if ps[i] < alpha / (len(ps) - k):
            rej[i] = True
        else:
            break
    return rej


# ---------- gatekeeping ----------
def gatekeep(cells: dict[str, pd.Series | pd.DataFrame], fit, headline_ok: bool | None = None):
    """cells: name -> per-unit values (Series indexed by unit) or shift frames (Q1).
    Returns (headline_result, {cell: result}) with 'decision' fields filled."""
    if isinstance(next(iter(cells.values())), pd.DataFrame):  # Q1 shift frames
        key = ["noun", "lang"]
        idx = None
        for v in cells.values():
            s = v.set_index(key)["shift"]
            idx = s.index if idx is None else idx.intersection(s.index)
        base = next(iter(cells.values())).set_index(key).loc[idx]
        avg = pd.concat([v.set_index(key).loc[idx, "shift"] for v in cells.values()], axis=1).mean(
            1
        )
        head = fit(base.assign(shift=avg).reset_index())
    else:
        avg = pd.concat(list(cells.values()), axis=1).dropna().mean(1)
        head = fit(avg)
    passed = head["p"] < 0.05 if headline_ok is None else headline_ok
    head["decision"] = decide(head, passed)
    res = {c: fit(v) for c, v in cells.items()}
    rej = holm([r["p"] for r in res.values()]) if passed else [False] * len(res)
    for (c, r), s in zip(res.items(), rej):
        r["decision"] = decide(r, s)
    return head, res


# ---------- Q2 / Q5 helpers ----------
def base_of(phrase: str) -> str | None:
    if phrase == "plain_aan" or phrase.startswith("wug_"):
        return None if phrase == "plain_aan" else "plain_aan"
    for w in ("the", "your", "my"):
        if phrase.startswith(w + " "):
            return f"plain_{w}"
    return "plain_aan"


def noun_diffs(ss: pd.Series, a: str, b: str, frames=("writer", "consider")) -> pd.Series:
    """Per noun, mean over frames of she(frame|a) - she(frame|b)."""
    t = ss.unstack()
    cols = [(f"{fr}|{a}", f"{fr}|{b}") for fr in frames if f"{fr}|{a}" in t and f"{fr}|{b}" in t]
    return (
        pd.concat([t[x] - t[y] for x, y in cols], axis=1).mean(1)
        if cols
        else pd.Series(dtype=float)
    )


def frame_diffs(ss: pd.Series, phrases: list[str]) -> pd.Series:
    t = ss.unstack()
    cols = [t[f"writer|{p}"] - t[f"consider|{p}"] for p in phrases
            if f"writer|{p}" in t and f"consider|{p}" in t]  # fmt: skip
    return pd.concat(cols, axis=1).mean(1)
