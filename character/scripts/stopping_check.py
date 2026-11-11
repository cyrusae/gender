"""Precision stopping rule (prereg v1 §6 + amendment 2): prints CI HALF-WIDTHS ONLY, never an
estimate, sign or p-value. Writer: pooled feminine-minus-masculine shift (foreign arm − English-word
arm; de/es citation form, ru/hi/ar bare), noun-clustered regression with language effects, per
model × thinking cell and for the four-cell average (gatekeeping headline). Target ≤ 2.5 points.
Character: each modifier vs its baseline (the/your/my modifiers vs plain_the/your/my, plain
the/my/your and everything else vs plain a/an), mean over nouns of the she-share difference;
median half-width per cell × frame. Target ≤ 5 points.
Usage: uv run python character/scripts/stopping_check.py   (reads results/main_v1/*_summary_coded.csv)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results/main_v1"
G = pd.read_csv(R.parents[1] / "stimuli/main_writer_v1.csv", keep_default_na=False)
GEND = {(r.en, r.lang): r.gender for r in G.itertuples()}
CELLS = [("gpt-6-luna", True, "Luna medium"), ("gpt-6-luna", False, "Luna none"),
         ("qwen3-14b_off", False, "Qwen off"), ("qwen3-14b_on", True, "Qwen on")]  # fmt: skip


def she_share(d: pd.DataFrame, keys: list[str]) -> pd.Series:
    d = d[(d.valid.astype(str) == "True") & d.pron.isin(["she", "he"])]
    g = d.groupby(keys).pron
    return g.apply(lambda s: (s == "she").mean())


def writer_shifts(f: str, think: bool) -> pd.DataFrame:
    d = pd.read_csv(R / f"writer_{f}_summary_coded.csv", keep_default_na=False)
    d = d[d.thinking.astype(str) == str(think)]
    ss = she_share(d, ["noun", "modifier"])
    rows = []
    for lg in ("de", "ru", "es", "hi", "ar"):
        fa = f"writer|{lg}:citation" if lg in ("de", "es") else f"writer|{lg}:foreign"
        for noun in d.noun.unique():
            g = GEND.get((noun, lg))
            if (
                g not in ("f", "m")
                or (noun, fa) not in ss
                or (noun, f"writer|{lg}:english") not in ss
            ):
                continue
            rows.append({"noun": noun, "lang": lg, "code": 1 if g == "f" else -1,
                         "shift": ss[(noun, fa)] - ss[(noun, f"writer|{lg}:english")]})  # fmt: skip
    return pd.DataFrame(rows)


def halfwidth(df: pd.DataFrame) -> float:
    """Half-width (points) of the 95% CI of 2*coef(code), noun-clustered (CR1)."""
    X = pd.get_dummies(df.lang, drop_first=True, dtype=float)
    X.insert(0, "code", df.code.astype(float))
    X.insert(0, "const", 1.0)
    X, y = X.to_numpy(), df["shift"].to_numpy()
    XtX = np.linalg.inv(X.T @ X)
    e = y - X @ (XtX @ X.T @ y)
    meat = np.zeros((X.shape[1],) * 2)
    for n in df.noun.unique():
        m = (df.noun == n).to_numpy()
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    k, N, p = df.noun.nunique(), len(y), X.shape[1]
    V = XtX @ meat @ XtX * k / (k - 1) * (N - 1) / (N - p)
    return 100 * 1.96 * 2 * np.sqrt(V[1, 1])


def base_of(mod: str) -> str | None:
    frame, m = mod.split("|", 1)
    if m == "plain_aan":
        return None
    for w in ("the", "your", "my"):
        if m.startswith(w + " "):
            return f"{frame}|plain_{w}"
    return f"{frame}|plain_aan"


def main() -> None:
    print("WRITER (target <= 2.5 points half-width)")
    shifts = {}
    for f, th, lab in CELLS:
        s = writer_shifts(f, th)
        shifts[lab] = s.set_index(["noun", "lang"])
        print(f"  {lab:12s} n noun×lang {len(s):4d}  half-width {halfwidth(s):5.2f}")
    avg = pd.concat([v["shift"] for v in shifts.values()], axis=1).dropna().mean(axis=1)
    a = shifts["Luna medium"].loc[avg.index, ["code"]].assign(shift=avg).reset_index()
    print(
        f"  {'4-cell avg':12s} n noun×lang {len(a):4d}  half-width {halfwidth(a):5.2f}  (headline)"
    )
    print("CHARACTER (target: median modifier half-width <= 5 points)")
    for f, th, lab in CELLS:
        d = pd.read_csv(R / f"character_{f}_summary_coded.csv", keep_default_na=False)
        ss = she_share(d[d.thinking.astype(str) == str(th)], ["noun", "modifier"]).unstack()
        for frame in ("writer", "consider"):
            hw = []
            for mod in [c for c in ss.columns if c.startswith(frame + "|")]:
                b = base_of(mod)
                if b is None or b not in ss:
                    continue
                diff = (ss[mod] - ss[b]).dropna()
                hw.append(100 * 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)))
            hw = np.array(hw)
            print(f"  {lab:12s} {frame:8s} modifiers {len(hw):3d}  median {np.median(hw):5.2f}  "
                  f"max {hw.max():5.2f}  share <=5: {np.mean(hw <= 5):.0%}")  # fmt: skip
    sys.stdout.flush()


if __name__ == "__main__":
    main()
