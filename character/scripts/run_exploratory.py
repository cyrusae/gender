"""Exploratory analyses after the confirmatory run (labelled exploratory; nothing here is a test of
a pre-registered hypothesis). 2026-11-11.

E1 noun-baseline follow-up for Q1 (Qwen's genderless-language control was not clean): for each noun,
   the genderless shift = mean foreign-minus-English shift over fi/ja/tr (frames averaged). Then
   (a) difference-in-differences: gendered-language shift minus that noun's genderless shift;
   (b) the genderless shift as a covariate in the Q1 regression.
   Both remove whatever a noun does under ANY foreign word, leaving the part tied to the shown
   word's grammatical gender.
E2 thinking (Q4): on - off per family for the Q1 effect (noun-clustered SE of the paired
   difference) and for pronoun-class shares (she / he / they / it) over all outputs.
E3 settings control (amendment 3): on the control subset, Qwen off vs off_thinkset vs on: pronoun
   shares and the writer persona-arm she-share; splits on - off into settings + mode.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import analysis_main as am
from run_confirmatory import CELLS, load

R, OUT = HERE / "results/main_v1", HERE / "results/exploratory"
NOGEN = ("fi", "ja", "tr")


def genderless_shift(d: pd.DataFrame) -> pd.Series:
    ss = am.she_table(d)
    out = {}
    for noun in d.noun.unique():
        v = [ss[(noun, f"{fr}|{lg}:foreign")] - ss[(noun, am.english_arm(fr, lg))]
             for lg in NOGEN for fr in ("writer", "consider")
             if (noun, f"{fr}|{lg}:foreign") in ss and (noun, am.english_arm(fr, lg)) in ss]  # fmt: skip
        if v:
            out[noun] = float(np.mean(v))
    return pd.Series(out)


def q1_with_cov(sh: pd.DataFrame, cov: pd.Series) -> dict:
    df = sh[sh.g.isin(["f", "m"])].copy()
    df["cov"] = df.noun.map(cov)
    df = df.dropna().reset_index(drop=True)
    X = pd.get_dummies(df.lang, drop_first=True, dtype=float)
    X.insert(0, "cov", df["cov"])
    X.insert(0, "code", np.where(df.g == "f", 1.0, -1.0))
    X.insert(0, "const", 1.0)
    X, y = X.to_numpy(), df["shift"].to_numpy()
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
    return am.summarise(100 * 2 * b[1], 100 * 2 * np.sqrt(V[1, 1]), k - 1)


def shares(d: pd.DataFrame) -> dict:
    v = d[d.valid.astype(str) == "True"].pron
    return {c: round(100 * (v == c).mean(), 1) for c in ("she", "he", "they", "it")} | {
        "n_valid": len(v)
    }


def fmt(r: dict) -> str:
    return f"{r['est']:+6.1f} [{r['lo95']:+6.1f}, {r['hi95']:+6.1f}]"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    G = pd.read_csv(HERE / "stimuli/main_writer_v1.csv", keep_default_na=False)
    gend = {(r.en, r.lang): r.gender for r in G.itertuples()}
    rows = []
    wr = {c: load("writer", f, th) for c, (f, th) in CELLS.items()}
    sh = {c: am.writer_shifts(d, gend) for c, d in wr.items()}

    print("E1 noun-baseline follow-up (Q1 effect, points)")
    for c in CELLS:
        gl = genderless_shift(wr[c])
        base = am.q1_fit(sh[c])
        did = sh[c].assign(shift=sh[c]["shift"] - sh[c].noun.map(gl)).dropna()
        a, b = am.q1_fit(did), q1_with_cov(sh[c], gl)
        print(
            f"  {c:12s} original {fmt(base)} | minus genderless shift {fmt(a)} | covariate {fmt(b)}"
        )
        rows += [{"analysis": "E1", "cell": c, "version": v} | r
                 for v, r in (("original", base), ("diff-in-diff", a), ("covariate", b))]  # fmt: skip

    print("E2 thinking: on - off")
    for fam, on, off in (("Luna", "Luna medium", "Luna none"), ("Qwen", "Qwen on", "Qwen off")):
        a = sh[on].set_index(["noun", "lang"])
        b = sh[off].set_index(["noun", "lang"])
        idx = a.index.intersection(b.index)
        dd = a.loc[idx].assign(shift=a.loc[idx, "shift"] - b.loc[idx, "shift"]).reset_index()
        r = am.q1_fit(dd)
        print(f"  {fam} Q1 effect on - off {fmt(r)}")
        rows.append({"analysis": "E2 Q1 on-off", "cell": fam} | r)
        for kind in ("writer", "character"):
            s_on = shares(load(kind, *CELLS[on]))
            s_off = shares(load(kind, *CELLS[off]))
            print(f"  {fam} {kind:9s} off {s_off}  on {s_on}")
            rows.append({"analysis": f"E2 shares {kind}", "cell": fam, "off": s_off, "on": s_on})

    print("E3 settings control (Qwen; control subset only)")
    wrx = r"^(?:writer\|plain|writer\|[a-z]+:english|consider\|english)$"
    for kind, rx in (("writer", wrx), ("character", r"\|plain_aan$")):
        ctrl = pd.read_csv(
            R / f"{kind}_qwen3-14b_offthinkset_summary_coded.csv", keep_default_na=False
        )
        parts = {"off (T0.7)": load(kind, "qwen3-14b_off", False), "off + thinking settings (T0.6)": ctrl,
                 "on (T0.6)": load(kind, "qwen3-14b_on", True)}  # fmt: skip
        for lab, d in parts.items():
            s = shares(d[d.modifier.str.contains(rx)])
            print(f"  {kind:9s} {lab:32s} {s}")
            rows.append({"analysis": f"E3 {kind}", "cell": lab} | s)
    pd.DataFrame(rows).to_csv(OUT / "exploratory.csv", index=False)
    print("=== EXPLORATORY DONE")


if __name__ == "__main__":
    main()
