"""Exploratory content analyses (see content_main.py). Score = content score per paragraph
(Glasgow word average: 1 = feminine ... 7 = masculine; embedders: feminine - masculine anchor
cosine, positive = feminine). Per cell:
V  validity: she-paragraph minus he-paragraph score, within noun x arm (does the scorer see gender?)
Q1c writer: per noun x language, content shift = score(foreign arm) - score(English arm), frames
    averaged; feminine-minus-masculine difference (noun-clustered, as Q1). Versions: all
    paragraphs; she-only; he-only (does the description move beyond the pronoun?).
Q2c character (Glasgow only, modifier words masked): each modifier's content shift vs its
    baseline, correlated across the 70 modifiers with its pronoun effect (C2).
Usage: content_analysis.py SCORE_SOURCE   (glasgow | emb_<name>)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import analysis_main as am

R, C = HERE / "results/main_v1", HERE / "results/content"
CELLS = {"Luna medium": ("gpt-6-luna", True), "Luna none": ("gpt-6-luna", False),
         "Qwen off": ("qwen3-14b_off", False), "Qwen on": ("qwen3-14b_on", True)}  # fmt: skip
EMB_CELL = {
    "Luna medium": "luna_medium",
    "Luna none": "luna_none",
    "Qwen off": "qwen_off",
    "Qwen on": "qwen_on",
}
K = ["noun", "modifier", "thinking", "sample"]


def merged(kind: str, cell: str, src: str) -> pd.DataFrame:
    f, th = CELLS[cell]
    d = pd.read_csv(
        R / f"{kind}_{f}_summary_coded.csv", keep_default_na=False, usecols=K + ["valid", "pron"]
    )
    d = d[(d.thinking.astype(str) == str(th)) & (d.valid.astype(str) == "True")]
    if src == "glasgow":
        g = pd.read_csv(C / f"glasgow_{kind}_{f}.csv")
        g = g[(g.thinking.astype(str) == str(th)) & (g.glasgow_n_rated >= 10)]
        s = g.assign(score=g.glasgow_gend_mean)[K + ["score"]]
    else:  # embedder on the writer sample: primary = character-only masking (kept for he/she)
        a = pd.read_csv(C / f"emb_{src}.csv")
        b = pd.read_csv(C / f"emb_{src}_kept.csv")
        col = f"emb_{src}"
        m = a.merge(b[K + [col]], on=K, how="left", suffixes=("", "_k"))
        m["score"] = np.where(
            m.pron.isin(["she", "he"]) & m[col + "_k"].notna(), m[col + "_k"], m[col]
        )
        s = m[m.thinking.astype(str) == str(th)][K + ["score"]]
    for x in (d, s):
        x["thinking"] = x.thinking.astype(str)
        x["sample"] = x["sample"].astype(int)
    return d.merge(s, on=K)


def cl_mean(units: pd.Series) -> str:
    r = am.summarise(units.mean(), units.std(ddof=1) / np.sqrt(len(units)), len(units) - 1)
    return f"{r['est']:+.4f} [{r['lo95']:+.4f}, {r['hi95']:+.4f}]"


def q1c(d: pd.DataFrame, gend: dict, prons=None) -> str:
    if prons:
        d = d[d.pron.isin(prons)]
    t = d.groupby(["noun", "modifier"]).score.mean()
    rows = []
    for (noun, lg), g in gend.items():
        if g not in ("f", "m"):
            continue
        v = [t[(noun, am.foreign_arm(fr, lg))] - t[(noun, am.english_arm(fr, lg))]
             for fr in ("writer", "consider")
             if (noun, am.foreign_arm(fr, lg)) in t and (noun, am.english_arm(fr, lg)) in t]  # fmt: skip
        if v:
            rows.append({"noun": noun, "lang": lg, "g": g, "shift": float(np.mean(v))})
    r = am.q1_fit(pd.DataFrame(rows))
    return f"{r['est'] / 100:+.4f} [{r['lo95'] / 100:+.4f}, {r['hi95'] / 100:+.4f}] (nouns {r['n_nouns']})"


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "glasgow"
    G = pd.read_csv(HERE / "stimuli/main_writer_v1.csv", keep_default_na=False)
    gend = {(r.en, r.lang): r.gender for r in G.itertuples() if r.lang in am.GLANGS}
    conf = pd.read_csv(HERE / "results/confirmatory/confirmatory.csv")
    print(f"SOURCE {src}  (Glasgow: lower = more feminine; embedders: higher = more feminine)")
    for cell in CELLS:
        d = merged("writer", cell, src)
        # within noun, arms pooled (the embedder sample has one draw per noun x arm, so a
        # noun x arm cell never holds both a she and a he paragraph)
        t = d[d.pron.isin(["she", "he"])].groupby(["noun", "pron"]).score.mean().unstack()
        v = (t["she"] - t["he"]).dropna()
        print(f"{cell:12s} paragraph SD {d.score.std():.4f}")
        print(
            f"{cell:12s} V she-he {cl_mean(v)} | Q1c all {q1c(d, gend)} | she-only {q1c(d, gend, ['she'])} | he-only {q1c(d, gend, ['he'])}"
        )
        if src == "glasgow":
            c = merged("character", cell, src)
            ss = c.groupby(["noun", "modifier"]).score.mean()
            pts = []
            for k in conf.test[
                conf.test.str.startswith("Q2 mod") & conf.test.str.endswith(f" {cell}")
            ]:
                p = k[len("Q2 mod ") : -len(cell) - 1]
                mod, base = p.split(" vs ")
                diff = am.noun_diffs(ss, mod, base)
                pron = conf.loc[conf.test == k, "est"].iloc[0]
                pts.append((pron, diff.mean()))
            a = np.array(pts)
            print(
                f"{'':12s} Q2c: modifiers {len(a)}, corr(pronoun effect, Glasgow shift) = {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:+.2f}"
            )
    print("=== CONTENT ANALYSIS DONE")


if __name__ == "__main__":
    main()
