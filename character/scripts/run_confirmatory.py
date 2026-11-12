"""Run the confirmatory analysis (prereg v1 §5, amendments 1-4) on the main-run coded summaries.
Writes character/results/confirmatory/*.csv and prints the decision tables.
Usage: uv run python character/scripts/run_confirmatory.py [--perm 2000]
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import analysis_main as am

# CONF_IN / CONF_OUT override the folders (synthetic dry runs only)
R = Path(os.environ.get("CONF_IN", HERE / "results/main_v1"))
OUT = Path(os.environ.get("CONF_OUT", HERE / "results/confirmatory"))
CELLS = {"Luna medium": ("gpt-6-luna", True), "Luna none": ("gpt-6-luna", False),
         "Qwen off": ("qwen3-14b_off", False), "Qwen on": ("qwen3-14b_on", True)}  # fmt: skip
PAIRS = [("an interesting", "a boring"), ("the best", "the worst"),
         ("your favorite", "your least favorite"), ("an ideal", "the ideal"),
         ("a traditional", "a modern"), ("a large", "a small"), ("a new", "an old"),
         ("a deliberate", "an accidental"), ("a popular", "an unpopular"),
         ("a famous", "an obscure"), ("a random", "a specific"), ("a real", "an imaginary"),
         ("a familiar", "an unfamiliar"), ("a rough", "a smooth"),
         ("familiar to you", "unfamiliar to you")]  # fmt: skip


def load(kind: str, f: str, th: bool) -> pd.DataFrame:
    d = pd.read_csv(R / f"{kind}_{f}_summary_coded.csv", keep_default_na=False)
    return d[d.thinking.astype(str) == str(th)]


def row(name: str, r: dict) -> dict:
    return {"test": name} | {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=2000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    G = pd.read_csv(HERE / "stimuli/main_writer_v1.csv", keep_default_na=False)
    gend = {(r.en, r.lang): r.gender for r in G.itertuples()}
    out = []

    # Q1 writer: headline + cells (frames averaged), then per frame, per language, robustness
    wr = {c: load("writer", f, th) for c, (f, th) in CELLS.items()}
    sh = {c: am.writer_shifts(d, gend) for c, d in wr.items()}
    head, cells = am.gatekeep(sh, am.q1_fit)
    out.append(row("Q1 headline (4-cell avg)", head))
    out += [row(f"Q1 {c}", r) for c, r in cells.items()]
    for c, s in sh.items():
        p = am.q1_fit(s, perm=a.perm)
        out.append(row(f"Q1 robustness permutation {c}", {"p_perm": p["p_perm"]}))
        for fr in ("writer", "consider"):
            out.append(row(f"Q1 frame={fr} {c}", am.q1_fit(am.writer_shifts(wr[c], gend, (fr,)))))
        for lg in am.GLANGS:
            sl = s[(s.lang == lg) & s.g.isin(["f", "m"])]
            if sl.g.nunique() == 2:
                out.append(row(f"Q1 lang={lg} {c} (estimate)", am.q1_fit(sl)))
        n = s[s.g == "n"]
        if len(n):
            out.append(
                row(
                    f"Q1 neuter mean shift {c} (estimate)",
                    am.mean_fit(n.groupby("noun")["shift"].mean()),
                )
            )

    # Q2 character: each modifier vs baseline; pairs; wugs. Headline per member, Holm across.
    ch = {c: am.she_table(load("character", f, th)) for c, (f, th) in CELLS.items()}
    phrases = sorted({m.split("|", 1)[1] for m in ch["Luna medium"].index.get_level_values(1)})
    mods = [
        p
        for p in phrases
        if not p.startswith("plain_") and p != "wug_frindle" and not p.startswith("wug_")
    ]
    fam = {}
    for p in mods:
        fam[f"mod {p} vs {am.base_of(p)}"] = {
            c: am.noun_diffs(ss, p, am.base_of(p)) for c, ss in ch.items()
        }
    for x, y in PAIRS:
        fam[f"pair {x} - {y}"] = {c: am.noun_diffs(ss, x, y) for c, ss in ch.items()}
    heads = {k: am.gatekeep(v, am.mean_fit, headline_ok=False) for k, v in fam.items()}
    for kind in ("mod", "pair"):
        ks = [k for k in fam if k.startswith(kind)]
        rej = am.holm([heads[k][0]["p"] for k in ks])
        for k, r in zip(ks, rej):
            h, cl = am.gatekeep(fam[k], am.mean_fit, headline_ok=r)
            out.append(row(f"Q2 {k} headline", h))
            out += [row(f"Q2 {k} {c}", v) for c, v in cl.items()]
    for c, ss in ch.items():
        w = pd.concat([am.noun_diffs(ss, f"wug_{x}", "plain_aan") for x in am.WUGS], axis=1).mean(1)
        out.append(row(f"Q2 wugs (6) vs plain {c}", am.mean_fit(w)))
        out.append(
            row(
                f"Q2 miryal - sputul {c} (estimate)",
                am.mean_fit(am.noun_diffs(ss, "wug_miryal", "wug_sputul")),
            )
        )

    # Q5 frame (writer - consider), gatekept like Q1/Q2
    q5w = {
        c: am.frame_diffs(
            am.she_table(d),
            [
                f"{lg}:{k}"
                for lg in ("de", "ru", "es", "hi", "ar", "fi", "ja", "tr")
                for k in ("foreign", "citation")
            ],
        )
        for c, d in wr.items()
    }
    q5c = {c: am.frame_diffs(ss, phrases) for c, ss in ch.items()}
    for lab, cells5 in (("writer experiment", q5w), ("character experiment", q5c)):
        h, cl = am.gatekeep(cells5, am.mean_fit)
        out.append(row(f"Q5 frame {lab} headline", h))
        out += [row(f"Q5 frame {lab} {c}", v) for c, v in cl.items()]

    df = pd.DataFrame(out)
    df.to_csv(OUT / "confirmatory.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 2000):
        cols = [
            c
            for c in ("test", "est", "lo95", "hi95", "p", "decision", "n_nouns", "p_perm")
            if c in df
        ]
        print(df[cols].to_string(index=False))
    print("=== CONFIRMATORY DONE")


if __name__ == "__main__":
    main()
