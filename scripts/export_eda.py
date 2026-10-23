"""Export a small, word-free, CC0-safe dataset for exploratory data visualisation.

Writes eda/: noun_funnel.csv, nouns.csv, layers_wide.csv (+ README.md, written by hand).
Only values computed in this project are exported: no words, glosses, frequencies (wordfreq is
CC BY-SA) or rating norms. Grammatical gender, animacy and loan status are labels; endings are
spelling facts; token counts come from the Qwen3 tokenizer.

    uv run python scripts/export_eda.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from gbleed.lexicon import load_lexicon
from gbleed.phase2_stimuli import PREDICTIVE

OUT = Path("eda")
RES = Path("worktrees/gender-pod/results")  # pod runs: all Qwen3 sizes
MODELS = {"Qwen3-0.6B": 0.6, "Qwen3-1.7B": 1.7, "Qwen3-4B": 4.0, "Qwen3-8B": 8.0,
          "Qwen3-14B": 14.0, "Qwen3-30B-A3B": 30.0}  # fmt: skip
STEPS = [  # (name, column test) applied in order; each removes what fails it
    ("not single-gender", lambda d: d.gender.isin(["m", "f", "n"]) & (d.genders == d.gender)),
    ("animate (person/animal)", lambda d: d.animacy == "inanimate"),
    ("marked (slang or archaic)", lambda d: d.marked == ""),
    ("regional only", lambda d: d.regions == ""),
    ("also another part of speech", lambda d: d.also_pos == ""),
    ("also a form of another word", lambda d: d.also_form == ""),
    ("commoner in English", lambda d: d.zipf_en < d.zipf),
    ("too rare", lambda d: d.zipf >= 2.0),
]


def _first_fail(lex: pd.DataFrame) -> pd.Series:
    reason = pd.Series("", index=lex.index)
    for name, test in STEPS:
        reason[(reason == "") & ~test(lex)] = name
    return reason


def _training(lang: str) -> set[str]:
    f, s = {"es": ("phase2_final_v4", "strat"), "de": ("phase3_final_v3", "strat3")}[lang]
    d = pd.read_csv(f"data/stimuli/{f}.csv", keep_default_na=False)
    return set(d[d.set == s].lemma)


RENAME = {
    "compound": "compound (gender = last part)",
    "Ge-": "Ge- collective prefix",
    "phase 4-5 held out": "reserved for a later test",
    "phase 0 shot": "used as a quiz example",
    "animate (gloss)": "animate (from definition)",
    "animate (wordnet)": "animate (WordNet)",
    "group (wordnet)": "group noun (WordNet)",
}


def _german_reasons(lex: pd.DataFrame, kept: pd.Series) -> pd.Series:
    """German-specific exclusions (compounds, agent nouns, suffixes…), first reason only."""
    from gbleed import phase3_stimuli as p3

    base = lex[kept]
    why = p3.exclusions(base, set(lex.lemma.str.lower()), p3.german_verbs())
    why = why.str.split("; ").str[0].replace(RENAME)
    suffix = (why == "") & (base.de_suffix != "")
    why[suffix] = "gender-predicting suffix"
    return why


def nouns() -> pd.DataFrame:
    rows = []
    for lang in ("es", "de"):
        lex = load_lexicon(lang).reset_index(drop=True)
        lex = lex[lex.gender.isin(["m", "f", "n"]) | (lex.genders != "")]
        reason = _first_fail(lex)
        ok = reason == ""
        if lang == "es":
            reason[ok & lex.lemma.str.contains(PREDICTIVE)] = "gender-predicting ending"
            reason[ok & (reason == "") & lex.initial_a_f.astype(str).eq("True")] = (
                "feminine with initial stressed a-"
            )
        else:
            reason[ok] = _german_reasons(lex, ok).reindex(lex.index[ok]).values
        train = _training(lang)
        w = lex.lemma.str.lower()
        rows.append(pd.DataFrame({
            "lang": {"es": "Spanish", "de": "German"}[lang],
            "gender": lex.gender.where(lex.gender.isin(["m", "f", "n"]), "several"),
            "ending1": w.str[-1:], "ending2": w.str[-2:], "length": w.str.len(),
            "n_tokens": lex.n_tokens_qwen3, "animate": lex.animacy != "inanimate",
            "drop_reason": reason.where(reason != "", "kept"),
            "in_training_set": lex.lemma.isin(train),
        }))  # fmt: skip
    d = pd.concat(rows, ignore_index=True)
    d = d.sample(frac=1, random_state=0).reset_index(drop=True)  # no dictionary order
    d.insert(0, "noun_id", np.arange(1, len(d) + 1))
    return d


def funnel(n: pd.DataFrame) -> pd.DataFrame:
    out = []
    order = ["start"] + [s for s, _ in STEPS] + [
        "gender-predicting ending", "feminine with initial stressed a-",
        "gender-predicting suffix"] + sorted(set(n.drop_reason) - {s for s, _ in STEPS} - {
        "kept", "gender-predicting ending", "feminine with initial stressed a-",
        "gender-predicting suffix"})  # fmt: skip
    for lang, g in n.groupby("lang"):
        left = len(g)
        out.append((lang, 0, "start", left, 0))
        for i, step in enumerate(order[1:], 1):
            k = int((g.drop_reason == step).sum())
            if k:
                left -= k
                out.append((lang, i, step, left, k))
        out.append(
            (
                lang,
                len(order),
                "final training set (model knows it; cell has both genders)",
                int(g.in_training_set.sum()),
                left - int(g.in_training_set.sum()),
            )
        )
    d = pd.DataFrame(out, columns=["lang", "step_order", "step", "n_left", "n_removed"])
    d["step_order"] = d.groupby("lang").cumcount() + 1
    return d


def layers() -> pd.DataFrame:
    tests = [("phase2", "Spanish", "exceptions (el día…)", "exc_auc", "probe_strat"),
             ("phase2", "Spanish", "homographs, noun reading", "homo_auc_noun", "probe_strat"),
             ("phase3", "German", "compounds: head's gender", "head_auc", None)]  # fmt: skip
    rows = []
    for model, size in MODELS.items():
        for phase, lang, test, col, direction in tests:
            parts = {}
            for pos, sub in (("last", phase), ("after", f"{phase}_after")):
                f = RES / sub / f"Qwen__{model}-Base" / "layers.csv"
                if not f.exists():
                    continue
                t = pd.read_csv(f)
                if direction:
                    t = t[t.direction == direction]
                parts[pos] = t.set_index("layer")[[col, f"{col}_lo", f"{col}_hi"]]
            if "last" not in parts:
                continue
            d = parts["last"].set_axis(["auc_last", "lo_last", "hi_last"], axis=1)
            if "after" in parts:
                d = d.join(parts["after"].set_axis(["auc_after", "lo_after", "hi_after"], axis=1))
            d = d.reset_index()
            d.insert(0, "test", test)
            d.insert(0, "lang", lang)
            d.insert(0, "params_b", size)
            d.insert(0, "model", model)
            d.insert(5, "depth", (d.layer / d.layer.max()).round(3))
            rows.append(d)
    return pd.concat(rows, ignore_index=True).round(4)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    n = nouns()
    n.to_csv(OUT / "nouns.csv", index=False)
    funnel(n).to_csv(OUT / "noun_funnel.csv", index=False)
    layers().to_csv(OUT / "layers_wide.csv", index=False)
    for f in sorted(OUT.glob("*.csv")):
        print(f, len(pd.read_csv(f)), "rows")


if __name__ == "__main__":
    main()
