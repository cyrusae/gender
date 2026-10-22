"""`gbleed lang-audit`: playbook Stages 2, 3 and 7 in one report, before anyone reviews a list.

For a candidate noun list (or a language's eligible lexicon nouns), by gender:
  spelling   how well the last 1-3 letters predict gender (leave-one-out); the residual cue
             within the language's cells (last 3-4 letters vs the cell's majority); initial
             letters whose gender mix departs from the overall one
  balance    length, token count, frequency, age of acquisition and concreteness of the
             English gloss: per-gender means and the within-cell standardised difference
  structure  compound/derived-word flags by gender (if the language has a rule)
  coverage   glosses, and gloss coverage in the rating norms
Flags use fixed thresholds (below). They are prompts for a look, not tests: the PI's review of
the list (Stage 7) decides what to do about them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .languages import spec

OUT = Path("results/lang_audit")
D_FLAG = 0.3  # |standardised difference| between genders, within cells
INITIAL_MIN_N = 10  # initial letters with at least this many nouns are checked
INITIAL_FLAG = 0.25  # a gender's share departs from its overall share by this much
MISSING_EXPECTED = 2.0  # ... or a gender is absent where at least this many were expected
STRUCT_FLAG = 0.10  # share of complex nouns differs between genders by this much
CUE_FLAG = 0.05  # leave-one-out accuracy gain over the cell majority (residual spelling cue)


# ---- spelling ---------------------------------------------------------------------------
def loo_ending_accuracy(words, genders, k: int, base=None) -> float:
    """Leave-one-out accuracy of predicting gender from the last k letters (majority of the other
    words with that ending; no other word -> `base` (each word's fallback label), else the
    overall majority)."""
    words = [str(w).lower() for w in words]
    g = list(genders)
    tab: dict[str, Counter] = defaultdict(Counter)
    for w, x in zip(words, g, strict=True):
        tab[w[-k:]][x] += 1
    overall = Counter(g).most_common(1)[0][0]
    hit = 0
    for i, (w, x) in enumerate(zip(words, g, strict=True)):
        c = tab[w[-k:]].copy()
        c[x] -= 1
        c = +c
        if c:
            top = max(c.values())
            pred = min(k_ for k_, v in c.items() if v == top)
        else:
            pred = base[i] if base is not None else overall
        hit += pred == x
    return hit / len(g)


def loo_cell_majority(cells, genders) -> list[str]:
    """Each word's leave-one-out majority gender within its cell."""
    tab: dict[str, Counter] = defaultdict(Counter)
    for c, x in zip(cells, genders, strict=True):
        tab[c][x] += 1
    overall = Counter(genders).most_common(1)[0][0]
    out = []
    for c, x in zip(cells, genders, strict=True):
        t = tab[c].copy()
        t[x] -= 1
        t = +t
        out.append(min(k for k, v in t.items() if v == max(t.values())) if t else overall)
    return out


def _initial(w: str) -> str:
    """First letter, lowercased, accents folded (á -> a; Cyrillic й stays й)."""
    import unicodedata

    c = str(w)[:1].lower()
    if c in "йё":
        return c
    return unicodedata.normalize("NFD", c)[:1]


def initial_letters(df: pd.DataFrame) -> pd.DataFrame:
    overall = df.gender.value_counts(normalize=True)
    rows = []
    for ch, g in df.groupby(df.lemma.map(_initial)):
        if len(g) < INITIAL_MIN_N:
            continue
        sh = g.gender.value_counts(normalize=True).reindex(overall.index, fill_value=0)
        dev = (sh - overall).abs().max()
        # a gender entirely missing where >= MISSING_EXPECTED were expected: a perfect one-sided
        # cue even when the share gap is small (Spanish a-: 17 m, 0 f at 20% feminine overall)
        missing = [
            k for k in overall.index if sh[k] == 0 and overall[k] * len(g) >= MISSING_EXPECTED
        ]
        rows.append({"initial": ch, "n": len(g), **{f"share_{k}": float(v) for k, v in sh.items()},
                     "max_dev": float(dev), "missing": ",".join(missing),
                     "flag": bool(dev >= INITIAL_FLAG or missing)})  # fmt: skip
    return pd.DataFrame(rows)


# ---- balance ----------------------------------------------------------------------------
def within_cell_d(x: np.ndarray, gender: np.ndarray, cells: np.ndarray, ref: str) -> dict:
    """Per non-reference gender: (gender - ref) difference at equal cell (least squares on cell
    dummies), divided by the pooled SD of x. NaN-rated rows are left out."""
    from .estimators import nuisance

    k = np.isfinite(x)
    out = {}
    for g in sorted(set(gender[k]) - {ref}):
        sel = k & np.isin(gender, [g, ref])
        if (gender[sel] == g).sum() < 2 or (gender[sel] == ref).sum() < 2:
            continue
        M = np.column_stack([(gender[sel] == g).astype(float), nuisance(cells[sel])])
        b = np.linalg.lstsq(M, x[sel], rcond=None)[0][0]
        sd = np.std(x[sel])
        out[g] = float(b / sd) if sd > 0 else np.nan
    return out


def _token_counts(words) -> np.ndarray:
    from transformers import AutoTokenizer

    from .tokreport import TOKENIZER

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    return np.array([len(tok(" " + str(w), add_special_tokens=False)["input_ids"]) for w in words])


def features(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    from wordfreq import zipf_frequency

    from .norms import rate
    from .register_axis import gloss_aoa

    s = spec(lang)
    f = pd.DataFrame(index=df.index)
    f["length"] = df.lemma.str.len().astype(float)
    f["tokens"] = (pd.to_numeric(df["n_tokens_qwen3"], errors="coerce") if "n_tokens_qwen3" in df
                   else pd.Series(np.nan, index=df.index))  # fmt: skip
    miss = f.tokens.isna()
    if miss.any():
        f.loc[miss, "tokens"] = _token_counts(df.lemma[miss])
    f["zipf"] = (pd.to_numeric(df["zipf"], errors="coerce") if "zipf" in df
                 else df.lemma.map(lambda w: zipf_frequency(w, s.wordfreq)))  # fmt: skip
    aoa, _ = gloss_aoa(df.lemma, lang) if "concept_en" not in df else _aoa_from(df)
    f["aoa"] = aoa
    f["concreteness"] = [np.nan if (r := rate(c)[0]) is None else r
                         for c in df.get("concept_en", pd.Series("", index=df.index))]  # fmt: skip
    return f


def _aoa_from(df: pd.DataFrame):
    """AoA from a list's own `concept_en` column (same matching rule as register_axis)."""
    from .register_axis import aoa_table

    t = aoa_table()
    out = []
    for g in df.concept_en.astype(str).str.lower().str.strip():
        p = g.split()
        out.append(t.get(g, t.get(p[1]) if len(p) == 2 and p[1] in t else np.nan))
    return np.array(out, float), None


# ---- the audit ----------------------------------------------------------------------------
def load_list(lang: str, path: str | None) -> pd.DataFrame:
    """A candidate list (needs lemma, gender) or, without one, the eligible lexicon nouns.
    Missing columns are filled from the lexicon."""
    from .lexicon import eligible, load_lexicon

    lex = load_lexicon(lang)
    if path is None:
        df = eligible(lex).copy()
    else:
        df = pd.read_csv(path, keep_default_na=False)
        assert {"lemma", "gender"} <= set(df.columns), "a list needs lemma and gender columns"
        extra = [c for c in ("concept_en", "zipf", "n_tokens_qwen3") if c in lex and c not in df]
        if extra:
            df = df.merge(lex.drop_duplicates("lemma")[["lemma", *extra]], on="lemma", how="left")
    df = df[df.gender.isin(spec(lang).genders)].reset_index(drop=True)
    return df


def audit(lang: str, path: str | None = None, tag: str | None = None,
          cells_rule: str | None = None) -> dict:  # fmt: skip
    from .languages import CELL_RULES

    s = spec(lang)
    cell_fn = CELL_RULES[f"{lang}:{cells_rule}"]() if cells_rule else s.cell
    df = load_list(lang, path)
    tag = tag or (Path(path).stem if path else "lexicon_eligible")
    if cells_rule:
        tag += f"_cells-{cells_rule}"
    out = OUT / lang / tag
    out.mkdir(parents=True, exist_ok=True)
    words, gen = df.lemma.tolist(), df.gender.to_numpy()
    cells = df.lemma.map(cell_fn).to_numpy()
    ref = s.genders[0]
    res: dict = {"lang": lang, "list": tag, "n": len(df),
                 "counts": df.gender.value_counts().to_dict(), "flags": []}  # fmt: skip
    # spelling
    base = Counter(gen).most_common(1)[0][1] / len(gen)
    res["ending_loo_accuracy"] = {k: loo_ending_accuracy(words, gen, k) for k in (1, 2, 3)}
    res["majority_baseline"] = base
    cm = loo_cell_majority(cells, gen)
    cm_acc = float(np.mean(np.array(cm) == gen))
    res["cell_majority_accuracy"] = cm_acc
    res["residual_cue"] = {k: loo_ending_accuracy(words, gen, k, base=cm) - cm_acc for k in (3, 4)}
    for k, v in res["residual_cue"].items():
        if v >= CUE_FLAG:
            res["flags"].append(f"residual spelling cue: last {k} letters beat the cell majority "
                                f"by {v:.2f}")  # fmt: skip
    ini = initial_letters(df)
    ini.to_csv(out / "initial_letters.csv", index=False)
    for r in ini[ini.flag].itertuples() if len(ini) else []:
        ex = df[df.lemma.map(_initial) == r.initial]
        miss = f"; no {r.missing}" if r.missing else ""
        res["flags"].append(f"initial '{r.initial}' (n={r.n}): gender mix departs by {r.max_dev:.2f}"
                            f"{miss} ({ex.gender.value_counts().to_dict()})")  # fmt: skip
    # balance
    f = features(df, lang)
    bal = []
    for c in f.columns:
        x = f[c].to_numpy(float)
        row = {"feature": c, "coverage": float(np.isfinite(x).mean())}
        row |= {f"mean_{g}": float(np.nanmean(x[gen == g])) for g in s.genders if (gen == g).any()}
        row |= {f"d_{g}_vs_{ref}": d for g, d in within_cell_d(x, gen, cells, ref).items()}
        bal.append(row)
        for k, v in row.items():
            if k.startswith("d_") and np.isfinite(v) and abs(v) >= D_FLAG:
                res["flags"].append(f"{c}: {k} = {v:+.2f} (within cells)")
    pd.DataFrame(bal).to_csv(out / "balance.csv", index=False)
    res["balance"] = bal
    # structure
    if s.structure is not None:
        why = s.structure()
        df["structure"] = [why(w) for w in words]
        sh = df.assign(cx=df.structure != "").groupby("gender").cx.mean().to_dict()
        res["structure_share"] = sh
        if len(sh) > 1 and max(sh.values()) - min(sh.values()) >= STRUCT_FLAG:
            res["flags"].append(f"word structure: complex share by gender {sh}")
        df[df.structure != ""][["lemma", "gender", "structure", "concept_en"]].to_csv(
            out / "complex_nouns.csv", index=False)  # fmt: skip
    else:
        res["structure_share"] = None
        res["flags"].append(
            "no hand-written word-structure rule for this language (playbook Stage 3)"
        )
    # Wiktionary's own structure templates (any language; transparent ones only)
    from . import wk_structure

    wk = wk_structure.rule(lang)
    df["structure_wk"] = [wk(w) for w in words]
    shw = df.assign(cx=df.structure_wk != "").groupby("gender").cx.mean().to_dict()
    res["structure_wk_share"] = shw
    if len(shw) > 1 and max(shw.values()) - min(shw.values()) >= STRUCT_FLAG:
        res["flags"].append(f"word structure (Wiktionary): complex share by gender {shw}")
    df.assign(cell=cells).to_csv(out / "nouns.csv", index=False)
    (out / "report.md").write_text(_report(res))
    print(_report(res))
    return res


def _report(r: dict) -> str:
    lines = [f"# Language audit: {r['lang']} / {r['list']}", "",
             f"{r['n']} nouns; by gender {r['counts']}.", "", "## Flags", ""]  # fmt: skip
    lines += [f"- {x}" for x in r["flags"]] or ["- none"]
    lines += ["", "## Spelling", "",
              "- Gender from the last 1/2/3 letters (leave-one-out accuracy): "
              + " / ".join(f"{v:.2f}" for v in r["ending_loo_accuracy"].values())
              + f" (majority class {r['majority_baseline']:.2f}).",
              f"- Within cells: cell majority {r['cell_majority_accuracy']:.2f}; last 3/4 letters "
              f"add " + " / ".join(f"{v:+.2f}" for v in r["residual_cue"].values()) + ".",
              "", "## Balance (within-cell standardised differences vs the first gender)", ""]  # fmt: skip
    for b in r["balance"]:
        rest = ", ".join(f"{k} {v:.2f}" for k, v in b.items() if k not in ("feature",))
        lines.append(f"- **{b['feature']}**: {rest}")
    lines += ["", "## Word structure", "",
              f"- hand rule, complex share by gender: {r['structure_share']}",
              f"- Wiktionary templates (transparent), complex share by gender: {r['structure_wk_share']}", ""]  # fmt: skip
    return "\n".join(lines)
