"""Token counts for every lexicon noun under the Qwen3 tokenizer (shared by all Qwen3 sizes),
and a short report on what token count correlates with.

Adds `n_tokens_qwen3` (the noun as it appears mid-sentence, with a leading space) and
`n_tokens_qwen3_gloss` (its English gloss) to data/lexicon/{de,es}_nouns.csv, and writes
docs/reports/tokenization-qwen3.md. Tokenisation isn't random: frequent strings become single
tokens, so token count tracks frequency, length and similarity to English, and in German it
also differs by gender, which matters for word-level analyses (see the report).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .lexicon import LEX_DIR, eligible, load_lexicon

TOKENIZER = "Qwen/Qwen3-1.7B-Base"
REPORT = Path("docs/reports/tokenization-qwen3.md")


def add_token_counts() -> dict[str, pd.DataFrame]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    def n(text) -> int:
        return len(tok(" " + str(text), add_special_tokens=False)["input_ids"])

    out = {}
    for lang in ("de", "es"):
        df = load_lexicon(lang)
        df["n_tokens_qwen3"] = [n(w) for w in df.lemma]
        df["n_tokens_qwen3_gloss"] = [n(c) for c in df.concept_en]
        df.to_csv(LEX_DIR / f"{lang}_nouns.csv", index=False)
        out[lang] = df
    return out


def write_report(lex: dict[str, pd.DataFrame]) -> str:
    from scipy.stats import spearmanr

    lines = [
        "# Tokenisation of the lexicon nouns (Qwen3 tokenizer)",
        "",
        (
            f"*Generated {datetime.now(UTC):%Y-%m-%d} by `gbleed token-report` with `{TOKENIZER}`'s "
            "tokenizer (identical across Qwen3 sizes). Eligible nouns with Zipf ≥ 2.5. Words are "
            "tokenised with a leading space, as they appear mid-sentence.*"
        ),
        "",
        "| | nouns | mean tokens | single-token | English glosses: mean tokens | single-token |",
        "|---|---|---|---|---|---|",
    ]
    corr_rows, bin_tables = [], []
    for lang, df in lex.items():
        d = eligible(df)
        d = d[d.zipf >= 2.5].copy()
        lines.append(
            f"| {lang} | {len(d)} | {d.n_tokens_qwen3.mean():.2f} | {np.mean(d.n_tokens_qwen3 == 1):.0%} "
            f"| {d.n_tokens_qwen3_gloss.mean():.2f} | {np.mean(d.n_tokens_qwen3_gloss == 1):.0%} |"
        )
        d["nchar"] = d.lemma.str.len()
        d["fem"] = (d.gender == "f").astype(int)
        for col, label in [("zipf", "frequency (Zipf)"), ("nchar", "length in letters"),
                           ("zipf_en", "frequency of the same string in English"),
                           ("fem", "feminine (vs masculine)")]:  # fmt: skip
            r = spearmanr(d.n_tokens_qwen3, d[col])
            corr_rows.append(f"| {lang} | {label} | {r.statistic:+.2f} | {r.pvalue:.0e} |")
        d["bin"] = pd.cut(d.zipf, [2.5, 3.5, 4.5, 9], labels=["low", "mid", "high"])
        t = d.groupby(["bin", "gender"], observed=True).n_tokens_qwen3.mean().unstack().round(2)
        bin_tables.append(
            f"\n**{lang}**: mean tokens by frequency bin and gender\n\n" + t.to_markdown()
        )
    lines += ["", "## Token count vs other properties (Spearman ρ)", "",
              "| lang | property | ρ | p |", "|---|---|---|---|", *corr_rows, *bin_tables]  # fmt: skip
    lines += [
        "",
        "## Why it matters",
        "",
        (
            "- Token count changes *where* a word's last token sits and *what kind* of token it is (a whole "
            "word vs a fragment like `ung`), so it can leak into word-level representations."
        ),
        (
            "- In **German** it differs by gender even within frequency bins (feminine suffixes like "
            "*-ung*, *-heit*, *-keit*, *-schaft* lengthen words and often get their own tokens), so Phase 3 "
            "should match or adjust for token count. In **Spanish** it doesn't."
        ),
        (
            "- History: the Phase 2 attention-sink bug (single-token words measured on a sink position) "
            "was a token-count artifact; see `docs/decisions.md`."
        ),
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    REPORT.write_text(text)
    return text
