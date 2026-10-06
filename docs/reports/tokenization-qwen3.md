# Tokenisation of the lexicon nouns (Qwen3 tokenizer)

*Generated 2026-10-06 by `gbleed token-report` with `Qwen/Qwen3-1.7B-Base`'s tokenizer (identical across Qwen3 sizes). Eligible nouns with Zipf ≥ 2.5. Words are tokenised with a leading space, as they appear mid-sentence.*

| | nouns | mean tokens | single-token | English glosses: mean tokens | single-token |
|---|---|---|---|---|---|
| de | 8727 | 3.35 | 2% | 1.89 | 44% |
| es | 5588 | 2.57 | 4% | 1.69 | 53% |

## Token count vs other properties (Spearman ρ)

| lang | property | ρ | p |
|---|---|---|---|
| de | frequency (Zipf) | -0.36 | 5e-268 |
| de | length in letters | +0.70 | 0e+00 |
| de | frequency of the same string in English | -0.44 | 0e+00 |
| de | feminine (vs masculine) | +0.09 | 7e-17 |
| es | frequency (Zipf) | -0.34 | 1e-155 |
| es | length in letters | +0.37 | 1e-178 |
| es | frequency of the same string in English | -0.38 | 6e-196 |
| es | feminine (vs masculine) | -0.01 | 6e-01 |

**de**: mean tokens by frequency bin and gender

| bin   |    f |    m |
|:------|-----:|-----:|
| low   | 3.64 | 3.45 |
| mid   | 3.01 | 2.6  |
| high  | 2.02 | 1.93 |

**es**: mean tokens by frequency bin and gender

| bin   |    f |    m |
|:------|-----:|-----:|
| low   | 2.75 | 2.73 |
| mid   | 2.4  | 2.38 |
| high  | 1.68 | 1.71 |

## Why it matters

- Token count changes *where* a word's last token sits and *what kind* of token it is (a whole word vs a fragment like `ung`), so it can leak into word-level representations.
- In **German** it differs by gender even within frequency bins (feminine suffixes like *-ung*, *-heit*, *-keit*, *-schaft* lengthen words and often get their own tokens), so Phase 3 should match or adjust for token count. In **Spanish** it doesn't.
- History: the Phase 2 attention-sink bug (single-token words measured on a sink position) was a token-count artifact; see `docs/decisions.md`.
