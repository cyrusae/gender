# XWEAT T7/T8 sensitivity: results (2026-11-11)

Pre-registration: `docs/xweat-sensitivity-prereg.md` (adopted as written; see its status note).
Script: `scripts/xweat_sensitivity.py`. Vectors: MUSE `wiki.multi.{en,de,es,it,hr,ru,tr}.vec`,
first 200k words, cosine (`euclidean/`: Russian and Turkish only, R1 follow-up). Lists and
translations: XWEAT commit 4f95484 (their pickled dictionaries). Issues: `docs/upstream-issues.md`
X1–X13.

Files: `summary.csv` (one row per language × test × arm), `leave_one_out.csv` (clean sets minus
one word), `per_word.csv` (every word: source, translation, vocabulary status, drop reason,
Wiktionary gender, association score), `info.json` (vector files, XWEAT commit).

**Reading the numbers.** Effect sizes are Cohen's-d-like (difference of group means over the
pooled SD). Each group has 6–9 words, and the per-word association scores behind them are small
cosine differences (about −0.06 to +0.03 in Spanish T8). A "large" effect here can rest on
differences in the second decimal of a cosine.

## R1: reproduction

The repro arm scores every word XWEAT's code could have popped (X2); the paper's value
"reproduces" if it lies within that range ± 0.15.

| test | lang | repro range (median) | paper | R1 |
|---|---|---|---|---|
| T7 | en | 1.30 | 1.30 | reproduces (exact) |
| T7 | de | 0.46 | 0.46 | reproduces (exact) |
| T7 | es | 1.19–1.54 (1.21) | 1.47 | reproduces (one draw of 8) |
| T7 | it | 0.78–1.15 (0.89) | 1.00 | reproduces (one draw of 8) |
| T7 | hr | 0.61–0.92 (0.69) | 0.72 | reproduces (one draw of 8) |
| T7 | ru | 0.37–1.03 (0.67) | 0.59 | reproduces (one draw of 63) |
| T7 | tr | −1.19 to +0.63 (−0.26) | −0.88 | reproduces (one draw of 21; sign depends on the draw) |
| T8 | en | 1.30 | 1.30 | reproduces (exact) |
| T8 | de | 0.05 | 0.05 | reproduces (exact) |
| T8 | es | 1.11 | 1.16 | reproduces (within 0.15) |
| T8 | it | 0.10 | 0.10 | reproduces (exact) |
| T8 | hr | 0.13 | 0.13 | reproduces (exact) |
| T8 | ru | −0.13 to 0.08 | 0.37 | **doesn't reproduce** (Euclidean −0.07 to 0.13) |
| T8 | tr | 0.87–1.26 | 1.72 | **doesn't reproduce** (Euclidean 0.88–1.28) |

Exact matches from different vector files confirm that MUSE keeps the paper's within-language
geometry, and are consistent with Table 5 using cosine as the paper says (their committed run
script uses Euclidean, X7; Euclidean wasn't run for the matching cells). RU/TR T8: cause
unknown (X13).

## Clean arm (mechanical fixes) vs repro

| test | en | de | es | it | hr | ru | tr |
|---|---|---|---|---|---|---|---|
| T7 repro (median) | 1.30 | 0.46 | 1.21 | 0.89 | 0.69 | 0.67 | −0.26 |
| T7 clean | 1.30 | 0.46 | 1.25 | 0.90 | 0.71 | 0.69 | −0.35 |
| T8 repro (median) | 1.30 | 0.05 | 1.11 | 0.10 | 0.13 | 0.06 | 0.96 |
| T8 clean | 1.30 | 0.37 | 1.31 | 0.44 | −0.06 | 0.04 | 1.10 |

T7 barely moves; T8 moves a lot (DE +0.32, IT +0.34, HR −0.19). In German the only T8 change is
dropping the untranslated *Einstein*, *NASA*, *Shakespeare* (X3): the three English names, two of
them men, carry about a third of a standard deviation of German T8.

## R2: fragility (clean arm, leave-one-out)

Fragile = some single drop flips the sign (S), crosses p = .05 (P) or moves the effect ≥ 0.5 (B).

| | en | de | es | it | hr | ru | tr |
|---|---|---|---|---|---|---|---|
| T7 | robust | P | robust | P | P | P | S, B |
| T8 | robust | S | robust | robust | S | S | P |

Largest single-word moves: TR T7 *onunki* ("hers") +0.85, flips sign (X5); DE T8 *Oma* ("grandma",
vs *Großvater* "grandfather", X4) +0.47; HR T8 *eksperiment* and RU T8 *эксперимент* ("experiment")
each flip the sign; DE T7 *Junge* ("boy") +0.42.

## R3: cross-language comparisons (leave-one-out ranges overlap?)

| pair | test | ranges | reading |
|---|---|---|---|
| ES vs IT | T8 | 1.03–1.57 vs 0.16–0.81 | distinguishable |
| DE vs ES | T7 | 0.09–0.87 vs 1.05–1.54 | distinguishable |
| DE vs ES | T8 | −0.06–0.84 vs 1.03–1.57 | distinguishable |
| EN vs ES | T7, T8 | overlap | not distinguishable |
| EN vs TR | T8 | overlap | not distinguishable |
| EN vs DE, IT, HR, RU | T7, T8 | no overlap | distinguishable (EN higher) |
| EN vs TR | T7 | no overlap | distinguishable |

So the Spanish/Italian and German/Spanish gaps survive single-word perturbation; Spanish can't be
told apart from English. This **cuts against an earlier expectation** (in chat, before the run)
that the cross-language comparisons might not survive at all.

## R4: within-gender (descriptive)

Feminine-only target sets on both sides (Wiktionary genders; plural and two-gender entries such as
*arte* excluded): ES T7 1.74 (3 vs 6 words, p < .001), ES T8 1.33 (5 vs 5, p = .02); DE T7 −1.04
(4 vs 5), DE T8 0.26 (5 vs 4); RU T8 0.44 (5 vs 4). Other cells have < 3 words per side. The Spanish
effect is present among same-gender nouns, so it isn't produced by the targets' grammatical gender.

## Exploratory (after seeing scores)

- ES vs IT T8 per word: Spanish *física*, *química*, *ciencia*, *experimento* lean slightly male;
  Italian *fisica*, *tecnologia*, *chimica* lean female (`per_word.csv`, `s_clean`). Same grammar,
  near-identical spellings; not explained here.
