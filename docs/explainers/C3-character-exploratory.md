# C3. Character study: exploratory follow-ups (2026-11-11)

All labelled **exploratory**: run after the confirmatory results (C2), not pre-registered as
tests. Scripts: `character/scripts/run_exploratory.py`, `run_style_leak.py`, `coref_check.py`.
Results: `character/results/exploratory/`, `character/results/coref/`.

## E1. Is Qwen's writer effect really about the shown word's gender?

The pre-registered control (C2) found that for Qwen, a noun's German or Russian gender still
nudged the pronoun when the word shown was Finnish, Japanese or Turkish, which have no gender.
That means some nouns lean one way under *any* foreign word. Two ways to take that out:
- **Difference-in-differences:** subtract each noun's average genderless-language shift from its
  gendered-language shifts ("what this word's gender adds beyond what any foreign word does").
- **Covariate:** keep the genderless shift as a control variable in the same regression.

| cell | original | minus genderless shift | covariate |
|---|---|---|---|
| Luna medium | +17.8 | +18.0 (15.3 to 20.7) | +18.0 |
| Luna none | +33.2 | +32.9 (29.3 to 36.5) | +33.1 |
| Qwen off | +32.7 | **+29.3** (24.2 to 34.4) | +30.6 |
| Qwen on | +19.6 | **+17.6** (13.7 to 21.5) | +18.6 |

Luna is unchanged; Qwen loses 2–3 points. **The noun confound is real for Qwen but small: the
grammatical-gender effect survives almost whole.**

## E2. Thinking / reasoning (Q4)

- **The writer effect shrinks with reasoning in both families:** Luna −15.5 points (−18.7 to
  −12.2), Qwen −13.1 (−17.5 to −8.7). The models still follow grammatical gender with reasoning
  on, about half as strongly.
- Overall pronoun mix moves little. Luna: with reasoning, slightly more *she* in the writer study
  (80 → 87%). Qwen: writer study a bit more *they* with thinking (16 → 21%), character study a bit
  less (13 → 10%).

## E3. Mode or settings? (Qwen, amendment 3 control)

Qwen's two modes use different recommended sampling (T 0.7 off, T 0.6 on). The control ran
thinking off with the thinking settings on a subset of prompts.

| subset | off (T 0.7) | off + thinking settings | on |
|---|---|---|---|
| writer, she / he / they % | 50.0 / 40.8 / 9.2 | 49.8 / 40.8 / 9.3 | 56.1 / 34.6 / 9.3 |
| character plain a/an | 74.1 / 6.4 / 18.0 | 74.0 / 6.6 / 18.2 | 82.4 / 8.3 / 9.0 |

The settings change does nothing; the whole difference comes from the thinking mode itself. So
Qwen's on-vs-off contrast can be read as a mode effect. (The control has no foreign-word arms, so
this speaks to the pronoun mix, not directly to the size of the Q1 effect.)

## Style (amendment 3: the literature predicted more structured/formal prose with reasoning)

Paired on − off, mean over nouns:
- **Sentences get longer** with reasoning: Qwen +2.8 (writer) / +3.5 (character) words per
  sentence; Luna +0.8 / +0.3. Paragraphs get slightly shorter (−1.5 to −3.7 words).
- Word variety up slightly for Qwen (type-token ratio +0.02).
- **Formal connectives** (*however, moreover*…) essentially unchanged (rare in both modes); **no
  list formatting** in any output.
So the predicted drift shows only as longer sentences, clearly for Qwen; no sign of
structured/argumentative prose.

## Leakage

0 of ~480,000 outputs open with reasoning-style text or contain think tags, in any cell. Thinking
off really was off.

## Coder check: FastCoref vs regex (validation, not outcome)

300 random valid writer outputs per cell, FastCoref's main pronoun cluster vs the regex class:

| cell | agreement | kappa |
|---|---|---|
| Luna none | 100.0% | 1.00 |
| Luna medium | 98.7% | 0.95 |
| Qwen off | 99.7% | 0.99 |
| Qwen on | 99.3% | 0.99 |

Disagreements (8 of 1,200) are edge cases: a *she* paragraph where FastCoref's biggest cluster was
an *it*, and a few *mixed* / *they* boundary cases. Well above the 0.6 limitation threshold. FastCoref
ran in its own environment (`transformers==4.36.2`, spaCy `en_core_web_sm`); it fails with current
transformers.

## Content: does the description change, not just the pronoun? (2026-11-19, full run 2026-11-21)

**Exploratory.** The pre-registration froze content measures for the Casey/Riley study (deferred),
not for these studies. Scripts `character/scripts/content_main.py`, `content_analysis.py`,
`character/kaggle/run.py`; results `character/results/content/analysis_full.txt`.

- **Glasgow word average** (human ratings, no model) and **five embedders** (bge-m3, e5-instruct,
  LaBSE, Qwen3-Embedding-0.6B, EmbeddingGemma; feminine-minus-masculine anchor similarity), on
  **every valid output of both studies** (~489,000). Pronouns, names, gendered person words, the
  noun and (character study) the modifier's own words are masked; plural *they* is left intact in
  he/she paragraphs (pre-set rule). Glasgow: paragraphs with at least 10 rated words.
- Embedders ran on Kaggle (two T4 GPUs, fp32, one scorer per card; three sessions, ~17 GPU-hours,
  free). Checks: T4 vs Mac CPU on 300 paragraphs, max difference 0.00001; Kaggle's EmbeddingGemma
  copy byte-identical to the Hugging Face one; batch size and a 2,048-token cap (needed for one
  runaway 128,813-character invalid output) leave scores unchanged. fastText ran on the sample only.
- **Correction:** the first (sample-only) embedder numbers reported on 2026-11-19 were wrong: the
  merge matched outputs without the model, so Luna-medium was mixed with Qwen-on and Luna-none with
  Qwen-off (they share the thinking flag). Glasgow was unaffected. Fixed; the corrected sample
  agrees with the full run below.

Numbers are in **paragraph standard deviations**, signed so that **positive = more feminine** for
every scorer (Glasgow is flipped, since its scale runs 1 = feminine). `*` = 95% CI excludes 0.

| scorer | she minus he paragraphs | Q1 content effect (f − m nouns) | same, *she* paragraphs only |
|---|---|---|---|
| Glasgow | 0.08 to 0.32 | 0.06 to 0.10 (3 of 4 *) | 0.05 to 0.11 (2 of 4 *) |
| bge-m3 | 0.12 to 0.34 | 0.11 to 0.13 (4 of 4 *) | 0.06 to 0.10 (2 of 4 *) |
| e5-instruct | 0.16 to 0.35 | 0.09 to 0.13 (4 of 4 *) | 0.04 to 0.09 (1 of 4 *) |
| LaBSE | 0.04 to 0.25 | 0.04 to 0.09 (2 of 4 *) | 0.01 to 0.10 (1 of 4 *) |
| Qwen3-Embedding | 0.11 to 0.30 | 0.07 to 0.11 (4 of 4 *) | 0.01 to 0.09 (1 of 4 *) |
| EmbeddingGemma | 0.15 to 0.36 | 0.08 to 0.18 (4 of 4 *) | 0.05 to 0.10 (2 of 4 *) |

How to read it:
- **Every scorer is a weak gender detector**: a she-paragraph and a he-paragraph about the same
  noun differ by only 0.04–0.36 of a typical paragraph difference.
- **Q1: the description follows the foreign word's gender.** All 24 scorer × cell combinations are
  positive and 21 exclude 0 (the exceptions: LaBSE for both Qwen cells, Glasgow for Qwen on). The
  shift is about as large as the whole she/he gap for Luna and about a third of it for Qwen.
- **Beyond the pronoun:** among paragraphs that already use *she*, the shift is still positive in
  all 24 combinations; it excludes 0 for **Luna medium with every scorer** and in 2 of 12 Qwen
  combinations. So for Luna the foreign word's gender colours the description itself, weakly; for
  Qwen the content shift is mostly carried by the pronoun change.
- **Character study (all scorers now):** across the 70 modifiers, a modifier's pronoun effect and
  its content effect move together in every cell: embedders r = +0.54 to +0.71 (LaBSE +0.09 to
  +0.37), Glasgow −0.36 to −0.46 (same direction, flipped scale). Modifiers that pull toward *he*
  also pull the description toward masculine-reading content.

Limits: weak instruments; anchors are word lists (properties primary) fixed before the main run;
effects are a direction, not a size. Report as "content moves with the pronoun, weakly; beyond the
pronoun for Luna".
