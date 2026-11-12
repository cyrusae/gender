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
