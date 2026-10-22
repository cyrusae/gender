# Why the German gender direction points toward "rarer": pre-registration (ADOPTED, PI 2026-10-21)

Written 2026-10-21, after the confidence control's part (c) and before any of these numbers.
Diagnostic, CPU only, existing activations (Qwen3 8B and 14B, LAST, the four Phase 5 steering
layers). Follows `confidence-ctl-prereg.md` (c).

## What (c) found, and the question

The German feminine-minus-masculine direction has a cross-fitted cosine of −0.34 to −0.69 (8B)
and −0.26 to −0.46 (14B) with the frequency (Zipf) coefficient of the same fit: it points
substantially toward "rarer word". Spanish is ≤ 0.14. Frequency was already a covariate, and in
both languages the *feminine* training nouns are the commoner ones (mean Zipf 3.77 vs 3.28 in
German), so this isn't leftover confounding in the simple sense. Four candidate explanations,
one test each:

| | Explanation | Test |
|---|---|---|
| A | The linear frequency control is too crude (frequency's effect is curved, or interacts with token count) | refit with a flexible frequency control |
| B | (Not an explanation: does it matter?) Do the reported German geometry results depend on this component? | remove the frequency axis, rerun Phase 4 |
| C | The regression itself (collinearity of gender and frequency) produces it | a regression-free estimator: frequency-matched pairs |
| D | Word structure: the masculine set contains compounds and derived words, the feminine set none | refit without them / with a structure covariate |

**The word-structure imbalance (found while preparing D, by rule, not by model scores):** of the
91 German m/f training nouns (Phase 3 `strat3`), 12 of 64 masculine and 0 of 27 feminine are
complex by the rule below. Phase 3's "simplex only" filter missed them: its agent-noun rule
tests the whole word minus *-er* (*Bohrer* ← *bohren*), so compounds with an agent/instrument
head (*Wasserkocher* "kettle" ← *Wasser* + *kochen*) and numeral nouns (*Vierer* "a four") got
through. List: *Vierer, Fünfer, Sechser, Zehner* (numeral + *-er*); *Feuerlöscher* "fire
extinguisher", *Plattenspieler* "record player", *Wasserkocher* "kettle", *Zahnstocher*
"toothpick", *Akkuschrauber* "cordless screwdriver", *Rohrkrepierer* "dud (shell)",
*Wasserspeier* "gargoyle", *Pfannenwender* "spatula" (noun + verb + *-er*). This is a stimulus
issue for every German direction (Phases 3–5) whatever D finds; it is logged in decisions.md.

**Complexity rule (D), fixed now:** a German noun is *complex* if (i) it ends in *-er* and the
rest is a cardinal number word (*eins* … *zwölf*, *zwanzig* … *fünfzig*, *hundert*, *tausend*);
or (ii) it is noun + noun: some split at position ≥ 3 leaves a lexicon noun of ≥ 4 letters as the
head and a lexicon noun of ≥ 3 letters as the first part (optionally plus a linking *-s/-n/-en/
-es/-e*); or (iii) it ends in *-er* and the whole word minus *-er* plus *-en* is a verb (Phase 3's
agent rule); or (iv) it ends in *-er* with a first part as in (ii) and a head (≥ 5 letters)
whose stem minus *-er* plus *-en* is a verb. Lexicon = `data/lexicon/de_nouns.csv` (all
animacies); verbs = `phase3_stimuli.german_verbs()`. Known misses: noun + *-er* derivations
(*Benziner* ← *Benzin*). No hand edits.

## Shared machinery (as (c))

The fit is least squares of the activations on [gender, ending cells, covariates], with Phase 3's
German covariates (concreteness, its missing flag, Zipf, token count) and Phase 2's Spanish ones.
g = gender coefficient; f = Zipf coefficient. Cosines are cross-fitted (g from one half, f from
the other, squared lengths cross-fitted too), 200 splits within cell × gender strata. A cosine
counts as **frequency overlap** if the cross term beats the within-cell × gender permutation null
(200 permutations × 20 splits, 95th percentile of |cross term|) *and* |cos| ≥ 0.3, as (c).
Spanish is run alongside A and C as the comparison (it showed no overlap).

## A. Flexible frequency control

Replace the linear Zipf and token-count covariates with: Zipf as a cubic polynomial
(standardised Zipf, its square and cube); token count as categories (1, 2, 3, 4+); and Zipf ×
token count. g_flex = the gender coefficient of this fit. Report the cross-fitted cos(g_flex, f),
with f the *linear* Zipf coefficient of the original fit on the other half. That is the same
frequency axis as (c), so only the gender estimate changes.

**Reading:** **"the overlap is an artefact of the linear control"** if it disappears (fails the overlap
rule) at a majority of the layers where (c) found it. Otherwise **"survives a flexible frequency
control"**.

## B. Do the German geometry results depend on it?

At every inner layer, f̂ = the unit Zipf coefficient of the full German fit. Project f̂ out of
*every* activation Phase 4 uses (Spanish and German training nouns, the flipped pairs, Russian),
x ← x − (x·f̂) f̂, and rerun `phase4.analyze` (LAST) unchanged on 8B and 14B. Compare with the
published Phase 4 verdicts: T1 (transfer), T3 (flipped pairs, ES/DE on both), their layer counts,
the T2 mean cosine and the T2 layers above the null.
- Secondary: project out both languages' f̂ (the span of the two).
- Removing one direction estimated on the same German nouns from those nouns is a mild
  circularity (1 of ~4,000 dimensions). State it.

**Reading:** **"German geometry doesn't depend on the rarity axis"** if T1's and T3's verdicts are
unchanged and the T2 mean cosine moves by < 0.05. Otherwise report what changed. Those results
then carry the caveat in the write-up.

## C. Regression-free estimate: frequency-matched pairs

Within each ending cell, pair feminine with masculine nouns 1:1, without replacement: equal
token count and |ΔZipf| ≤ 0.3, greedy from the rarest feminine noun (nearest Zipf; ties by the
fixed seed). g_pair = mean over pairs of (x_f − x_m). Cross-fitted cosine with f: pairs split in
halves (both nouns of a pair in the same half; unpaired nouns assigned at random). g_pair comes
from one half's pairs, f from the linear fit on the other half's nouns. Same overlap rule. With
fewer than 10 pairs it's reported as **too few** (descriptive). Report the number of pairs and
their Zipf balance.

**Reading:** if the overlap persists here, the regression adjustment isn't what produces it.

## D. Word structure

- **D1 (primary):** refit without the complex nouns (79 German nouns remain).
- **D2:** all nouns, with a complex indicator as one more covariate.
Same cross-fitted cosine and overlap rule.

**Reading:** **"word structure explains it"** if in D1 the overlap disappears at a majority of the
layers where (c) found it. Otherwise **"not word structure (as detected by this rule)"**.

## What this can and can't show

All four are diagnostics on the model's geometry; none changes a published verdict by itself.
B is the one that decides whether anything reported needs a caveat. A, C and D narrow down
*why*. Several can hold at once (they aren't mutually exclusive), and an explanation outside the
four (e.g. a genuine shared feminine/rare axis in German) is what's left if none removes it.

Code: `src/gbleed/frequency_axis.py`, CLI `gbleed frequency-axis`. Results go to
`results/frequency_axis/`. Committed before it's run.

## Results A, C, D (2026-10-21; Qwen3 8B/14B pod activations; `results/frequency_axis/`)

Cross-fitted cos(gender, frequency) at the four steering layers (8B: 9/14/20/25; 14B:
10/16/22/28). The (c) column is the original fit.

| | (c) original | A flexible control | C matched pairs (14, ΔZipf −0.01) | D1 no complex nouns (79) | D2 complex covariate |
|---|---|---|---|---|---|
| 8B German | −0.34 / −0.56 / −0.62 / −0.69 | −0.31 / −0.55 / −0.54 / −0.69 | −0.37 / −0.53 / −0.47 / −0.52 | −0.26 / −0.52 / −0.57 / −0.69 | −0.27 / −0.55 / −0.57 / −0.68 |
| 14B German | −0.26 / −0.34 / −0.46 / −0.44 | −0.19 / −0.34 / −0.43 / −0.43 | −0.20 / −0.46 / −0.52 / −0.48 | −0.22 / −0.31 / −0.39 / −0.42 | −0.22 / −0.31 / −0.41 / −0.41 |
| 8B Spanish | ≤ 0.09 | ≤ 0.08 | too few pairs (8) | n/a | n/a |
| 14B Spanish | ≤ 0.14 | ≤ 0.16 | too few pairs (8) | n/a | n/a |

**Readings (as pre-registered):** in no variant does the overlap disappear at a majority of the
layers where (c) found it. A: 0 of 4 (8B), 0 of 3 (14B). D1: 1 of 4, 0 of 3. C: 1 of 4, 1 of 3,
in each case the top layer, where the cosine stays around −0.5 but the cross term misses the
null's bar on 14 pairs. So the overlap is **not an artefact of the linear frequency control
(A)**, **not produced by the regression (C: frequency-matched pairs with mean ΔZipf ≈ 0 give
the same cosines)**, and **not word structure as detected by the rule (D)**. What's left is the
residual explanation: in Qwen3 8B/14B, German grammatical femininity is encoded partly along
the same axis as low frequency. Mechanism unknown.

Two hypotheses this leaves (not tested here):
- **The model's frequency axis isn't purely about frequency** (for example, how "dictionary-like" or
  technical a word is), and German feminine nouns sit at one end of it.
- **Something specific to German *die*:** the article is shared with the plural. The number
  direction's own frequency alignment is small (≤ 0.17), which argues against this.

## Result B (2026-10-22; Phase 4 rerun with the frequency axis projected out, LAST)

The unprojected rerun reproduces the published Phase 4 exactly on both models (same layer
counts; T2 mean cosine 0.4287 / 0.3517), so the comparisons are like for like.

| | 8B baseline | 8B − German freq | 8B − both | 14B baseline | 14B − German freq | 14B − both |
|---|---|---|---|---|---|---|
| T1 transfers (both directions) | no | no | no | no | no | no |
| ES → DE layers (Holm) | 6/35 | 6 | 7 | 7/39 | 6 | 12 |
| **DE → ES layers (Holm)** | **5/35** | **18** | 15 | **22/39** | **29** | 29 |
| T3 pairs follow gender | yes (35/35, 35/35) | yes | yes | yes (39/39) | yes | yes |
| T2 mean cosine | 0.429 | 0.419 | 0.403 | 0.352 | 0.303 | 0.324 |
| T2 layers above null | 23 | 23 | 22 | 28 | 27 | 26 |

**Reading (as pre-registered): robust on both models.** The T1 and T3 verdicts are unchanged
and the T2 mean cosine moves by −0.009 (8B) and **−0.049 (14B, just inside the 0.05 bar)**.
So the published German geometry results don't depend on the rarity axis. The 14B T2 cosine is
the most sensitive number: about an eighth of it goes with the frequency axis.

**Not predicted (reported as found):** removing the German frequency axis makes the *German*
direction transfer to Spanish much better: 5 → 18 layers on 8B (a bare majority of 35), and
22 → 29 on 14B. Spanish → German barely changes. A reading: the frequency component is part of
the German direction that Spanish nouns don't share. It adds noise when that direction scores
Spanish nouns, and removing it leaves a cleaner shared gender component. T1 as pre-registered
still fails (it needs both directions), so this is descriptive.
