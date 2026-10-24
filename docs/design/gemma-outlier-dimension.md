# Gemma's outlier dimension and the cross-fitted cosine

*Companion note to `gemma-slim-prereg.md` (amendment of 2026-10-24). This records a measurement
problem found in the Gemma replication: what it affects, how it was diagnosed, what was tried,
and what it may mean for later work with Gemma. It is not part of the results narrative.*

## 1. The measurement

Several results compare two **directions** in a model's activations: the feminine−masculine
direction against a frequency (Zipf) direction (R2, the "German frequency overlap"), or the
Spanish gender direction against the German one (Phase 4, T2). The natural comparison is the
cosine of the angle between them.

Each direction is estimated from roughly 100 nouns, so it carries noise. The ordinary
("plug-in") cosine between two noisy directions is biased, and so are their lengths. This is the
project's estimator problem, documented in `docs/methods-summary.md`. The fix used throughout is
**cross-fitting**:
- Split the nouns into halves A and B, within strata.
- Estimate each direction separately in each half.
- Build the cosine from cross-half products: the overlap term `(gA·fB + gB·fA)/2` and the squared lengths `gA·gB` and `fA·fB`.
- Average each over 200 splits, then take the ratio.

The two halves' noise is independent, so each piece is an unbiased estimate of its true value.

**The catch: the pieces don't have to agree with each other.** An ordinary cosine can't leave
[−1, 1], because of the Cauchy–Schwarz inequality. A ratio of three separately estimated pieces
can. If a squared length comes out too small, the ratio overshoots. If it comes out negative, the
ratio is undefined. The same thing happens with Spearman's correction for attenuation, which
sometimes yields "correlations" above 1 when reliabilities are low. It means the correction is
unstable at that signal-to-noise level, not that the data are wrong.

**What depends on it:**

| quantity | role | affected? |
|---|---|---|
| R2: cos(f−m, Zipf), confidence control (c) | R2's primary test | yes |
| frequency-axis A/C/D | take (c)'s overlap layers as their reference | yes, through (c) |
| Phase 4 T2 cosine *values* | descriptive | yes |
| Phase 4 T2 verdict (layers above the null) | rule-based | no: 0 layers on every model |
| Phase 3 "interpretable cosine" layers | descriptive | the same estimator: 0 on Gemma |
| Phase 3 compounds and markedness, Phase 4 T1/T3, Phase 5 | AUCs, random-direction comparisons | **no** |

## 2. Symptom

**Gemma 3 (4B and 12B), at the Phase 5 layers.** About half of the R2 cosines fell outside [−1, 1], with values of −3.97, −1.98 and +1.11. Three were undefined, because a squared length came out negative. The Spanish control showed the same pattern, and so did the number-vs-frequency check (up to +1.51). The Phase 4 T2 mean was 1.13 on 4B.

**Qwen3 8B and 14B.** All 16 R2 rows were in range.

## 3. Diagnosis (outcome-blind)

Everything below was computed without any frequency or cross-language quantity. It used only the gender
directions on training nouns and the activations themselves (`scripts/outlier_fix_diagnostics.py`,
`scripts/outlier_fix_components.py`; outputs in `results/outlier_fix/`).

**One dimension dominates Gemma's residual stream:**

| model | layers | top dimension's share of variance across nouns | mean activation norm |
|---|---|---|---|
| Gemma-3-4B | 8 / 14 / 19 / 24 | 59% / 90% / 75% / 30% | 6,900–41,000 |
| Gemma-3-12B | 12 / 19 / 26 / 34 | 59% / 92% / 87% / 31% | 8,600–66,000 |
| Qwen3-8B | 9 / 14 / 20 / 25 | 1.6% / 1.4% / 0.8% / 0.7% | 70–240 |

The dimension is the same at every layer from 3 on: **index 443 in 4B, 2339 in 12B**. It is a
single fixed channel, not a direction that rotates between layers.

**It hides signal.** On raw Gemma activations, gender read from held-out training nouns gives an
AUC of 0.57–0.60. With the dimension removed, the AUC is about 0.70–0.75, in line with Qwen. The Phase 3
and 4 verdicts replicated on raw activations anyway, so this worked against them rather than for
them. **Gemma's raw AUCs understate what the model encodes.**

## 4. What was tried

Candidate fixes were judged on five outcome-blind criteria:
1. **validity:** a test cosine stays in [−1, 1];
2. **reliability:** split-half agreement of the f−m direction;
3. **information kept:** held-out gender AUC;
4. **Qwen invariance:** Qwen has no outlier, so a good fix leaves it alone;
5. **simplicity:** fewer free choices is better.

The test cosine was R2's exact estimator applied to a quantity unrelated to frequency:
cos(f−m, (f+m)/2 − n), with German training nouns.

| fix | what it does | result |
|---|---|---|
| A. standardise each dimension | unit variance for every dimension | **rejected**: damages Qwen (AUC and agreement fall, Qwen test cosines go invalid). It amplifies noise in near-constant dimensions |
| B. drop dimensions with >10% variance | removes 1 dimension per Gemma model, 0 in Qwen | **chosen**: Qwen unchanged, Gemma AUC restored, no tuned parameter in practice |
| C. remove the top principal component | | same as B in effect |
| D0. per-token RMS normalisation (Gemma's RMSNorm without its learned weights) | | same as B in effect. The learned weights weren't available (the HF token was revoked) |
| E. full whitening | | not tried: too many free choices for about 100 nouns in 2,500–3,800 dimensions |
| F. plug-in cosine checked against the null | | not tried: reintroduces the bias that cross-fitting removes |

**Result, with halves split within cell × gender and each half residualised on its own cells,
as the production estimator does:**

| | raw: test cosine valid | B / C / D0: valid |
|---|---|---|
| Qwen3-8B / 14B | 4/4, 4/4 | 4/4, 4/4 |
| Gemma-4B | 1/4 | 1/4 (misses −1.2 to −1.4) |
| Gemma-12B | 1/4 | 3/4 (miss about −1.2) |

**A mistake in the first diagnostic, corrected.** The first version residualised all nouns on
their spelling cells, then split them. That couples the two halves, because each half's residuals
depend on the other half's nouns, and it biases every cross-half product. It made even Qwen look
invalid. The production estimator refits the cell design inside each half, so the reported Qwen
results never had this problem. The corrected diagnostic (commit 51e2315) matches production.

**Why it still fails at some Gemma layers once the outlier is removed.** The three pieces are then all positive
and reasonably precise. The failures are modest overshoots, about −1.1 to −1.4, at layers where
the true cosine appears to lie close to −1. When the true value sits at the boundary of [−1, 1],
an unbiased ratio estimator crosses it roughly half the time. On Qwen the same test cosine is
−0.2 to −0.5, well inside the range.

## 5. Decision

Pre-registered as an amendment to `gemma-slim-prereg.md`, before any rerun:
- **Fix B.** It has no tuning beyond the 10% threshold, and every Gemma layer has exactly one dimension far above it.
- **A validity gate.** A layer with an invalid cosine shows no overlap, and 2 or more invalid layers out of 4 means "no valid test".
- **No second fix.**

## 6. Results under the fix

*(Filled in after the run; see §7 of the prereg's results section.)*

## 7. What this may mean for future Gemma work

- **Check variance concentration first.** Before any geometric analysis on Gemma (cosines,
  lengths, distances, difference-of-means, unregularised regressions), compute the top
  dimension's share of variance. It takes one line, and it would have caught this before the run.
- **Large single-channel activations are a known LLM phenomenon**, often called "massive activations" or "outlier
  dimensions". They're reported across several model families, and they tend to sit in a fixed
  channel across layers, as here. Gemma's are unusually large relative to its other dimensions
  (norms of 10⁴ against about 10² in Qwen). Literature to check: Sun et al. 2024, *Massive
  Activations in Large Language Models*; Kovaleva et al. 2021, *BERT Busters* (outlier dimensions);
  Mu & Viswanath 2018, *All-but-the-Top*. *(Claude-supplied pointers, not verified in this session.)*
- **AUC-based probes are robust to it in direction but not in magnitude.** Expect raw-activation
  probes to understate gender decodability on Gemma.
- **Steering is less exposed.** α is scaled to the median residual norm at noun positions, and
  random-direction comparisons share the same geometry. Still, a norm dominated by one channel
  means "α × median norm" is mostly a multiple of that channel's size. A future Gemma steering study
  could scale doses by the norm with the outlier removed and check whether verdicts hold.
- **SAEs.** Gemma Scope 2 SAEs are trained on raw activations, so this channel is presumably
  handled inside the SAE, by features or a bias term. G2-style analyses should check whether
  candidate features mainly track dimension 443/2339.
- **Model-native normalisation.** The principled version of D0 applies Gemma's own RMSNorm weights
  (the input to the next layer). Worth running when a token is available. If it agrees with B, B is
  justified on the model's own terms.

## 8. Open questions

- Why is the f−m vs neuter cosine near −1 on Gemma but about −0.4 on Qwen? It may be a real difference
  in how Gemma arranges German gender (feminine nearer neuter than masculine is?), or residual
  geometry. That is descriptive, and not part of any verdict.
- Does the same channel dominate on non-noun inputs (sentences, other languages)? If so, it is a
  model-wide property, not a noun-list artefact.
