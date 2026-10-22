# Confidence/flattening control: pre-registration (ADOPTED, PI 2026-10-21)

Written 2026-10-21, before any of these numbers were computed. Diagnostic, CPU only, data
already on disk. Origin: a research-agent spec (`docs/feynman/confidence-direction-control-spec.md`)
on Buttigieg et al. 2026 (arXiv 2610.08559). Claude's review cut it down to the three parts
below (reasons in `docs/decisions.md`, 2026-10-21).

## The worry in plain words

Buttigieg et al. found that "bias directions" built by contrasting biased with anti-biased answers
were mostly **confidence** directions. Pushing along them made the model less sure of
everything: it flattened the probabilities over the answer options, much like raising the
temperature. Two routes by which that could fake our results:

1. **Flattening mimics a gender score.** If a push (or an article) just flattens the adjective
   distribution, every adjective's log-probability moves toward the average. Likely adjectives
   lose and unlikely ones gain. If feminine-rated adjectives happen to be the less likely ones,
   that looks like "pull toward feminine" with no gender involved. Our frequency covariate
   (Zipf) catches part of this, but a word's probability *in this sentence* isn't its corpus
   frequency.
2. **The gender direction contains a "rare vs common word" axis.** Feminine and masculine noun
   sets could differ in how frequent or predictable the words are.

## (a) Baseline-probability covariate

For every scored row, the graded score (−β_GEND from regressing the shift on standardised
[GEND, VAL, AROU, SIZE, zipf] across adjectives) gets one more covariate: the adjective's
**midpoint log-probability**, the average of its log-probability in the two conditions being
compared, standardised within the row.
- **Why the midpoint, not the unsteered value:** the shift is (B − A). Using A as a covariate
  puts A's idiosyncrasies into both sides with opposite signs, which manufactures a negative
  slope (regression to the mean). The midpoint (A + B)/2 is uncorrelated with that error when
  the two conditions are equally noisy.
- The regression is fitted per row because the midpoint differs from prompt to prompt; the
  rating columns are as before.

Applied, unchanged otherwise (same rows, wordings, frame-check exclusions, bootstraps, seeds), to:

| Result | Conditions A → B | Pre-registered statistic re-reported |
|---|---|---|
| Phase 5 Spanish, 8B and 14B, R1 and R1-EN, every steering layer | unsteered → steered at α\* (real and all 100 random directions) | working test: effect, random 95th percentile, p, diff CI; per-layer pass |
| P29 (8B, 14B) | *El* → *La* | M1/M2/M3 contrasts |
| P30 (all models run) | *El* → *La* | S(object), S(neutral), S(person), verdict |
| P31 (all models run) | *der* → *die* (and → *das*) | S(context), verdict |

**Reading (fixed now):** a result is **robust to flattening** if, with the covariate, its
pre-registered verdict is unchanged *and* its point estimate keeps ≥ 50% of its size with the same sign.
Otherwise it's reported as **flattening-sensitive**, and the original result carries that
caveat. This is a diagnostic: it can't turn a null into a positive. The original rules stand.

## (b) Does gender rating track baseline probability?

Per model × readout (R1, R1-EN, R3 invariant; P30/P31 readouts): the Spearman correlation,
across adjectives, between the GEND rating and the adjective's mean unsteered log-probability
(averaged over that readout's primary-wording prompts), plus the same after partialling out the
other covariates (VAL, AROU, SIZE, zipf). Descriptive. It says how much room route 1 has: near 0
means flattening can't masquerade as a gender shift whatever it does.

## (c) Gender direction vs frequency direction

At each Phase 5 steering layer (25/40/55/70% depth), on the 8B and 14B activations, for Spanish
(Phase 2 `strat` + `matched`) and German (Phase 3 `strat3` m/f): the steering vector is the
gender coefficient of a least-squares fit of the activations on [gender, ending cells,
covariates], and the covariates already include Zipf frequency. So the **frequency direction**
is the Zipf coefficient of *that same fit*. No new stimuli or choices are needed.

- **Estimate:** cross-fitted cosine, as `estimators.split_half_geometry`: gender coefficient from
  half A with Zipf coefficient from half B (and vice versa), over squared lengths that are also
  cross-fitted, 200 splits within cell × gender strata. A plain cosine of two vectors fitted on
  the same nouns is biased by shared noise (project gotcha).
- **Comparators:** the same cross-fitted cosine of the **number direction** with the Zipf
  coefficient (on the singulars' fit), and a null with Zipf permuted within cell × gender
  (200 permutations, 95th percentile of |cos|).
- **Reading (fixed now):** **frequency overlap** at a layer if |cos| > the null's 95th percentile
  *and* |cos| ≥ 0.3.
- **Amendment (2026-10-21, before the full run):** the null is computed on the **cross term**
  (g^A · f^B + g^B · f^A)/2, and the observed cross term must exceed its 95th percentile (20
  splits per permutation). A smoke run (8B, 10 splits, 5 permutations) gave null "cosines"
  above 1. The permuted Zipf coefficient has squared length ≈ 0, so a cosine null divides by
  noise. That is the same ill-conditioning as Phase 4's shuffle null (fixed the same way,
  2026-10-10). The magnitude bar (|cos| ≥ 0.3) is unchanged. Disclosure: the smoke run's 8B
  cosines were seen (10 splits; Spanish |cos| ≤ 0.13, German −0.35 to −0.71); no rule was
  changed in response to them. A cosine of 0.3 means about 9% of the direction's squared length lies along
  frequency, roughly the size of the cross-language gender overlaps we have treated as real.
  Report the per-layer table. A frequency overlap at a majority of steering layers is a caveat on that
  model × language's steering results. Below the bar, report it as a clean control.

## Not adopted from the spec (why)

- **The cloze "confidence direction" (new stimuli, pod).** As specified, it contrasts states at
  different tokens (*Paris* vs a rare noun), so it would mostly be a word-frequency direction.
  (c) covers that more directly. If done later, hold the token fixed and vary the context.
- **"Above the shuffle null at most layers" as the caveat rule.** Nearly any nonzero cosine
  passes a pure-noise null with 150 items; (c) adds a magnitude bar.
- **Kendall τ.** At our doses rankings barely change under any direction (τ ≈ 1 everywhere);
  the graded regression already tests the gender-specific reordering. List entropy at α = 0
  and α\* is reported descriptively in (a)'s Phase 5 table at no extra cost.

## Practicalities

Code: `src/gbleed/confidence_ctl.py`, CLI `gbleed confidence-ctl`. Reads the Phase 5 sweep arrays and the
activations through `paths`. Results to `results/confidence_ctl/` (small tables). Committed
before it is run.

## Results (2026-10-21/22; `results/confidence_ctl/`)

**(a) Midpoint log-probability covariate: every result is robust.**
- **Phase 5 Spanish (8B, 14B; R1 and R1-EN; noun-only and every-position):** the same layers
  pass with and without the covariate. 8B noun-only: R1 3/4, R1-EN 3/4 (majority). 14B: 0/3.
  Effects move by < 3% (e.g. 8B noun R1, mean over layers, 0.0203 → 0.0201; random 95th
  percentile 0.0120 → 0.0121).
- **P29, P30, P31 (every model run):** every verdict unchanged. Every estimate keeps its sign and
  ≥ 50% of its size (most move < 15%; P30 1.7B S(object) rises 0.038 → 0.052). The three rows
  the rule marks "not robust" are *das* contrasts estimated at 0.000 before and after, where
  "keeps half of zero" is meaningless.

**(b) Gender rating vs baseline probability:** Spearman −0.10 to −0.32 (partial −0.04 to −0.25).
More masculine-rated adjectives are a little less likely at baseline, most of all on the
invariant lists that P29/P30 use. So flattening *could* have masqueraded as a gender shift
(in the masculine direction), and sharpening as a feminine one. (a) shows that this channel
doesn't drive any result.

**(c) Gender direction vs frequency direction:**
- **Spanish: no overlap** at any layer (|cos| ≤ 0.14; number direction ≤ 0.09).
- **German: frequency overlap.** 8B at 4/4 layers (−0.34 to −0.69), 14B at 3/4 (−0.34 to −0.46);
  the number direction is ≤ 0.17. By the pre-registered rule, German results built on the gender
  direction carry the caveat. Followed up in `frequency-axis-prereg.md`.

**Summary for the write-up:** the confidence/flattening concern of Buttigieg et al. doesn't
explain any of our steering or article results. Their mechanism (a direction that mostly
changes how confident the model is) would show up in (a), and it doesn't. Separately, the German
gender direction is entangled with word frequency, a finding about German geometry rather than
about confidence.
