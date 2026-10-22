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
  *and* |cos| ≥ 0.3. A cosine of 0.3 means about 9% of the direction's squared length lies along
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
