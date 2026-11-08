# P29: is the pooled adjective readout agreement-neutral? (ADOPTED, PI 2026-10-17)

Drafted and adopted 2026-10-17 (M0–M3 as written, bound at half the smallest effect, the 14B
deep-layer run kept separate). Code: `src/gbleed/p29.py`, pod script `runpod/p29.sh`. Thresholds are fixed here, before any data,
from the R3-POOL results already on file (`results/runpod2/phase5_analysis/*/r3_pool.csv`).

## The question in plain words

R3-POOL reads Spanish gendered adjectives as *lemmas*: *blanco* and *blanca* count as one word,
so that when steering flips agreement (probability moving from *blanco* to *blanca*) the total
shouldn't change. Only a change in *which adjective* the model picks should register. On both
8B and 14B, the grammatical push makes the model prefer adjectives that English speakers rate
as feminine (noun-only effect 0.052–0.075 on 8B, 0.038–0.045 on 14B; random directions' 95th
percentile about 0.006–0.010).

That reading rests on an untested assumption: **pooling really is neutral to agreement.** If
it isn't, part of the effect could be the agreement flip itself leaking into the score. Two
ways that could happen:

1. **Tokenisation proxy.** For 106 of the 116 lemmas we don't read the two words, only their
   shared first token (*blanc*-). That token also starts other words, and its probability need
   not move exactly like P(*blanco*) + P(*blanca*). Only 10 lemmas are exact sums (both forms
   single tokens). The exact-sum subset mostly *doesn't* show the effect: noun-only it
   passes at 1/4 layers on 8B and 1/3 on 14B. With ten items it has little power (random
   95th percentile ~0.03 vs ~0.007), but it's a reason to look.
2. **Form availability.** Once agreement demands the feminine form, the model may favour
   lemmas whose feminine form is common in text. If those happen to be feminine-rated, the
   score moves with no change in meaning. Example: *calvo* "bald" (rated 6.46 of 7, where 7 =
   very masculine) has a feminine form, *calva*, about 6× rarer than the masculine. In text
   counts (wordfreq) this is unlikely to drive the result: over the 116 R3-POOL lemmas,
   ρ(rating, zipf(f) − zipf(m)) = +0.15, p = 0.12, the opposite sign from the worry. Most large
   gaps are unrelated to gender (*nublado*, the impersonal weather word; *Atlético*, the club).
   Text counts aren't the model's behaviour, though, so the test measures it directly. Where
   rarity is itself social (*calva*), avoiding the form would arguably be a shallow, word-form
   version of the association. M2 helps here too, since invariant adjectives have no form to be
   rare.

## Design

Change agreement **without** changing meaning, and see whether the R3-POOL score moves.

- **Items: nonce nouns**, so there's no meaning to change. The R-NONCE Spanish set
  (`phase5_nonce_es_v1.csv`, 150 stems, already screened as non-words): primary set =
  the **-e and -iz forms** (300 words). Those endings don't settle gender, so the article does.
  Sensitivity set: the -o/-a forms (300). There the ending conflicts with the article half the
  time.
- **Frames:** the P20 Spanish wordings with the article as the only gender cue:
  W1 *{El/La} {N} es muy ___*, W2 *Es {un/una} {N} muy ___*. Every word appears with both
  articles. Primary = mean of the two wordings (as P20).
- **One unsteered forward pass per prompt**, next token after *muy*, reading:
  - the R3-POOL lemma ids (exactly the R3-POOL columns and pooling function);
  - the R3 invariant adjectives (*grande*, *feliz*, ...: one form for both genders);
  - the gate adjectives (-o/-a pairs), for the manipulation check;
  - plus whole-word scoring of both forms of every pooled lemma, for the fidelity check (M3).

## Measures and rules (pre-registered)

**M0, manipulation check (validity, as the gate).** Agreement margin = log P(feminine forms) −
log P(masculine forms) over the gate adjectives. Valid test iff the margin is higher under
*La/una* than under *El/un* for ≥ 90% of the primary nonce words (per wording, then
the mean), with the median margin > 0 under *La* and < 0 under *El*. Fail → no valid
test (like German Phase 5), reported as such.

**M1, primary.** A = mean over nonce words of the R3-POOL graded score (−β_GEND, same
regression and covariates as R3-POOL, `all` subset) of the shift *La* minus *El*. 95% CI:
bootstrap over stems (a stem's -e and -iz words stay together).
- **Pass (agreement can't explain R3-POOL)** iff A's upper 95% bound < **half** of that
  model's smallest noun-only R3-POOL effect: **8B < 0.0260** (half of 0.0520, k20), **14B <
  0.0188** (half of 0.0376, k10). Exact values are in `p29.R3_POOL_MIN_EFFECT`.
- Why this is conservative: the article flips agreement in nearly every word (M0 checks
  this). Steering at α\* flips it in only about half the nouns, since α\* is defined as the
  dose where the median agreement margin crosses zero. So A overstates how much agreement
  could contribute to R3-POOL.
- Also reported: the `both_single` and `no_homograph` subsets, and the -o/-a set.

**M2, secondary (interpreting a failed M1; exploratory).** A_inv = the same graded score on
the invariant adjectives, under the same *La* − *El* contrast. Invariant adjectives have no
agreement to leak, so a shift there is the article's own association (a grammatical-to-social
link, i.e. bleed-like), not a pooling artefact.
- If A > 0 and A_inv ≈ 0: **mechanical leakage**. R3-POOL can't be read as evidence about
  meaning.
- If A_inv is about as large as A: **the article itself carries the association**. That's
  substantively interesting (P20-POOL's 8B result points the same way), but it means R3-POOL
  doesn't separate "steering grammar" from "grammar implies social gender".
- The adjective sets differ, so this comparison is rough. The invariant readout also failed
  R3's frame check (share < 5%), so it's noisy; its share here is reported.

**M3, fidelity diagnostic (reported, not a gate).** On the shared-first-token lemmas, compare
the *La* − *El* shift read from the first token with the shift from the whole words
(log[P(m) + P(f)]): correlation across lemmas, and A recomputed from whole-word pooling. If
whole-word A ≈ first-token A, the proxy behaves like the real lemma probability under an
agreement change.

## What each outcome means for the write-up

| M0 | M1 | Reading |
|---|---|---|
| fail | n/a | No valid test. R3-POOL stays caveated as now. |
| pass | pass | A full agreement flip moves the score by less than half the steering effect. R3-POOL is not an agreement artefact. |
| pass | fail | Agreement (or the article's own association) could account for R3-POOL. M2/M3 say which. R3-POOL is reported as uninterpretable on its own. |

## Hardware and cost

Local smoke test (0.6B, fp16, 4 stems) ran end to end on 2026-10-17. On 0.6B the article
already moves the pooled score by ~0.07; that's a different model at n = 8, not evidence about
8B/14B, and no rule was changed after seeing it.

8B and 14B don't fit on the Mac in fp16 (8B alone is ~16 GB), and R3-POOL ran in bf16 on an
L40S. So: one RunPod L40S session, both models, unsteered scoring only (~1,200 prompts per model
plus ~280k whole-word continuations). Estimated well under an hour, about $1 at session 2's
rate; the live price will be stated before creating the pod. Smoke test on 0.6B locally first.
The code is new (`p29` step), committed before the run.

## Decisions (PI, 2026-10-17)

Adopted as recommended:

1. M0–M3 as written.
2. Pass bound at **half** the smallest R3-POOL effect. Alternatives not taken: below the random
   95th percentile (stricter), below the full effect (looser).
3. The 14B deep-layer (k28) run stays a separate decision.

## Results (2026-10-17; L40S bf16, commit c48d86b; `results/p29/`)

| | 8B | 14B |
|---|---|---|
| M0: words where the article flips agreement | 100% | 100% |
| **M1: A (all), primary** | **+0.096 [0.090, 0.103]** | **+0.038 [0.033, 0.043]** |
| bound (half the smallest R3-POOL effect) | 0.0260 | 0.0188 |
| **M1 verdict** | **fail** | **fail** |
| A, no homographs | +0.111 | +0.048 |
| A, exact sums (10 lemmas) | −0.083 | −0.078 |
| A, -o/-a set | +0.083 | +0.030 |
| M2: invariant adjectives, A_inv | +0.028 [0.018, 0.039] | +0.082 [0.073, 0.091] |
| invariant share (median) | 4.6% | 3.2% |
| M3: A from whole-word pooling | +0.120 | +0.060 |
| M3: ρ first-token vs whole-word lemma shifts | 0.46 | 0.57 |

**Reading (as pre-registered: M0 pass, M1 fail):** a grammar-only change produces a shift as
large as R3-POOL's steering effect, so R3-POOL is uninterpretable on its own.
- **M3:** the first-token proxy isn't the cause; whole words shift more.
- **M2:** invariant adjectives shift too, so part of the effect is the feminine article's own
  pull toward feminine-rated words, not leakage through word forms. That holds for most of it
  on 14B, about a third on 8B.
- **Not anticipated in this design:** nouns in *-e*/*-iz* are often person nouns
  (*la estudiante*, *la actriz*), so *la* + unknown noun may read as "a woman". By ending, the
  invariant shift is +0.027/+0.029/+0.017/+0.018 for -e/-iz/-a/-o on 8B and
  +0.083/+0.081/+0.054/+0.068 on 14B. Somewhat larger for -e/-iz, but present for -a/-o too, so
  this can't be the whole story, and the design can't separate it.
