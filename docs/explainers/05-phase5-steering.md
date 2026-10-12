# Phase 5 explained: pushing a noun's grammatical gender and watching what moves

*Code: `src/gbleed/steering.py` (the split forward pass), `phase5.py` (gate, sweeps, readouts,
damage), `phase5_stimuli.py` (word lists), `phase5_analysis.py` (scores, tests, verdicts).
Design and every pre-registered choice: `docs/design/phase5-design.md` (P1–P27) and
`docs/decisions.md`. Terms in **bold** are in the [glossary](../glossary.md). Nothing in this
file is a result yet: the 4B gate is a laptop development run, and the confirmatory runs are
8B/14B in cloud session 2.*

## The question

Phases 2–4 found a direction in the model's activations that sorts nouns by grammatical gender
(*la luna* vs *el puente*). Phase 5 asks whether that direction does anything *social*: if we
push the representation of *bridge* along the "feminine" direction, does the model start
describing it with words people rate as feminine (*elegant*, *delicate*) more than a push of the
same size in a random direction would? This is the model-internal version of Boroditsky's
bridge study, with the advantage that we can intervene instead of only correlating.

## How a push works (steering)

At a chosen layer, the model's internal state at the noun's tokens gets a vector added:
*state + α × v*, where *v* is the unit-length gender direction (feminine minus masculine
**difference of means**, fitted on the Phase 2/3 training nouns, never on the steered nouns) and
α is the **dose**. Everything after that layer runs as normal, so later words "read" the edited
noun through attention.

- **Dose units:** α is measured in multiples of the typical size of a noun's internal state at
  that layer (the median **norm**). α = 0.2 means a push of 20% of a typical state.
- **Only the noun is pushed** (all its tokens). Pushing every position would act directly on the
  word being predicted, so a change could happen without the noun playing any role. That
  version is run too, as a secondary test of a different claim (below).
- **Speed trick:** the layers below the push are computed once per sentence; only the layers
  above it are re-run for each condition. Identical to the plain method (checked to rounding),
  8–30× faster.

## Step 1: the gate (does the push do what it should grammatically?)

Before any social readout, the push must flip **grammatical agreement**; otherwise it isn't on
the pathway the model uses for gender.

- **Spanish:** *Mi {noun} es muy ___* (*mi* is the same for both genders), scored on adjective
  pairs that differ only in the ending (*blanco/blanca*). Pushing a masculine noun feminine should
  make *-a* endings likelier.
- **German:** the dictionary frame *Wörterbuch:\n{Noun},* followed by *der* or *die*,
  **calibrated** against a meaningless headword (*N/A*) so the frame's own lean toward *die*
  (also the plural article) cancels. Chosen from five pre-declared candidates by a pre-declared
  rule.
- **Pass rule:** the flip dose **α\*** is the smallest dose at which the median noun has flipped;
  20 random directions of the same size must flip fewer than 10% of nouns at α\*.
- **Frame check first:** unsteered, each model must get agreement right for ≥ 70% of its known
  common nouns per gender in that language's gate frame; a language that fails gets no gate and
  no steering for that model (why 1.7B has no German steering).
- **Layers:** four fixed depths (25/40/55/70%). Every layer that passes is used (PI), and a
  result counts only if it holds at a majority of them (below).

Laptop development run, 4B: Spanish passes at all four depths (α\* = 0.2, random flips 0%);
German passes at 25/40/55% (α\* 0.75/0.4/0.3) and fails at 70% (random directions flip 10%).

## Step 2: the readouts (does anything social move?)

Nothing is generated; each readout scores a fixed list of next words, so every number is an
exact probability (**scoring, not sampling**).

| readout | sentence | scored words | status |
|---|---|---|---|
| R1 | *The Spanish word "puente" means bridge. Described in one word, it is very ___* (3 wordings averaged; plus W4 *Everyone says…* and a no-*very* wording as checks) | ~650 English adjectives with human gender ratings (Glasgow Norms) | co-primary |
| R1-EN | *The bridge is very ___* (3 wordings; same two checks) | same | co-primary |
| R2 / R2-EN | *If the bridge were a person, would it be a man or a woman? It would be a ___* (3 framings × 2 orders), plus a story frame *…came to life. Every morning, ___* | man/woman, male/female, boy/girl; story: she/he (it/they recorded) | secondary |
| R3 | *Mi puente es muy ___* | gender-invariant Spanish adjectives (*fuerte*, *elegante*) | secondary |
| R3-POOL (P28) | same sentence | gendered Spanish adjectives read as the lemma, P(*blanco*) + P(*blanca*), so agreement drops out (Qwen3: 116) | secondary, exploratory |
| R-NONCE | R1-EN/R2-EN frames with invented words (*The sianta is very ___*; 600 Spanish-looking, 100 English-looking) | as R1-EN/R2-EN | secondary (cheaper set: ±1 × α\*, 20 random directions) |

**Readout frame check:** before steering is analysed, each wording must put a clear share of
the next-word probability on its scored words, unsteered (median ≥ 5% for adjective lists, ≥ 25%
for the two named options); a failing wording is dropped (thresholds fixed 2026-10-10, before
any share was seen).

**Which nouns:** every noun of the steering language: Spanish 83 steering nouns + 271 flipped-pair
nouns + 16 classics; German 100 + 271 + 16. The English readouts use their English concepts.

**The graded score.** Gender ratings and pleasantness are tangled in the Glasgow Norms
(feminine-rated adjectives are much more positive, r ≈ −0.47), so "more feminine adjectives"
could just mean "nicer adjectives". For each steered sentence, the change in every adjective's
log probability is **regressed** on its gender rating *with valence, arousal, size and frequency
as covariates*; the gender coefficient is the score. Positive = probability moved toward
feminine-rated adjectives, holding pleasantness fixed.

## Step 3: the controls (is it gender, or any push?)

- **Random directions:** 100 random unit vectors at the working dose on every sentence. The test
  asks whether the real direction's effect beats them: the **rank p-value** is the share of
  random directions doing at least as well (so 100 directions allow p down to 0.01, which the
  Holm correction needs). Twenty of them also run across all doses, on a fixed quarter of the
  nouns, to draw the random dose curve (trim adopted 2026-10-09; the curve is descriptive).
- **Damage:** how much a push disturbs the model in general, as the **KL divergence** of its
  next-word predictions on a neutral sentence containing the noun. The **working window** is
  the set of doses where the real direction does no more damage than random directions do at
  α\*, so effects are compared at matched disruption. Exploratory addition (2026-10-10): the 20
  dose-curve random directions also run at 1.5 and 3 × α\*, so a random direction can be read at
  the dose where *its* damage equals the real direction's at α\*.
- **Number direction:** a plural-minus-singular direction from the same training nouns (151 Spanish, 46 German), pushed the same
  way. If it moves the adjectives as much as gender does, the effect is "any grammatical push".
- **Social direction (positive control):** the woman-minus-man direction from English person
  words. Pushing *bridge* along it must move the readouts; if not, the readouts are insensitive
  and a null result is uninformative. Its per-adjective profile is also compared with the
  grammatical direction's: does the grammatical push move associations *the way* social gender
  does?
- **Cross-language direction (P26, exploratory):** the *other* language's grammatical direction
  (German's when steering Spanish nouns, and vice versa), same doses. Phase 4 found the two
  directions partly shared; does the German push do to *el puente* what the Spanish push does?
- **Size control:** the same regression's size coefficient should stay near zero.

## Step 4: the verdict

- **Per layer:** R1 and R1-EN each get the rank p-value and a **bootstrap** interval (nouns
  resampled) for real-minus-random; **Holm** across the two (the stronger must reach p ≤ 0.025
  with a 97.5% interval above 0; then the other p ≤ 0.05 with a 95% interval).
- **Across layers:** a readout counts as positive for a language only if it passes at a
  majority of that language's gate-passing layers; every layer is reported (PI, adopted before
  any data).
- **Across languages:** a "bleed" verdict needs Spanish and German to agree in sign; one alone is
  "language-specific".
- **Every-position steering** answers a different question (does the *vector itself* carry
  social content?), with its own confound (the push acts directly on the predicted word), so it
  is reported separately and labelled by the claim it supports.

## Side questions answered with the same runs

- **P25:** unsteered, do English concepts already lean toward one translation's gender? On the
  flipped pairs (*moon*: *la luna*/*der Mond* vs *bridge*: *el puente*/*die Brücke*). Net
  contrast only: equal leaks from both languages would cancel.
- **P20 (R3-PAIR):** *El puente es muy ___* / *Die Brücke ist sehr ___*, unsteered, on the
  flipped pairs: the in-language Boroditsky question, concept held fixed within a pair.
- **P21:** the top 50 next tokens of every scored sentence, steered or not, at every vector and
  dose, are saved, to see what the model would have said beyond the shortlist (browsing data,
  no test).
- **P22:** the model's own woman↔man axis in its output vocabulary (*woman − man*,
  *mother − father*, in each language), checked against the Glasgow ratings. No steering.
  Laptop result (1.7B/4B, exploratory): English agrees weakly (ρ ≈ 0.18 with valence partialled
  out); Spanish and German translations ≈ 0, which weakens R3/P20's use of English ratings for
  Spanish (a crude probe: two word pairs, first-token vectors; 8B/14B in session 2).

## Limits worth stating

- A push along one linear direction at one place. A null doesn't rule out other routes; a
  positive shows the direction is *sufficient* to move associations, not that the model uses it
  that way unprompted.
- Glasgow ratings are UK English raters' associations; R3 and P20 apply them to Spanish and
  German words through translation (no native adjective gender norms exist).
- The translation lists come from Wiktionary glosses by a strict automatic rule; a few wrong
  senses remain (*tiny* → *meñique*).
- **Single tokens only (P16):** the steered readouts score only words that are one token, so
  everything is one forward pass. English loses little (656 of 739 adjectives), but Spanish R3
  keeps 51 of 110 invariant adjectives, skewed toward short Latinate words (*natural*, *cruel*;
  R3-POOL adds 116 gendered ones, read through the first token their two forms share);
  tokenisation costs non-English readouts more. P20 (unsteered) scores whole words, all 110.
  Nothing is generated: what the model would say over several tokens under steering is not
  measured (a candidate for later, P21 note).
- German has no frame where a noun's gender can be pushed against a gender-neutral determiner,
  so R3 is Spanish only.
