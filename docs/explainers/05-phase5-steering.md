# Phase 5 explained: pushing a noun's grammatical gender and watching what moves

*Code: `src/gbleed/steering.py` (the split forward pass), `phase5.py` (gate, sweeps, readouts,
damage), `phase5_stimuli.py` (word lists), `phase5_analysis.py` (scores, tests, verdicts).
Design and every pre-registered choice: `docs/design/phase5-design.md` (P1–P27) and
`docs/decisions.md`. Terms in **bold** are in the [glossary](../glossary.md). Confirmatory
results (8B/14B, cloud session 2, L40S bf16) are in [Results](#results-session-2-confirmatory-8b14b);
the 4B numbers elsewhere are laptop development runs.*

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

## Results (session 2, confirmatory, 8B/14B)

**German: no valid test on either size.** The pre-registered German gate frame (C4) failed: on
8B it failed its frame check (unsteered, the feminine nouns were read correctly only 62% of
the time, below the 70% bar); on 14B it passed that check but failed the gate at every layer.
On 14B the push didn't swap genders, it collapsed both toward neutral: feminine nouns' scores
crossed zero, masculine nouns' stayed negative. The secondary frame (C1) passed, but switching to it after seeing
C4 fail would be choosing a frame on the model's scores, so (PI decision, 2026-10-13) German
is reported as "no valid confirmatory test" and **every verdict below is Spanish-only**. At most
"language-specific"; "bleed" needed both languages.

**Gate (Spanish):** the push flips the median noun's article at small doses on both sizes
(α\*, in multiples of a typical noun state's size at that layer: 8B 0.3/0.2/0.15/0.3
at layers 9/14/20/25; 14B 0.2/0.3/0.2 at 10/16/22; 14B layer 28 never flipped the median
within the random-flip limit and drops out).

**Readouts, noun-only steering (the pre-registered claim):** mean shift in log-probability
toward feminine-rated adjectives, at the gate dose, averaged over passing layers; "random 95%"
is the 95th percentile of 100 random directions of the same size.

| | 8B | 14B |
|---|---|---|
| R1 (*very ___*): grammatical push | **0.020** (passes 3/4 layers) | **0.000** (0/3) |
| R1: random 95% | 0.012 | 0.008 |
| R1: social push (control) | 0.064 | 0.047 |
| R1: number push (control) | −0.003 | 0.002 |
| R1-EN (English noun only): grammatical push | **0.011** (3/4) | **−0.001** (0/3) |
| R1-EN: social push | 0.033 | 0.029 |
| **Verdict** | language-specific effect | **no effect** |

How to read it: on 8B, pushing *el puente* toward feminine moves English adjective choices
toward feminine-rated words, about a third as much as pushing along the woman–man direction,
and plural/singular pushes don't. On 14B the same push still flips the article but moves the
adjectives **not at all**, although the social push shows the readout can move. So there's
no growth with size; if anything the link fades. One result at one size isn't a trend, and
the 8B effect is small (and largest at the 70% depth, where 14B couldn't be tested: see
Limits), so the safe summary is: **grammatical gender can be made to leak into
gendered adjective choice in one model size, weakly, and not in the larger one.**

- **R2 (the man/woman forced choice) moves on both sizes** (14B: all 3 layers above random). It
  shows the Spanish noun next to the question, so the model can answer "woman" just because
  the noun is grammatically feminine; R2-EN (English noun only) moved on 14B at 1 of 3 layers.
  Read R2 as "the model connects grammatical gender with man/woman words when it can see the
  noun", not as social association.
- **Cross-language (P26, exploratory):** the German direction pushed into Spanish nouns moves 14B
  R1 slightly above random at layers 10/16, while Spanish's own direction doesn't. Small, not
  pre-registered: a footnote.
- **Every-position steering** (pushing every token, a different claim): 8B passes R1 at 3/4
  layers, R1-EN 2/4 (not a majority); 14B 0/3 on both.
- **R3 (in-language Spanish adjectives)** failed its readout check on 8B (the listed adjectives
  took 2.9% of the probability, below the 5% bar) and is excluded.
- **R3-POOL (Spanish adjectives, pooled across agreement; exploratory).** R3 proper failed its
  check on both sizes (its 51 adjectives got 2.9% / 2.4% of the probability). R3-POOL got its
  own check, fixed before computing it (PI, 2026-10-15, a logged deviation: by the P20-POOL
  analogy it would have been excluded). It passes (18%). Result: **positive at every layer of
  both models**, noun-only 8B 0.05–0.07, 14B 0.04–0.05, against a random 95% of ≤ 0.01. The push
  moves the same adjectives the woman–man push moves (rank correlation 0.56–0.87; number and
  random pushes ≈ 0). So on 14B, a push that changes nothing in English *does* change Spanish
  adjective choice: the "inside the language but not reaching English" pattern. Three reasons
  for caution: it passed only a revised check; reading a shared first token (*blanc-*) is assumed
  not to depend on agreement, which wasn't tested; and the woman–man direction also makes a
  Spanish noun grammatically feminine, so the profile match doesn't prove the content is social.
- **P20 / P20-POOL (unsteered, flipped pairs).** Concept held fixed within a pair (*el puente* /
  *die Brücke*): does the grammatically feminine side draw more feminine-rated adjectives?
  Difference-in-differences, > 0 = yes. Invariant Spanish adjectives: 8B +0.056 [−0.009, 0.115],
  14B +0.032 [−0.024, 0.086]. Pooled gendered adjectives: **8B +0.103 [0.050, 0.155]** (also
  without cognates and without homographs), 14B +0.010 [−0.034, 0.056]. All of it is Spanish:
  German nouns' own feminine-minus-masculine contrast is ≈ 0 on both. A frequency confound
  (feminine forms of feminine-rated adjectives being commoner) was checked and isn't there.
- **P25** (unsteered lean of English concepts toward a translation's gender): the difference is
  near zero on both, with intervals covering 0 (8B −0.03 [−0.06, 0.01]; 14B −0.03 [−0.07, 0.00]).

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
  Mostly a measurement artefact: see P22b.
- **P22b:** the same question asked in sentences, with whole words: how much more likely is
  the adjective after *The woman is very* than after *The man is very* (and mother/father), in
  each language. Gendered Spanish forms (*blanca*/*blanco*) use a double difference: the same
  gap after feminine vs masculine inanimate nouns is subtracted, so only the woman/man part is
  left. Laptop result (ρ with Glasgow, raw / pleasantness removed; 1.7B, 4B):

  | | 1.7B | 4B | 8B | 14B |
  |---|---|---|---|---|
  | English (656) | 0.59 / 0.57 | 0.52 / 0.52 | 0.51 / 0.49 | 0.57 / 0.56 |
  | German (238) | 0.43 / 0.36 | 0.36 / 0.32 | 0.28 / 0.32 | 0.39 / 0.43 |
  | Spanish invariant (110) | 0.45 / 0.28 | 0.25 / −0.01 | 0.27 / 0.04 | 0.34 / 0.12 |
  | Spanish gendered, double (157) | 0.24 / 0.19 | 0.21 / 0.07 | 0.23 / 0.08 | 0.21 / 0.19 |

  English and German track the human ratings at every size, so English ratings on German
  translations are reasonable. Spanish stays weak at every size: once pleasantness is removed
  (woman frames favour nicer words), little Glasgow gender is left (8B/14B are bf16 cloud runs;
  1.7B/4B laptop). Spanish R3/P20 results carry that caveat.

## Limits worth stating

- A push along one linear direction at one place. A null doesn't rule out other routes; a
  positive shows the direction is *sufficient* to move associations, not that the model uses it
  that way unprompted.
- Glasgow ratings are UK English raters' associations; R3 and P20 apply them to Spanish and
  German words through translation (no native adjective gender norms exist).
  P22b checks this: it holds for German, only weakly for Spanish (gone on 4B once
  pleasantness is removed).
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
- **The confirmatory verdicts are Spanish-only** (German's pre-registered frame failed on both
  sizes), so the cross-language half of the question is unanswered: "bleed" needed both.
- **14B wasn't tested where 8B's effect was largest** (flagged by a side agent, 2026-10-15). Steering
  layers sit at fixed relative depths. 8B's R1 effect grows with depth (25%: 0.004, none; 40%:
  0.018; 55%: 0.020; 70%: 0.039), but 14B's 70% layer (k28) failed the gate and dropped out. At
  the depths tested on both (40%, 55%), 8B moves and 14B doesn't, so the 14B null is real there.
  "8B yes, 14B no" is a statement about the middle of the model, not about size in general.
- **Two sizes, opposite answers.** 8B shows a small effect, 14B none. With one model family and
  two confirmatory sizes, "size-dependent" is a description, not an explanation; a third family
  (e.g. EuroLLM) would be needed to say more.
- A gate dose is chosen per layer (the smallest that flips the median noun), so 14B's readouts
  were measured at doses similar to 8B's (0.2–0.3 of a typical state). A larger push might move them; the
  pre-registered test asks whether the push that *just* changes the grammar also changes
  associations.
