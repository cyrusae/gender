# Phase 5 design: steering and social gender (adopted 2026-10-08, PI)

*Status: adopted 2026-10-08; PI decisions (P1–P15, all adopted 2026-10-08). It turns the Phase 5 outline in
`phases-3-5-plan.md` (§5.0–5.6) into concrete stimuli, metrics and a compute plan. Nothing here
has been run. Exploratory development on the Mac (1.7B/4B); confirmatory runs on the next A100
round (8B/14B). Phase 4 (cross-language) is analysis plus one extraction, so it rides along with
the same round.*

## The question in one line

If we push a noun's representation along the grammatical-gender direction, does the model's
*social* gender content about that noun move with it (stereotyped adjectives, personification),
beyond what any push of that size does?

## Scope of the first Phase 5 round (recommended; P1)

**In:**
- §5.2 agreement gate
- §5.0 baseline
- §5.3 dose-response with damage
- §5.4 specificity (number direction, Glasgow gradient, size control)

**Later:**
- §5.6 erasure arm
- epicene social-gender directions (§5.1)
- the genderless-language control

Each of those needs its own stimulus design, and the core result decides whether they're worth
it.

## The steering vector v (P2)

- **Spanish:** the stratified *difference of means* (`class_betas` on the `strat` set,
  residualised as in Phase 2), unit length.
- **German:** the same on `strat3`'s masculine and feminine nouns (a single m–f axis).
- **Why the difference of means and not the probe weights:** a probe's weights point wherever
  classes separate best given the noise, which can be nearly orthogonal to where the class means
  actually differ. Steering means moving the mean, so the difference of means is the standard
  choice (Marks & Tegmark 2023 found it causally more effective). The probe direction is reported
  as a secondary vector.
- **German separate vectors (masculine and feminine against neuter):** used only if the 8B/14B
  cosine is interpretable and clearly away from −1 (pre-registered Phase 3 reading). Otherwise
  one axis.
- **Dose units:** α in multiples of the layer's median residual-stream norm at noun positions
  (α = 0.1 means a push of 10% of a typical activation).

## Where to steer

**Why the noun only (PI question, 2026-10-08).** Activation-addition methods (Turner et al.
2023; contrastive activation addition, Rimsky et al. 2024) add the vector at every position,
because they aim to change overall behaviour. That would push the readout position itself
toward "feminine": priming the whole model, which moves adjectives without the noun playing any
role. The question here is whether the *noun's* grammatical gender changes what the model
associates with it, so only the noun is edited; later positions read it through attention.
The cost is a possibly weak or self-repaired edit, which is what the agreement gate checks.

- **Position:** only the noun's last token, in every prompt. (P13: all of the noun's tokens
  instead, for multi-token nouns and nonce words.)
- **Every-position steering (P14, amended 2026-10-08 after PI challenge): secondary test of
  vector-level overlap.** The first version called this "descriptive only, not evidence of
  bleed"; that was too strong. A noteworthy all-token effect shows the grammatical-gender
  vector (learned from inanimate nouns only) carries something that moves social-gender
  associations: the project's question in its *representational* form, a causal cousin of the
  §5.1 cosine. Noun-only steering answers the narrower Boroditsky-shaped question (does the
  noun's grammatical gender shape associations with *that noun*?). Both are reported, labelled
  by which claim they support.
  - Its specific confound: at the readout position the vector acts directly on the next-word
    choice, so features that go along with gender *within the training nouns* (not language
    identity, which a feminine-minus-masculine difference cancels) reach the output directly.
    E.g. if feminine training nouns lean learned/Latinate, the push could favour Latinate English
    adjectives (*elegant*, *delicate*), which are often feminine-rated, while Germanic ones
    (*strong*, *hard*) are often masculine-rated.
  - Controls: (1) **etymology covariate** in the R1 regression: each adjective's origin (Latin/
    Romance borrowing vs native English, from Wiktionary's English etymologies, automatic);
    (2) **Spanish and German vectors must agree in sign** (PI): flavour confounds predict
    different directions for the two languages (German is English's own family: a "Germanic
    flavour" push would favour masculine-rated vocabulary), real bleed predicts the same;
    (3) the **pooled vector** (Phase 4; centred within language, so language-specific
    correlates are diluted) as a third steering vector *if* Phase 4 finds the directions
    shared; (4) the shared controls: random directions, number direction, matched damage,
    social-gender direction as reference profile.
- **Layers:** a small fixed grid (about 25%, 40%, 55% and 70% of depth). The Phase 2/3 directions
  read gender in most inner layers, so this isn't selection on the result.
- **Choosing among them:** the gate picks the working layer(s), using the *grammatical* outcome
  only, blind to the social-gender readouts. Choosing a layer on agreement is a manipulation
  check, not tuning on the outcome; the write-up says so.

## Nouns

Held-out inanimate nouns only, never the nouns v was fitted on:
- **Spanish:** Phase 2 regular test nouns.
- **German:** Phase 3 `compound_test` heads are excluded (the heads are the issue). A fresh
  held-out draw comes from the Phase 3 pool with the same filters (P3).
- **Both languages:** the Phase 4–5 flipped pairs (*la luna* / *der Mond*) and the classics,
  which are frozen.
- About 100 per language plus the pairs.

## §5.2 The gate: does agreement follow the push?

Both frames need a behavioural check before use (as in Phase 0/3): unsteered, ≥ 70% of known
Zipf ≥ 4 nouns get the right gender in every model × gender, on Qwen3 + EuroLLM. The frames are
fixed on grammatical grounds, not tuned on scores.

**Spanish**
- Frame: *Mi {noun} es muy ___*. *Mi* is the same for both genders, so nothing before the noun
  gives its gender away.
- Readout: a fixed list of *-o/-a* adjective pairs (*blanco/blanca*, *viejo/vieja*,
  *nuevo/nueva*, *bonito/bonita*, *alto/alta*, *barato/barata*…), each word scored in full;
  agreement margin = mean log P(feminine form) − log P(masculine form).

**German** (predicate adjectives don't inflect, so a different agreement target)
- Candidate A, dictionary entry: *Wörterbuch:\n{Noun}, ___*, then P(*der/die/das*). This is the
  common German dictionary convention ("Brücke, die"), and the noun comes before any article.
- Candidate B, pronoun after a bare noun: *Thema: {Noun}. ___ ist*, then P(*Er/Sie/Es*). Weaker:
  expletive *Es* competes.
- Plan: A primary, B secondary (P4).
- **Revised 2026-10-09 (PI) after the pre-registered frame check:** B fails (sentence-initial
  *Sie* is also "they"/formal "you") and A fails on 4B (masculine nouns lose to *die*, also
  plural). A label frame (*Genus:* maskulin/feminin/neutrum) was tried and failed too. Five
  candidates were then declared with a selection rule and tested (decisions.md, 2026-10-09).
  **Selected (PI): C4, the dictionary frame calibrated against a content-free headword
  (*N/A*), primary; C1 (uncalibrated) secondary.** 1.7B fails C4's check: no German gate there.

**Pass rule**
- Flip dose α\*: the smallest dose at which the median agreement margin crosses zero (feminine
  nouns pushed toward masculine, masculine toward feminine).
- Random directions (20, same norm) at α\* must flip < 10% of nouns.
- If the gate fails, stop and diagnose (the table in the plan, §5.2); no social-gender readouts.

## §5.0 Baseline, and the social-gender readouts

The same readouts serve unsteered (baseline) and steered.

**R1 (primary): English adjectives with human gender ratings**
- Frame (revised 2026-10-08, PI): *The Spanish word "{noun}" means {gloss}. Described in one
  word, it is very ___*. "Very" makes the slot grammatically an adjective; without it the model
  may expect a noun or "a", so all listed adjectives get little probability and steering could
  move the slot type rather than the adjectives. Secondary frame without "very" (covers
  non-gradable adjectives: *military*, *bridal*), and the R1 result is also reported on
  gradable adjectives only. The noun is steered at its own position. Only listed words are
  scored (log P of each full word); nothing is generated.
- Frame check (pre-registered, as in Phase 0): unsteered, the listed adjectives' first tokens
  must take a clear share of the next-token probability (threshold to fix before running), on
  Qwen3 and EuroLLM. Fixed on grammatical grounds, checked, not tuned on scores.
- Score: log P of each adjective in a fixed set from the **Glasgow Norms** (Scott et al. 2019:
  ~5,500 English words with human ratings of gender association, valence, size and more).
- Adjectives: the ~40 most feminine-rated and ~40 most masculine-rated, matched on valence and
  frequency. The selection is fixed by rating thresholds before any data; the lists are in English,
  so the PI can review them.
- **Checked 2026-10-08 (data in hand, see P6): matched lists are too thin; proposed revision.**
  The Glasgow gender scale runs 1 = feminine to 7 = masculine (*woman* 1.3, *man* 7.0). Among
  803 adjective-dominant words (WordNet), **gender rating and valence correlate at r = −0.47**:
  feminine-rated adjectives are much more positive (mean valence 6.8 vs 4.1; *beautiful*,
  *elegant* vs *aggressive*, *violent*). An unmatched feminine-minus-masculine score would partly
  measure "pleasant vs unpleasant". Matching 1:1 on valence and frequency leaves only **28 pairs**
  at strong ratings (≤ 3 / ≥ 5, Zipf ≥ 3), 94 at weaker ones (≤ 3.5 / ≥ 4.5), and the matched
  pairs drift to person-describing words (*bitchy*, *insecure*). **Proposal:** make the primary
  R1 statistic the *graded* version: every rated adjective with Zipf ≥ 3 (~600), each one's
  shift in log probability regressed on its gender rating **with valence, arousal, semantic size
  and frequency as covariates**. The gender coefficient is the effect, so valence can't stand in
  for it. The matched lists are kept as a secondary, easy-to-read version. This also merges the
  §5.4 "Glasgow gradient" into R1.

**R1-EN (restored 2026-10-08, PI): pure English, no foreign word.** This is the original design's
Phase 5 step 3 (*"The bridge was ___"*), which the first version of this draft had dropped.
- Frame: *The {English noun} is very ___* (*The bridge is very ___*), the original Boroditsky
  bridge-study shape. The English noun's last token is steered along the Spanish (or German)
  grammatical-gender direction; scoring as R1 (Glasgow-rated adjectives, graded regression).
- What it tests: English has no grammatical gender, so nothing can "agree". A shift means the
  grammatical-gender direction carries social-gender content into a concept with no gender of
  its own: the cleanest form of the project's question. It also removes R1's possible
  translation-context confound (a feminine Spanish word in the prompt).
- Calibration: the gate can't run in English (no agreement), so layer and dose come from the
  Spanish/German gate; English damage is checked separately (KL on neutral English text).
- Reading a null depends on Phase 4: a Spanish-learned direction may not mean anything in an
  English context, so "no shift" is "no bleed" only if Phase 4 shows the direction is shared
  across languages; otherwise "direction doesn't transfer".
- Nouns: the English concepts, including the flipped pairs' glosses; steering the same English
  noun with Spanish's vs German's direction is a within-concept contrast. No unsteered
  baseline effect exists (English nouns have no gender): intervention-only.
- Intensifier: *very* (neutral, guarantees an adjective slot). Rejected: no intensifier (slot
  goes to "a", "located"), *looks* ("like"), *seems* ("to"), *so* (emotional colouring),
  *quite* ("fairly" in UK English), *It is a very ___ bridge* (noun after the blank, nothing to
  steer). Secondary frame without *very* and gradable-only reporting, as for R1.

**R2 (secondary): personification** (revised 2026-10-08, PI: an open pronoun slot can go to
*it*/*they*, which carry no gender information)
- Primary frame, forced choice: *If the {gloss} ({noun}) were a person, would it be a man or a
  woman? It would be a ___*, then log P(*man*) − log P(*woman*). Option order biases answers, so
  both orders are run (*a man or a woman* / *a woman or a man*) and averaged.
- Secondary frame, narrative: *In the story, the {gloss} ({noun}) came to life. Every morning,
  ___ woke up*, then log P(*she*) − log P(*he*). Closer to the human personification tasks
  (Sera et al. 1994), but *it*/*they* take more of the probability.
- **Framings (added 2026-10-08, PI):** the forced choice is asked three ways, *a man or a
  woman* (→ *man*/*woman*), *male or female* (*It would be ___*, no article → *male*/*female*),
  *a boy or a girl* (→ *boy*/*girl*), each in both orders. **The pre-registered R2 score is the
  average over the six versions**; each framing is reported separately as a consistency check
  (an effect in only one framing may be about those words, not gender). Watch-outs: *boy/girl*
  adds age; *male/female* is more biological.
- **R2-EN (added 2026-10-08, PI):** the same frames with the English noun only, no foreign word
  (*If the bridge were a person, would it be…*; *The bridge came to life. Every morning, ___
  woke up*), steered at the English noun, as R1-EN.
- Frame check: unsteered, the gendered options must take a clear share of the probability, for
  every framing. The pronoun split (*she*/*he*/*they*/*it*) is reported descriptively (base models have no
  trained-in preference for *they*, but singular *they* is common in web text).
- Caveat stated in advance: R2 can't tell social gender from grammatical agreement carried into
  English (a "pronoun echo"). R1 can, because adjective content isn't agreement. Bleed evidence
  rests on R1.

**R-NONCE (proposed 2026-10-08, PI): nonce words** (secondary)
- *Steered:* the R1-EN/R2-EN frames with a nonce word (*If the brelda were a person…*; *The
  brelda is very ___*), pushed along the grammatical-gender direction. A meaningless word has no
  concept for the push to interact with, so a shift comes from the direction alone.
- *Unsteered:* Phase 1's held-out nonce stems in *-a*/*-o* pairs (*brelda*/*brelpo*; never in
  any direction's training). Do *-a* forms personify as women / draw feminine-rated adjectives
  more than *-o* forms? Tests whether *spelling-carried* gender bleeds into social gender, with
  no real word involved (human analogue: invented names ending in *-a* are judged more often
  female).
- Nonce words are multi-token; steering is at the last token, as for real nouns.
- **Graded ending set (P15, proposed 2026-10-08, PI):** with only *-a*/*-o*, a difference can't
  say which ending moved. **New stems generated with Wuggy** (PI, 2026-10-08; Phase 1's stems
  stay frozen with Phase 1's result), four endings attached to each stem so every set is a
  minimal set, each with stated predictions, since no ending is truly neutral:
  *-a* (Spanish f; English names: vowel-final → female), *-o* (Spanish m), *-e* (Spanish mixed,
  leaning m: *el puente* / *la noche*), consonant-final (Spanish mixed; English names:
  consonant-final → male; Cassidy, Kelly & Sharoni 1999). This separates the Spanish
  grammatical cue from the English name-phonology cue. The consonant ending is chosen by rule
  before any model output (the ending whose Spanish nouns split closest to 50/50 in the
  lexicon); same checks as Phase 1 (absent from Wiktionary, zero frequency in six languages).
- **English-style nonce words** (*fleebleglorp*-type), if added: generated with **Wuggy**
  (Keuleers & Brysbaert 2010; the standard pseudoword generator, with English/Spanish/German
  modules), the documented-method answer for a write-up. (ARC Nonword Database, Rastle et al.
  2002, as an English alternative.)

**R3 (secondary): same-language, gender-invariant adjectives**
- Spanish adjectives that don't inflect for gender (*fuerte*, *elegante*, *frágil*, *suave*,
  *amable*…), in *Mi {noun} es muy ___*.
- Ratings via the English gloss to Glasgow, which is noisy; reported. This is the only readout
  applying Glasgow (UK English raters) across languages (PI concern 2026-10-08); check for native
  Spanish/German gender-association norms before finalising R3.

**Baseline test (§5.0)**
- On flipped pairs: the R1 score for the Spanish noun minus the German noun, regressed on the
  Spanish gender (feminine/masculine).
- A Boroditsky-type effect predicts that feminine-in-Spanish nouns get more feminine adjectives
  in the Spanish frame. Item-level bootstrap.
- Reported with and without *luna*/*estrella*/*mar*, which are also women's names.

## §5.3 Dose-response with damage

**Doses:** α ∈ {−2, −1, −½, −¼, 0, ¼, ½, 1, 2} × α\*, at the gate layer(s).

**Recorded at each dose**
- Target: the R1/R2/R3 scores.
- Damage: KL divergence of the next-token distribution, steered vs unsteered, over the 20 tokens
  after the noun in neutral sentences containing the noun. Plus the gate's agreement margin (it
  should move; nothing else should).

**Working window:** doses where the median damage is ≤ the random directions' median damage at
α\* (fixed in advance).

**Primary test**
- Slope of R1 on α within the window, per noun, averaged, with a bootstrap CI (nouns resampled).
- Compared with the same slope for 20 random directions of the same norm. Effect = the gender
  slope above the 95th percentile of random-direction slopes, with a CI above 0.

**Asymmetry (markedness)**
- Positive vs negative α, reported descriptively.
- Phase 3 didn't support markedness, so there's no directional prediction.

## §5.4 Specificity

- **Number direction:** singular vs plural, difference of means on the same nouns, the same
  norm and dose. If it shifts R1 as much as gender does, the effect is "any grammatical push".
- **Glasgow gradient:** each adjective's shift (steered at +α\* minus unsteered), regressed on
  its gender rating. Bleed predicts a positive slope.
  - Control: the same regression on its *size* rating, with a size-rated adjective set. Gender
    should predict, size shouldn't.
  - This replaces hand-picked off-target pairs with a rated dimension.
- **Spanish and German must agree in sign** for a "bleed" verdict. Either alone is "language-specific".

## §5.7 (new) Social-gender steering: positive control and comparison (added 2026-10-08, PI)

The social-gender direction of §5.1 (English person pairs: *he/she*, *man/woman*,
*king/queen*; difference of means at the same layer) is also used for **steering**, with the
same nouns, doses (in the same units), readouts and damage metric:
- **Positive control:** pushing *bridge* toward "woman" must move R1/R2. If it doesn't, the
  readouts are insensitive and a grammatical null is uninformative. It also shows what a real
  effect's size looks like.
- **Profile comparison:** each steering run gives a shift for every rated adjective.
  Correlation between the grammatical-direction profile and the social-direction profile:
  high = the grammatical push moves associations *the way* social gender does; low = it moves
  something else. Plus effect sizes at matched damage.
- Cheap: one more direction per run.

**Candidate follow-up ("Phase 6"), not part of Phase 5:** is *social* gender one direction
across English, Spanish and German? It has Phase 4's shape (reuse its methods), but its core
difficulty is that Spanish/German person words carry grammatical gender too (*la mujer*,
*die Frau*), so word pairs alone can't separate social from grammatical gender; epicenes
(*la persona*, *das Mädchen*) are the tool. Needs its own stimulus design.

## Verdict table (pre-registered once adopted)

| gate | R1 slope beyond random | number control | Glasgow gender slope | verdict |
|---|---|---|---|---|
| pass | yes | gender > number | > 0, size ≈ 0 | grammatical gender moves social-gender content (bleed) |
| pass | yes | gender ≈ number | any | generic grammatical perturbation |
| pass | no | — | — | clean null at this layer and dose |
| fail | — | — | — | v isn't causally on the agreement pathway; no Phase 5 claim |

## Compute (P7)

- **Mac, exploratory:** 1.7B/4B, a few hours, sequential GPU.
- **Next A100 round:** 8B/14B.
  - Phase 5: ~100 nouns × 2 languages × 9 doses × (1 + 20 random + 1 number) directions ×
    ~4 readouts × 1–2 layers ≈ 0.3–0.6M short forward passes per model, batched. Roughly
    30–60 min per model on an A100.
  - Plus Phase 4 extraction (flipped pairs), plus `suffix_ctrl` extraction (minutes).
  - Estimate: ~2.5–3.5 h ≈ $4–5.50, without 30B-A3B (its 25-min load doesn't justify it for
    steering).

## Same chip if the next round uses an H100 (2026-10-08)

The rule is "never compare activations across hardware", i.e. every comparison *within a phase*
happens on one chip. If an H100 is cheaper per run (published bf16 compute ~3×, memory bandwidth
1.7× an A100's; real speedups typically 1.5–3×; **benchmark one fixed batch at the start, switch
to an A100 if the speedup doesn't beat the price**), then in that session: re-extract the Phase
2/3 training activations on the H100 and refit the directions there (minutes per model), so all
of Phase 4 and Phase 5 is single-chip. Phases 2–3's confirmatory results stay on the A100.
Bonus check: cosine between the A100- and H100-fitted directions of each model (expected ≈ 1).

**Napkin cost (2026-10-08, before measuring):** ~500 steered items × ~600 conditions (real
directions × 9 doses × noun-only/every-position; 20 random across doses + 80 at the working dose;
1–2 layers) × all readouts: with P16, ~3–4 h (8B) + ~5–7 h (14B) + ~1 h extraction/setup on an
A100 ≈ 9–12 h ≈ $14–19. Firm estimate from the Mac exploratory run's measured throughput before
renting. Trims if needed: every-position steering at the working dose only (~−25%), one layer.

## Decisions for the PI

- **P1. Adopted (PI, 2026-10-08).** First-round scope as above (gate, baseline, dose-response, specificity; erasure,
  epicenes and the genderless control later)?
- **P2. Adopted (PI, 2026-10-08).** Difference-of-means steering vector, probe direction secondary?
- **P3. Adopted (PI, 2026-10-08).** Fresh held-out German nouns for steering (excluding all training and compound heads)?
- **P4. Adopted (PI, 2026-10-08).** German gate: dictionary frame primary, pronoun frame secondary?
- **P5. Adopted (PI, 2026-10-08).** R1 (English adjectives, Glasgow-rated) as the primary social-gender readout, with
  bleed resting on R1, not on the pronoun?
- **P6.** Glasgow Norms: **done 2026-10-08.** Downloaded the publisher's supplementary file
  (5,553 words, nine scales; `data/raw/glasgow/`, gitignored, SHA-256 30d7776e…). Licence:
  CC BY 4.0 (open-access article; the same data is redistributed under CC BY 4.0 by NoRaRe).
  Remaining decision: adopt the graded-regression revision of R1 above?
- **P9. Adopted (PI, 2026-10-08).** R1-EN and R1 as **co-primary** readouts with a Holm correction (recommended: they test
  different things, both central), or R1-EN primary and R1 secondary?
  Consequence (found 2026-10-08): a random-direction test's p-value can't go below
  1 / (number of random directions + 1); with 20 directions that's 0.048, which can never pass
  Holm's first threshold (0.025). So **≥ 40 random directions, proposed 100** (min p ≈ 0.01):
  all 100 at the working dose, the full dose sweep for the real direction and a subset of random
  ones (contains compute).
- **P10. Adopted (PI, 2026-10-08).** Social-gender steering (§5.7) as a required positive control: if it fails, Phase 5
  reports "readouts insensitive" instead of a null?
- **P11. Adopted (PI, 2026-10-08).** R2 as the average over three framings × two orders, framings reported separately?
- **P12. Adopted (PI, 2026-10-08).** Add R-NONCE (steered and unsteered nonce words, Phase 1 stems) as a secondary readout?
- **P13. Adopted (PI, 2026-10-08).** Steer all of the noun's tokens (not only the last)? Recommended.
- **P14. Adopted (PI, 2026-10-08), amended the same day (PI):** every-position steering as a
  *secondary test of vector-level overlap* (not descriptive only), with the etymology covariate,
  Spanish/German sign agreement and, if Phase 4 supports it, the pooled vector.
- **P17. Adopted (PI, 2026-10-08): R-NONCE build parameters + an English-style nonce set.**
  R-NONCE: Wuggy (`orthographic_spanish`) pseudowords from templates = common inanimate 2–3
  syllable Spanish nouns ending in *-a*/*-o* (equal numbers; templates only give shape, and none
  is in any test set); stem = pseudoword minus its final vowel (deduplicated; stems ending in
  *c*/*g*/*z*/*q* dropped, since a following *e*/*i* changes their pronunciation); four forms per
  stem: ***-a*** (Spanish strong f; English vowel-final names → female), ***-o*** (strong m),
  ***-e*** (Spanish *-e* nouns are only 16% feminine: a weak masculine cue, not neutral), ***-iz***
  (38% feminine, *la nariz* / *el lápiz*: the most even consonant ending that isn't an
  inflection; the rule's literal pick, *-s* at 45%, is also the plural marker in Spanish and
  English, so it was excluded on that ground; English consonant-final names → male, so the
  Spanish and English cues conflict here). All four forms absent from the Spanish Wiktionary and
  zero frequency in six languages; tokenisation logged; 150 stems = 600 words, test-only.
  **English-style set** (steered only; ~100 words): Wuggy `orthographic_english` from common
  English concrete nouns; zero frequency in six languages, not a WordNet lemma; final-letter
  class recorded (English name phonology), which cancels in steered-minus-unsteered shifts.
  German suffix nonces: candidate only, after the suffix follow-up's S1.
- **P16. Adopted (PI, 2026-10-08), conditional on the count below:** R1/R1-EN score only
  adjectives that are a **single token** in the model's vocabulary (one forward pass gives all
  their probabilities; full-word scoring of multi-token adjectives made R1 ~40× more expensive
  than everything else). Counted on the Glasgow adjectives (adjective-dominant, Zipf ≥ 3; 739):
  single-token in Qwen3 656 (89%; lost: *awesome*, *weird*, *cute*, *ridiculous*…), in EuroLLM
  471, in both 468. **Each model family uses its own set** (cross-family comparisons are
  conclusion-level); gender–valence correlation unchanged (≈ −0.45), so the covariate design
  stands.
- **P18. Adopted (PI, 2026-10-08): paraphrased wordings for R1 and R1-EN.**
  Scoring is deterministic (one forward pass gives exact probabilities; rerunning the same
  prompt gives the same numbers, up to GPU rounding), so repeated runs add nothing. The
  variation that matters is *which wording* was chosen: a result that holds under one wording
  only may be about that sentence. R2 already varies its wording (three framings × two orders);
  R1 and R1-EN, the co-primaries, have one wording each. Proposal: **three wordings each**, fixed
  now on grammatical grounds (every one keeps *very* before the slot, so the slot is an
  adjective; the noun never comes first; the noun is the only steered position, as before).
  - R1-EN: (W1) *The {noun} is very ___* (current); (W2) *The {noun} was very ___* (past tense:
    the original design's "The bridge was ___"); (W3) *I think the {noun} is very ___*
    (an opinion carrier).
  - R1 (Spanish or German word; "Spanish" ↔ "German"): (W1) *The Spanish word "{noun}" means
    {gloss}. Described in one word, it is very ___* (current); (W2) *In Spanish, "{gloss}" is
    "{noun}". Described in one word, it is very ___* (gloss and noun in the other order, no
    article to choose for mass nouns); (W3) *The Spanish word "{noun}" means {gloss}. I think it
    is very ___*.
  - Rejected from the averaged score: *looks very* (visual bias), *seems* and *so* (already
    rejected for R1-EN); wordings with the noun after the slot. *Everyone says…* is kept out of
    the average (a social frame would change what the primary score measures) but added as a
    contrast (below).
  - **Consensus contrast (PI, 2026-10-08), secondary:** (W4) R1-EN *Everyone says the {noun} is
    very ___*; R1 *The Spanish word "{noun}" means {gloss}. Everyone says it is very ___*. W3 and
    W4 are a minimal pair: a personal opinion vs what people say (consensus, i.e. the register
    stereotypes are reported in). Statistic: W4's gender coefficient minus W3's, with a noun
    bootstrap CI, two-sided (no direction pre-registered; a larger W4 effect would suggest the
    bleed travels through stereotype knowledge, a smaller one that consensus framing pulls
    toward generic praise/criticism). Also reported unsteered for R1 (do Spanish feminine vs
    masculine nouns draw more gendered adjectives under consensus framing?). Same frame check;
    outside the Holm family. Run at the working dose (real direction + all random directions,
    whose W4 − W3 differences are the null), not across the dose sweep.
  - **Score:** each adjective's steered-minus-unsteered shift is averaged over the wordings, then
    the same graded regression as now. The averaged score is the pre-registered statistic, so
    the Holm family stays two tests. Each wording's gender coefficient is reported separately;
    a result is **"wording-robust"** if every wording's coefficient has the same sign (as
    "position-robust").
  - **Frame check per wording** (unsteered, Qwen3 + EuroLLM, same threshold as W1). A wording
    that fails is dropped before any steering, on that ground alone, and the drop is reported;
    no substitute is chosen afterwards.
  - The no-*very* secondary frames stay W1-only; the random directions, dose sweep and
    every-position steering use the averaged score, so real and random are scored alike.
  - Cost: +2 forward passes per steered item and condition for R1 and for R1-EN, on top of the
    ~2 each now (with/without *very*) and R2/R2-EN's seven each: roughly **+20–35% steered
    compute**, i.e. the session-2 estimate of 9–12 A100-hours becomes **~11–16 h (~$17–25)**,
    to be firmed up by the Mac dry run. W4 at the working dose only adds a few percent. Trim if needed, decided before renting: W2/W3 at the working dose only
    (real + all 100 random directions), dose sweep on W1.
- **P19. Adopted (PI, 2026-10-08): a gender wug test on the R-NONCE words (exploratory).**
  Do models assign Spanish gender to brand-new words by their ending, from what size up, and in
  the order the lexicon predicts? Published LLM wug tests (Weissweiler et al. 2023; Anh et al.
  2024) cover inflection in large chat models, not gender assignment across one family's sizes.
  - Items: the 150 R-NONCE stems × *-a/-o/-e/-iz* (`phase5_nonce_es_v1.csv`), scored only, never
    trained on. **Stems beginning with *a-*/*ha-* dropped (5: *alboñ*, *aspann*, *haubl*,
    *albov*, *alboh*)**: feminine nouns with stressed initial *a* take *el* (*el aula*), so the
    article frames would be ambiguous there. 145 stems remain.
  - Method: Phase 0 exactly (`phase0.py`): *Esto tiene que ver con el/la X.* and *Aquí hay
    un/una X.*, margin = log P(word + rest | masculine) − log P(… | feminine), no thresholds
    retuned.
  - Prediction, fixed now from the Spanish lexicon (share of nouns feminine by ending): *-a*
    most feminine, then *-iz* (38%), *-e* (16%), *-o* least. Reported per model: mean margin and
    share preferring feminine per ending; Spearman correlation between the four endings' model
    order and the lexicon order; *-a* vs *-o* AUC over stems.
  - **"Passes the gender wug test"** (fixed before data, mirroring the Phase 0 frame check):
    in both frames, ≥ 70% of *-a* forms prefer feminine and ≥ 70% of *-o* forms prefer
    masculine. A model that fails gets its R-NONCE unsteered result flagged as uninterpretable
    (it doesn't treat the endings as gender cues at all).
  - Models: Qwen3 0.6B/1.7B/4B and EuroLLM-1.7B on the Mac (exploratory, fp16); 0.6B–14B in
    session 2 on one chip (minutes; scoring only). Tokenisation reported (whether the ending is
    its own token), since that may decide what small models can see.
  - Not tested: German (no German nonces yet; candidate after S1), English nonces (no grammatical
    gender).
- **P20. Adopted (PI, 2026-10-08): in-language baseline on flipped pairs (R3-PAIR), unsteered.**
  Ask each language directly, *El puente es muy ___* / *Die Brücke ist sehr ___*, and compare the
  same concept across the two languages. Extends R3 (Spanish only) to German and to the pairs.
  - Frames: Spanish *El/La {noun} es muy ___* (the noun's own article: the natural sentence, as
    a speaker would say it); German *Der/Die {Noun} ist sehr ___*. German predicate adjectives
    never inflect, so any adjective is grammatical; Spanish ones agree, so **Spanish uses only
    gender-invariant adjectives** (*fuerte*, *elegante*, *frágil*: same form for both genders),
    else the adjective's form would carry the noun's grammatical gender into the score.
  - Second Spanish wording (PI): *Es un/una {noun} muy ___* (attributive; no German counterpart,
    since a German attributive adjective comes before the noun). Scores averaged over the two
    Spanish wordings, each also reported; both must pass the frame check (as P18).
  - Adjectives: Glasgow-rated English adjectives (as R1) whose German and Spanish translations
    come from Wiktionary (kaikki dumps: a de/es adjective whose gloss is that English word;
    ambiguous mappings dropped, not reviewed). Each adjective keeps its English Glasgow rating
    in both languages, so the same rating is used on both sides of a pair. No native German or
    Spanish adjective gender norms were found (2026-10-08 search: role-noun and free-association
    norms exist, not adjective gender ratings); stated as a limit.
  - **Statistic (difference-in-differences):** per pair, the graded score (adjectives' log
    probability regressed on rating, covariates as R1) for the German noun minus the Spanish
    noun; then de-f/es-m pairs minus de-m/es-f pairs. The concept cancels within a pair
    (*bridge*-ness is on both sides); a language's overall lean (German frames drawing more
    masculine-rated adjectives, say) cancels between the two pair types. What's left is
    grammatical gender. Prediction: > 0 (the feminine side draws more feminine-rated
    adjectives). Item-level (pair) bootstrap; cognate sensitivity as in Phase 4.
  - Single-token adjectives not required (unsteered only: one pass per noun, cheap); Mac
    exploratory on 1.7B/4B + EuroLLM, confirmatory 8B/14B in session 2. Secondary, outside Holm.
  - Reading it: this is the in-language form of the human Boroditsky question. Boroditsky's
    participants answered *in English* to avoid agreement; R1 copies that design, R3-PAIR is the
    in-language counterpart. A positive R3-PAIR with a null R1 would say the association lives
    inside each language but doesn't reach English.
  - Later (not this round): steer these frames with the congruent vs the opposite gender
    direction.
- **P21. Adopted (PI, 2026-10-08): open-vocabulary record (exploratory).** Every scored frame
  (R1, R1-EN, R2, R3, R3-PAIR, R-NONCE; unsteered and steered at the working dose) also saves the
  top 50 next tokens with their log probabilities. Reported, labelled exploratory: (1) what the
  models actually put in the slot, per language and gender group; (2) the tokens steering moves
  most, over the whole vocabulary; (3) shortlist coverage: the share of slot probability the
  rated adjectives take, and the most probable unrated words (a check on the confirmatory
  design). No test is run on these; no choice is changed because of them.
- **P22. Adopted (PI, 2026-10-08): the model's own gender axis, validated against Glasgow
  (exploratory).** A measurement *of* the model, not an outcome scale for the bleed tests (those
  stay on human ratings: scoring a steered direction on the model's own axis would partly
  re-measure the overlap Phase 4/P14/§5.7 already test, and the axis also carries grammar).
  - **Axis, fixed now:** per model and language, the mean difference of the output (unembedding)
    vectors of suppletive woman/man pairs, the same concepts in every language: *woman/man*,
    *mother/father* (es *mujer/hombre*, *madre/padre*; de *Frau/Mann*, *Mutter/Vater*).
    Excluded on principle: Spanish/German pronouns (*ella/él*, *sie/er* agree with objects too:
    grammar; German *sie* is also "they"/formal "you"); *-a/-o* and *-in* pairs (*hija/hijo*,
    *Königin/König*: the ending would carry spelling/morphology into the axis). English is also
    reported with *she/he* added. Words are scored by cosine with the axis (first token, with a
    leading space; multi-token words flagged).
  - **Validation (the point of P22):** Spearman correlation between the axis score and the
    Glasgow gender rating over the rated adjectives, raw and partial on valence (r = −0.47: an
    axis could be a pleasantness axis). In English on the English adjectives; in Spanish and
    German on their Wiktionary translations (P20 mapping), still against the English rating.
  - **Cross-language:** rank correlation of the axis scores of translation-equivalent
    adjectives (es vs en, de vs en, es vs de), and cosine between the languages' axes.
  - **Reading it:** if the Spanish/German validation is close to the English one, the model's
    in-language sense of gendered adjectives agrees with the projected English ratings, which
    supports using them in R3/R3-PAIR; if much lower, R3's reliance on Glasgow is weakened and
    that is reported. Either way it describes this "speaker", not human Spanish or German
    speakers: it doesn't replace native norms.
  - **Coverage:** where validation is good, the axis also scores unrated tokens in the P21
    open-vocabulary record (descriptive only).
  - Cheap: no forward passes for the axis itself (output weights only); every model, Mac and
    pod. Outside Holm.
- **P15. Adopted (PI, 2026-10-08).** Graded nonce ending set (*-a*/*-o*/*-e*/consonant on the same stems), **all new nonce
  sets generated with Wuggy** (`wuggy` 1.1.2, MIT; confirm the Spanish module before building),
  plus the Phase 1 checks (absent from Wiktionary, zero frequency in six languages)?
- **P7. Adopted (PI, 2026-10-08):** 8B/14B only on the next round (30B-A3B skipped).
- **P8. Adopted (PI, 2026-10-08):** EuroLLM-1.7B as a cross-family check of the gate and baseline on the Mac.

- **P23. Adopted (PI, 2026-10-09): English base case (exploratory).** Before any steering, do
  English nouns already lean on the gender directions?
  - Items: bare English nouns, same input format as Phases 2–3: (a) the English concepts of
    the steering nouns and flipped pairs; (b) a calibration set of Glasgow-rated nouns
    (WordNet senses ≥ 50% noun, Zipf ≥ 3, inanimate by the Phase 3 WordNet check).
  - Directions: the social-gender direction (§5.7, definition in P24), ES, DE and POOLED
    (Phase 4), at every layer.
  - (1) **Social direction on things:** Spearman correlation of each calibration noun's
    projection with its Glasgow gender rating, raw and partial on valence (as P22).
  - (2) **Translation gender in English:** on the flipped-pair concepts, AUC of the ES
    projection for "the Spanish translation is feminine" and of the DE projection for "the
    German translation is feminine"; POOLED for both. Expected: 0.5 (noise). Because the pairs
    are flipped, the two predictions have opposite signs, so a meaning property shared by the
    concepts can't produce both. Bootstrap over concepts.
  - Reading: a lean following translation gender would be a baseline, intervention-free
    "bleed" into English, to know before reading R1-EN.
- **P24. Adopted (PI, 2026-10-09): cross-domain transfer and the §5.1 cosine (exploratory).**
  Is one gender direction serving both grammatical and social gender?
  - **Social direction (fixed now):** difference of means, female minus male, over English
    person pairs extracted bare: *woman/man, girl/boy, mother/father, daughter/son,
    sister/brother, queen/king, wife/husband, aunt/uncle, she/he*. Each pair's sex labels are
    checked automatically against WordNet (first-sense gloss); a pair that fails is dropped.
  - **Social → grammatical:** the social direction scores the Phase 2 `strat` and Phase 3
    `strat3` m/f nouns; AUC within each language's cells (as T1).
  - **Grammatical → social:** ES, DE and POOLED score held-out English sex-specific person
    nouns (not in the social-direction list). Labels come from WordNet: the noun's first-sense
    gloss begins "a woman/female…" or "a man/male…" (e.g. *actress* "a female actor"). AUC.
  - **§5.1 cosine:** split-half cosine between the social direction and ES/DE/POOLED, with the
    within-cell shuffle null and random directions as calibration (activations aren't
    centred).
  - Pooling grammatical and social into one vector: not done; its distance from each mostly
    restates their cosine.
  - Data: new extraction of English nouns and person words (minutes; Mac 1.7B/4B, session 2
    for 8B/14B). Outside Holm.

## Limits to state

- Steering is an intervention on a linear direction at one position. A null doesn't rule out a
  nonlinear or distributed route, and a positive result shows the direction is *sufficient* to
  move associations, not that the model uses it that way unsteered.
- English readouts after a Spanish/German noun mix in translation behaviour.
- Glasgow ratings are English-speaking, UK raters.
