# Phase 4 design: is grammatical gender shared across languages? (adopted 2026-10-08, PI)

*Status: adopted 2026-10-08; PI decisions (Q1–Q6, all adopted 2026-10-08). It builds on the design doc's Phase 4
(Spanish-only, German-only and pooled vectors; flipped pairs) and the additions in
`phases-3-5-plan.md`. Nothing here has been run. Exploratory on the Mac (1.7B/4B); confirmatory
8B/14B with the Phase 5 A100 round.*

## The question

Does the model use **one** grammatical-gender direction for German and Spanish, or one per
language? The **null** (design doc): which language assigns a noun which gender predicts nothing
about the model's shared representation of the noun.

The sharpest evidence is the **flipped pairs**: the same concept, opposite genders (*die Brücke*
/ *el puente*, *der Mond* / *la luna*).
- A shared **grammatical** direction follows each language's gender: *Brücke* scores feminine
  and *puente* masculine.
- A **meaning** direction scores both the same, because it's the same bridge.

## Directions

Every direction is fitted with the Phase 2/3 stratified estimators, so spelling and cell effects
are removed within each language.

| direction | trained on | note |
|---|---|---|
| ES | Phase 2 `strat` (m/f) | as pre-registered for Phase 2 |
| DE | Phase 3 `strat3`, masculine vs feminine (neuter left out) | as Phase 3's single axis |
| POOLED | ES + DE nouns, **centred within language** (language × cell fixed effects) | centring stops the direction from using "is this German?" |
| RU (Q1) | Russian *-ь* cell, m/f (see below) | the third-language test |

Each direction is tested both as a probe direction and as a stratified difference of means
(Phase 5 steers with the latter).

## Tests

**T1 Cross-language transfer (primary).**
- Train ES, test on held-out German m/f nouns; train DE, test on held-out Spanish nouns.
- **AUC computed within the test language's ending cells**, then averaged. This matters: German
  *-e* nouns are mostly feminine and Spanish *-e* nouns mostly masculine, so an unstratified
  transfer test would mix in spelling effects in either direction.
- "Transfers" if the lower bound is > 0.5 in a majority of inner layers.
- Test nouns (**revised 2026-10-09, PI**): the other language's training set (ES scores German
  `strat3` m/f nouns, DE scores Spanish `strat` nouns), since no fresh draw exists (every known
  mixed-cell noun is already in training); held out for the direction under test.
- Decision rules (2026-10-09, PI): Holm via interval width (97.5% first, 95% for the second
  test if one passes); T3 = ES-on-both and DE-on-both, paired AUC.

**T2 Geometry (descriptive).**
- Split-half cosine between ES and DE (noise-corrected, as in Phase 3), with the within-cell
  shuffle null.
- Anchors: 0 = unrelated directions, 1 = identical.

**T3 Flipped pairs (the null test).**
- For every pair, score both nouns on a direction and take score(German noun) − score(Spanish
  noun). A grammatical direction predicts the sign follows (German gender − Spanish gender).
- Statistic: the share of pairs in the predicted direction (paired, sign-test style), and the
  paired AUC, with a bootstrap over pairs.
- **Strict version:** ES scores the *German* nouns of the pairs. If it sorts them by German
  gender, against their Spanish translations' gender, the direction carries gender across
  languages and not meaning.
- Run with ES, DE and POOLED, at LAST (primary) and AFTER (secondary).
- **Only one-direction-scores-both versions test sharing** (found 2026-10-08): ES scoring the
  Spanish noun and DE scoring the German noun will follow each language's gender even if nothing
  is shared (that replicates Phases 2–3). The confirmatory T3 is: ES on both nouns of each pair,
  DE on both, POOLED on both. The same-language versions are reported descriptively.

**T4 Third language (if Q1 = Russian).** ES, DE and POOLED score Russian *-ь* nouns (m vs f,
spelling-matched by construction); AUC within sub-cells. Above chance means the direction
generalises to a language and script it never saw.

**Readings (design doc table, sharpened):**

| T1 transfer | T3 flipped pairs | reading |
|---|---|---|
| both directions transfer | follow each language's gender | a language-independent grammatical gender (shared) |
| each wins only at home | follow gender only with the matching language's direction | gender is encoded per language (a finding, not a failure) |
| transfer, but pairs score alike | — | the shared part is meaning, not grammatical gender |
| nothing transfers, pooled mediocre | — | directions differ enough that averaging blurs them |

This also decides how Phase 5's English-only readout (R1-EN) can be read: a null there means
"no bleed" only if T1/T3 show the direction is shared.

## Stimuli

- **Flipped pairs:** 519 in `pairs_de_es.csv`. Changes before use:
  1. **The PI's English review** (`docs/reports/flipped-pairs-review.md`): drop flagged pairs.
  2. **Training leak found 2026-10-08:** 37 pairs have a Spanish noun used in Phase 2
     *training* (5 `strat`, 5 `matched`, 27 regular-train). Phase 3 filtered flipped-pair nouns
     out; Phase 2 didn't. **These 37 pairs are dropped from Phase 4–5 tests** (PI, Q3), leaving
     482 before review and known checks. (Phase 2's own results are unaffected; the pairs aren't
     part of them.) The classic ***llave*** (*la llave* / *der Schlüssel*) is also in Phase 2
     training (equal-count set) and is dropped too. **How it happened:** Phase 2's lists were
     built 2026-10-06; the rule "training code must assert no test item appears" had no code
     behind it, and the held-out filter (`held_out_phase45`) was written on 10-07 for Phase 3
     and never added back to Phase 2; v4 carried v3's sets over. **Fix:** the Phase 4–5 test
     builders exclude (and assert against) every noun used in any Phase 2/3 training set, instead
     of relying on each phase's own filter. Phase 2's frozen lists are unchanged.
  3. **English overlap (PI, 2026-10-08):** pairs whose words *are* the English word (4) or are
     also unrelated English words (16) are dropped; the 63 **English-cognate** pairs are kept
     in the primary analysis with a **pre-registered sensitivity analysis without them** (and
     the with/without difference reported). Sex-associated pairs (6) dropped. Result: **389
     pairs** (239 German-f/Spanish-m, 150 German-m/Spanish-f) before known checks; every drop
     and reason in `docs/reports/flipped-pairs-flags.csv`.
  3. Known by every model compared (shared set), both nouns.
  4. Only 4 flipped pairs are cognates, so the planned cognate vs non-cognate comparison is
     dropped as underpowered.
- **Held-out German nouns for T1:** a fresh draw from `phase3_pool_v4` simplex nouns not in any
  training or test set, known by the models, in cells with both m and f.
- **Held-out Spanish nouns for T1:** Phase 2 regular *test* nouns plus fresh `stratum` nouns.
- **Russian (Q1):** the *-ь* cell after filters (loan status, month names and other one-gender
  semantic classes, suffix cells *-ость*/sibilant/*-тель*, two animacy checks: grammatical
  animacy from Wiktionary and our English-gloss check). About 136 f / 86 m before known checks
  (Zipf ≥ 3). Needs a Russian known check: Russian has no articles, so the frames must use
  agreement (e.g. a gender-marked adjective or past-tense verb). To be designed and checked on
  Qwen3 + EuroLLM.

## Extraction and compute

- **New extraction:** flipped-pair nouns, both languages, bare (~1,000 words); held-out German/
  Spanish draws; Russian set. Same input format and positions as Phases 2–3.
- **Mac (exploratory, 1.7B/4B):** minutes of GPU. Analysis is cheap (no LEACE, few bootstraps
  per direction), helped by the faster solver (`batched-solver` branch).
- **Pod (8B/14B):** rides along with the Phase 5 A100 round; a few minutes per model.

## Decisions for the PI

- **Q1. Adopted (PI, 2026-10-08).** Russian as the third language (T4)? It needs its own known-check frames, since it has
  no articles.
- **Q2. Adopted (PI, 2026-10-08): option A,** (co-primary with Holm, T3 = the one-direction-scores-both version.) T1 (cross-language transfer, within-cell AUC) as the primary test, T3 (flipped pairs)
  co-primary, with a Holm correction? Or T3 alone as primary, since it tests the design doc's
  null directly?
- **Q3. Adopted (PI, 2026-10-08): drop.** Drop the 37 leaked pairs (recommended) or refit the Spanish direction without them?
  Dropping is simpler and keeps Phase 2's pre-registered direction unchanged.
- **Q4. Adopted (PI, 2026-10-08).** Drop the cognate comparison (only 4 cognates)?
- **Q5. Adopted (PI, 2026-10-08):** (m/f only for tests, plus a pre-registered *descriptive* placement of
  German neuter nouns on the ES and POOLED directions, predicted nearer masculine because German
  neuter shares most inflection with masculine: *dem*, *des*, genitive *-s*.) German neuter: leave it out of Phase 4 (m/f only, as in the design doc)? The 519
  flipped pairs are all masculine/feminine by construction (327 German-f/Spanish-m, 192
  German-m/Spanish-f), so this only affects the directions and T1 test nouns.
- **Q6. Adopted (PI, 2026-10-08).** Positions: LAST primary, AFTER secondary, as for Phases 2–3?

## Limits to state

- "Shared" means shared *linear* directions at the noun's last token. Gender could be shared
  non-linearly or in later contextual positions.
- Flipped pairs differ in more than gender: word form, frequency, tokenisation. The paired
  design controls the concept but not these.
- Russian adds a script change, so a Russian null could be about the script, not the language.
