# What we've done, in order (an outline)

*A chronological outline of the project so far, for orienting yourself or as the skeleton of a
blog post. Dates are 2026. Details live in `docs/decisions.md` (every decision, dated),
`docs/explainers/` (methods, per phase), `docs/screening/` (word lists) and
`docs/methods-summary.md` (one page).*

## 0. The question (Oct 5)

- In German *die Brücke* ("bridge") is feminine; in Spanish *el puente* is masculine. For objects
  this is pure grammar.
- The famous (and contested) human claim: grammar shapes thought, so German speakers call bridges
  "elegant" and Spanish speakers "strong" (Boroditsky). Replications largely failed.
- **Our version:** inside a language model, does the representation of *grammatical* gender
  (learned from objects) overlap with *social* gender (men/women)?
- Design: five gated phases, each producing a usable result even if the next never runs.

## 1. Ground rules before any data (Oct 5–6)

- Gold labels come from a dictionary (Wiktionary), never from a language model.
- Drop anything that would need a manual check; small clean sets beat big messy ones.
- Pre-register every test (write down what counts as a result) before running it; log every
  decision with its reason.
- Held-out test sets are frozen and never trained on.
- Base models only, one device and precision per model, never compare activations across hardware.

## 2. Phase 0: does the model even know the genders? (Oct 5–6)

- Score sentences with each article (*Aquí hay **un**/**una** mesa*) and see which the model
  prefers; "known" = clear preference in two different frames.
- Lessons: score the noun *given* the article, not the article itself; avoid frames where an
  article has another meaning (*la* = "her"); Qwen3 adds no start token, so never put the tested
  word first.
- Model choice: **Qwen3 base models** (Gemma breaks in the Mac's half precision); 0.6B/1.7B/4B on
  the laptop know ~90% of the test nouns.

## 3. Phase 1: can we erase spelling? (Oct 6)

- Problem: Spanish *-a* is usually feminine, *-o* masculine, so a "gender direction" might just
  detect spelling.
- Built an eraser (LEACE) on verbs, where *-o/-a* marks person, not gender (*hablo/habla*), and
  tested it on invented words (*flitra/flitro*).
- Result: the eraser doesn't generalise; new words' endings stay readable. (Also caught a circular
  test of our own and fixed it.)
- Consequence: control spelling by *design* instead of erasure.

## 4. Phase 2: a Spanish gender direction (Oct 6–7)

- Train on nouns whose spelling doesn't give gender away; test where spelling and gender disagree:
  *el día* (masculine despite *-a*), *la mano*, Greek *-ma* nouns, clippings (*la foto*).
- Noun/verb homographs (*camino* "path" / "I walk") in a noun frame vs a verb frame. First verb
  frame used *yo*/*usted*, which gave away the person and thus the gender; replaced with
  *siempre ___* and checked on two model families.
- Bug caught: single-token words sat on an "attention sink" position with one giant number, which
  dominated everything; fixed by a neutral prefix.
- Readout position: the word's own last token (LAST) vs the next token (AFTER); LAST primary, and a
  result counts as robust only if it holds at both.
- Power problem → **stratified design**: compare genders only *within* spelling cells (same
  ending), using every noun.

## 5. Phase 3: German, three genders (Oct 7)

- Building clean German lists was the hardest part: compounds take their last part's gender,
  suffixes predict gender, neuter is more often borrowed, place names are all neuter, agent nouns
  (*Bohrer* "drill/driller") are masculine by rule, and the animacy tags missed quail, sable and
  "army". The PI's review caught four of these; each fix became a rule.
- Tests: do **compounds** follow their head's gender (*das Herrenhaus* from *das Haus*, even
  though *der Herr*)? Is masculine the "unmarked" default (a shorter vector)?

## 6. First results, and an estimator trap (Oct 7–8, exploratory, 1.7B/4B)

- Spanish: the stratified direction reads gender, not spelling, in most layers (at LAST only).
- German: compounds follow their head in every layer.
- **Markedness: the naive length comparison confirmed it in almost every layer, and it was
  noise bias** (smaller groups' vectors look longer). A split-half estimator removed the bias and
  the effect vanished. The cleanest methods result of the project.
- A sparse-autoencoder feature exposed a filter artifact: dropping *el agua*-type nouns had left
  word-initial *a-* as a perfect masculine cue; fixed by an amendment before the big run.

## 7. The confirmatory cloud run (Oct 8)

- Rented one A100 GPU; all six models (0.6B–14B plus a 30B mixture-of-experts) extracted in ~1.1
  hours for **$1.80**. Hiccups: the 30B model took 25 min to load; a disk filled up.
- The precision check narrowly missed its pre-registered bar (98.6% vs 99%): noise at a hard
  threshold; reported as a miss.
- The real bottleneck was the laptop analysis (~60 h estimated): sped up several-fold, finding two
  bugs on the way (a crash on a degenerate layer; a bootstrap that let copies of a word share
  noise).
- **Confirmed at 8B and 14B (LAST):** Spanish reads gender in 28/35 and 26/39 layers; German
  compounds follow their head in 35/35 and 39/39; markedness not supported. A spelling-erased
  direction reads *more* gender on homographs used as nouns than as verbs. At the next token
  (AFTER) the Spanish result vanishes: the gender lives on the word itself.

- **All sizes analysed (Oct 9):** every dense model from 0.6B to 14B shows the same picture at
  LAST, and none at AFTER. The 30B mixture-of-experts model (exploratory) is the only one where
  the Spanish result also holds at the next token. The laptop analysis ran overnight; it briefly
  filled the swap with eight parallel jobs (memory, not cores, is the limit).

## 8. Designing what comes next (Oct 8)

- **Phase 4 (shared across languages?):** train on Spanish, test on German; the decisive test is
  flipped pairs (*Brücke*/*puente*), scored by one language's direction. Russian added as a third
  language (only its *-ь* nouns are spelling-neutral; small models default those to feminine).
- **Phase 5 (does it bleed into social gender?):** push a noun along the gender direction and
  measure English adjectives rated for gender by humans (*"The bridge is very ___"*), regressed
  on the rating with pleasantness controlled (gender and pleasantness correlate at −0.47); plus
  personification ("would it be a man or a woman?"), nonsense words, random and social-gender
  control pushes.
- Word-list audits: the flipped pairs went 519 → 389 (sense mismatches caught in review; a
  training-set leak; English-identical words; people and groups), with one over-eager drop
  reversed after a devil's-advocate review.
- **Suffix puzzle:** feminine-suffix nouns (*-ung*, *-heit*) sit only a third of the way toward
  feminine. First look: partly a shared "derived noun" offset; whether gender is ranked inside it
  differs between models (one ranks, the other reverses in middle layers). Confirmation waits for
  the next cloud run.

## 9. More design, and speed (Oct 8–9)

- Phase 5 additions: three wordings per main readout plus an "Everyone says" vs "I think"
  contrast (P18); a gender wug test on the invented words across model sizes (P19); an
  in-language *El puente es muy ___* / *Die Brücke ist sehr ___* comparison on the flipped pairs
  (P20); a record of what the models actually say, beyond the shortlist (P21); the model's own
  woman/man axis checked against human ratings (P22).
- The gender wug test: every model, even 0.6B, gives invented *-a* words feminine and *-o* words
  masculine articles (~80% vs ~5%), with *-e* and *-iz* in between, as Spanish nouns are. No
  model was too small to pass.
- Closest prior work found: Flint & Ivanova (2024), same question in older embedding models.
- Efficiency pass before the next cloud run: a steering engine 8–30× faster than the plain
  method, verified identical; faster scoring and extraction; the next model downloads while the
  current one runs.

## Where things stand

- Done: Phases 0–3 confirmed on the target models; Phases 4–5 and the suffix follow-up fully
  designed and pre-registered; stimuli built (pairs, Russian list, nonce words, derived nouns).
- Next: Phase 4–5 code and a laptop dry run (which also measures how big the next cloud run
  must be), then the second cloud session.

## Threads a blog post could pull on

- How many ways a "gender direction" can secretly be a spelling detector, and how each was ruled
  out.
- A noise bias that would have handed us a false confirmation.
- Mistakes caught and logged: the circular eraser test, the attention-sink bug, the pronoun
  confound in the verb frame, the training leak, the bootstrap duplicates, applying a filter
  without sign-off.
- Doing serious interpretability research on a laptop plus $1.80 of cloud time.
