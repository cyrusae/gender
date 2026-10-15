# What we've done, in order (an outline)

*A chronological outline of the project so far, for orienting yourself or as the skeleton of a
blog post. Dates are 2026. Details live in `docs/decisions.md` (every decision, dated),
`docs/explainers/` (methods, per phase), `docs/screening/` and `docs/reports/` (word lists and
review sheets) and `docs/methods-summary.md` (one page). "PI" = the researcher; "side agent" = a
second Claude session that read along and flagged problems.*

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
  the laptop know ~90% of the test nouns. EuroLLM-1.7B as a second model family for checks.

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

## 7. The confirmatory cloud run (Oct 8–9)

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
  the Spanish result also holds at the next token. The laptop analysis briefly filled the swap
  with eight parallel jobs (memory, not cores, is the limit; a memory-guarded job queue followed).

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
  differs between models. Confirmation waits for the next cloud run.

## 9. More design, a tangent, and speed (Oct 8–9)

- Phase 5 additions: three wordings per main readout plus an "Everyone says" vs "I think"
  contrast (P18); an in-language *El puente es muy ___* / *Die Brücke ist sehr ___* comparison
  on the flipped pairs (P20); a record of what the models actually say, beyond the shortlist
  (P21); the model's own woman/man axis checked against human ratings (P22).
- **Tangent: the gender wug test (P19).** Linguists test children with invented words ("this is
  a *wug*; now there are two ___"). We gave the models invented Spanish nouns and asked which
  article fits. Every model, even 0.6B, gives invented *-a* words feminine and *-o* words
  masculine articles (~80% vs ~5%), with *-e* and *-iz* in between, as real Spanish nouns are. No
  model was too small to pass. It reassures us that the invented words used later in Phase 5
  carry a gender cue the models actually use.
- Closest prior work found: Flint & Ivanova (2024), same question in older embedding models.
- Efficiency pass before the next cloud run: a steering engine 8–30× faster than the plain
  method, verified identical; faster scoring and extraction; the next model downloads while the
  current one runs.

## 10. Laptop dry run of Phases 4–5 (Oct 9, development numbers, Mac fp16)

- Phase 4 on 1.7B/4B:
  - Flipped pairs (*la luna* / *der Mond*): a direction trained on one language scores each pair
    by its own language's gender in every layer (4B: mean AUC 0.93 with the Spanish direction,
    0.88 with the German one; same without English cognates). At AFTER, much weaker.
  - Strict transfer (Spanish direction sorting German nouns within matched endings, and the
    reverse): above chance in only a minority of layers (mean AUC ~0.70–0.74). Fails the
    majority rule.
  - Russian *-ь* nouns (one shared ending, so spelling can't help): both directions sort them by
    gender in almost every 4B layer.
  - Geometry: split-half cosine ~0.44 (4B), at first with no layer above the shuffle null (that
    test turned out to be broken; see §12).
- Phase 5 gate on 4B (does pushing a noun along the direction flip agreement?): Spanish flips at
  a small dose at all four depths, random directions flip nothing; German passes at 25%, 40% and
  55% depth and fails at 70%.
- **Steering plan (PI):** steer at every layer that passes the gate ("more data"), push the same
  absolute amount everywhere, decide by the majority of passing layers and report every layer.
  Random-direction dose curves on a seeded 25% of nouns to keep the cost down.

## 11. A costly pause, then a measured budget (Oct 9)

- Benchmarking chips for the second cloud session: a permission prompt blocked a step while two
  rented GPUs sat idle for ~55 minutes (~$4 of a $7 balance). New rule: shut idle machines down
  *before* stopping to ask.
- Measured throughput on an L40S GPU: the full second session (every passing layer, both
  languages, 8B and 14B) comes to **~$14**; faster chips cost more per hour than they save.

## 12. Overnight build, confirmatory leftovers, a test that couldn't fail (Oct 9–10)

- Phase 1 confirmatory (cloud data, all six models): outcome (b) everywhere; the spelling eraser
  leaves invented-word endings readable in every layer of every size.
- Known-only sensitivity (8B/14B, only nouns each model knows): the Spanish result holds; German
  compounds still follow the head; markedness still 0.
- English base case (P23/P24, laptop 1.7B/4B): the woman/man direction lines up with human gender
  ratings of things (rho ~0.2, also beyond pleasantness); the Spanish/German directions sort
  English sex-specific person words above chance. A bug in my first cosine (averaging ratios that
  explode near zero) was caught and fixed.
- P22: the model's own woman/man axis in its vocabulary agrees weakly with human ratings in
  English (rho 0.10–0.14), barely in Spanish/German.
- Phase 5 code complete (vectors, readouts, damage, analysis, session-2 driver).
- **PI decisions (Oct 10):** verdict by majority of layers; frame checks enforced (no steering for a
  language whose test sentence fails; failing wordings excluded); a damage-matched comparison
  with random pushes; the Spanish adjective list reviewed against the RAE dictionary (unfamiliar
  words and English loanwords dropped).
- **A test that couldn't fail.** The shuffle test for "do two directions overlap?" could never
  reject: under shuffled labels a direction has almost no length, so the null cosine blew up
  (above 1, impossible for a cosine). Switched (PI) to the split-half *cross term*, which needs no
  division. Reruns (LAST): Spanish × German overlap beyond chance in 22/27 (1.7B) and 25/35 (4B)
  layers, none at AFTER; social × Spanish in 23/27 and 29/35; social × German 6/27 and 24/35.
  Cosines ≈ 0.3 (Spanish), 0.12–0.22 (German), vs ≈ 0.01–0.03 for the plural−singular direction.
- **Speed review** (from a separate conversation the PI had): fewer GPU↔CPU syncs, damage scored in
  batches (checked identical to 1e-5), one model load for all Phase 5 steps, GPU utilization
  logged on the pod.
- **End-to-end test of the cloud script** on 0.6B in a small mode: caught a missing Russian
  word list in the bundle and a crash saving one summary; both fixed; the script can now resume at
  a named step. Full pass on Oct 10.

## 13. Curating the English person words (Oct 10–12)

- **Lists keep getting better, and smaller.** The person-word test (P24) asks whether the Spanish
  or German direction puts *bride*, *nun*, *lady* on one side and *groom*, *monk*, *prince* on the
  other. Steps:
  - v2: words whose spelling gives sex away (*waitress*, *salesman*, *ballerina*) moved out of the
    main set; GN-GloVe's gendered word lists (Zhao et al. 2018) added as a second source.
  - v3: the PI dropped *whore*, *jackass*, *prostitute* (WordNet has no register label for them).
  - v4: *miss*, *stud*, *groom* dropped because their verb uses dominate (the PI spotted *miss*;
    the rule existed but I had not applied it to the new source); *fiancé/fiancée* dropped (PI:
    swapped and merged in real use). Main set: 12 women's words, 10 men's.
- **First names (P24b):** 1,738 US first names (≥ 95% one sex), compared only with names ending in
  the same letter so *-a* can't do the work. Unisex names (*Jordan*, *Taylor*) are excluded by
  rule, not tagged.
- **Pairs (P24c):** female/male equivalents (*widow/widower*, *nun/monk*) from Wiktionary, WordNet
  and Bolukbasi et al. (2016), checked automatically and reviewed by the PI: 16 woman-vs-man pairs
  and 6 woman-vs-generic ones (*actress/actor*, *heroine/hero*: the base word is now used for
  anyone). The first review sheet had bugs (a dialect *da* "dad" matched to "district attorney";
  plurals that defeated the sex check); rebuilt by code.
- Results so far (laptop, Spanish direction): main set AUC ~0.82 (1.7B) / 0.76 (4B), names 0.78 /
  0.72; German weaker. Three small, complementary tests that agree, rather than one big one.

## 14. Tangents that became checks (Oct 12)

- **La muerte (PI).** Is the Spanish direction carried by nouns the culture personifies (Death as
  a woman, as in *La Catrina*)? Segel & Boroditsky (2011) showed artists personify by
  grammatical gender. Personified concepts taken from Iconclass (an art-subject catalogue,
  including Ripa's *Iconologia*); dropping the 7 Spanish training nouns it flags changes the
  direction no more than dropping 7 random nouns. *Muerte* isn't special.
- **Cross-language steering (P26, PI).** Push the *German* direction into a Spanish sentence
  (*Mi puente es muy ___*): does it move adjectives as the Spanish one does? Added to the cloud
  run for ~$1. (The social-gender push on the same sentence was already in.)
- **Grammar and sex pulled apart (P27).** A multilingual "social" direction was rejected (in
  Spanish/German, words for women are also grammatically feminine, so the two can't be
  separated). Instead: German diminutives are always neuter (*das Schwesterchen*, *das
  Brüderchen*). Does the German direction still sort them by sex? Side-agent flags shaped the
  design: the stem (*Schwester*) is itself feminine, so the effect is compared with neuter
  object diminutives of feminine vs masculine stems; the cleanest cases (*Dienstmädchen*, built
  on neuter *Mädchen*) are compared with sex-neutral person words (*Kindchen*, *Frühchen*) to
  rule out "person vs thing"; uncertainty is computed by word family. Suggestive only (one clean
  family, no clean male words).

## 15. The second cloud session: 8B and 14B (Oct 12–13)

- **The run.** One L40S, about 10 hours of work: Phases 4 and 5 (confirmatory), the suffix
  follow-up, the English and German side tests, and wug tests for the small models. German's
  failed gate cut the planned runs, so 14B finished early.
- **German: no valid test.** The pre-registered German steering frame failed on both sizes: on
  8B the model misread feminine nouns in it unsteered; on 14B the push didn't swap genders, it
  pushed both toward neutral. A backup frame passed, but switching to it after seeing the first
  one fail would be choosing on results, so the PI decided not to. Every Phase 5 verdict is
  Spanish-only.
- **The headline: steering works on 8B, not on 14B.** On 8B, pushing *el puente* toward feminine
  moves English adjective choices toward feminine-rated words, about a third as much as a
  woman–man push, and a plural–singular push doesn't. On 14B the same push still flips the
  article but moves the adjectives not at all, while the woman–man push still does. No growth
  with size; if anything, the link fades.
- **Phase 4 holds at both sizes.** Pairs follow each language's gender everywhere, Russian *-ь*
  nouns are sorted by both directions, the overlap beats chance, and the strict within-ending
  transfer test fails (on 14B one direction passes, German → Spanish).
- **English person words: still sorted, but less overlap.** The Spanish direction sorts English
  person words and names by sex on both sizes, but on 14B its overlap with the woman–man
  direction halves. That fits the steering null, but it's an observation on two sizes, not a test.
- **German neuter diminutives:** the null replicates; the grammatical directions don't read the
  sex of *das Schwesterchen*, the social direction does.
- **A bug found on real data:** 8 Spanish nouns sat in two word lists, so their sentences were
  scored twice; the analysis crashed until each copy was compared with its own baseline.
- **An expensive slip.** The script that should have told me the run had ended matched its own
  process, so the machine sat idle for 16.5 hours after finishing: $28.67 instead of about
  $11.50. Fixed by waiting for the log's "done" line and checking in on a timer.

## Where things stand (Oct 15)

- **Done:** Phases 0–4 confirmed on the target sizes; Phase 5 confirmatory for Spanish (8B:
  language-specific effect; 14B: none); German Phase 5 has no valid test; side tests replicated
  on 8B/14B.
- **Next:** analysis code for the remaining pooled readouts (R3-POOL, P20); then write-up.
  Candidates: steered text generation (what the model *says* under a push), a second model
  family (EuroLLM) to see whether 8B or 14B is the odd one out.

## Threads a blog post could pull on

- How many ways a "gender direction" can secretly be a spelling detector, and how each was ruled
  out.
- A noise bias that would have handed us a false confirmation; a shuffle test that could never
  fail.
- Mistakes caught and logged: the circular eraser test, the attention-sink bug, the pronoun
  confound in the verb frame, the training leak, the bootstrap duplicates, applying a filter
  without sign-off, a rule I forgot to apply to a new source, idle cloud machines (twice; the second time a watcher that matched itself).
- What curating a word list really involves: slurs, verbs hiding in nouns, words people misuse,
  and dictionaries written from one sex's point of view (*boyfriend* is "the lover of a girl").
- Tangents worth following: wug tests, Santa Muerte, and German's neuter girls.
- A pre-registered effect that shows up at 8B and is gone at 14B: what a null at the larger size
  does and doesn't mean.
- Doing serious interpretability research on a laptop plus a few dollars of cloud time.
