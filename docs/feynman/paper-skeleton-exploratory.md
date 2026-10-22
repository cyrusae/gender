# Exploratory paper skeleton: "Grammatical Gender Bleedthrough in LLMs"

**Prepared:** 2026-10-21 · **Status:** skeleton only — every claim below maps to an existing
result in the repo (explainer + results path given); nothing new needs to be run except the
confidence controls (spec: `outputs/confidence-direction-control-spec.md`). Sections marked
**(open)** are questions for you, not gaps I filled.

---

## 0. Orientation

Your repo already contains 90% of a paper: `docs/decisions.md` is the methods log,
`docs/explainers/00–06` are per-experiment method+results write-ups, `docs/project-outline.md`
is the narrative, `docs/design/` holds the pre-registrations, `sources/SUMMARIES.md` is the
related-work base. This skeleton tells you what to lift from where and what claims are
defensible as-is.

**Working titles** (pick one flavor):
- *Grammatical gender bleedthrough in open-weight LLMs: directions, steering, and a null at scale*
- *Does a language model think bridges are feminine? Grammatical-to-social gender bleedthrough in Qwen3*
- *The grammatical gender direction in Qwen3: shared across languages, steerable at 8B, absent at 14B*
- (workshop-friendly) *Mechanistic notes on grammatical gender bleedthrough: pre-registered checks, an estimator trap, and a size-dependent steering effect*

**One-paragraph abstract** (draft, to be tightened):

> In German *die Brücke* is feminine and in Spanish *el puente* masculine — for objects,
> gender is pure grammar. The famous human claim that this "qui t" shapes how speakers
> describe bridges (Boroditsky) has largely failed to replicate. We ask the same question
> *inside* a language model, where we can intervene rather than correlate: in Qwen3 base
> models (0.6B–14B, plus EuroLLM-1.7B as a cross-family check), we extract per-layer
> grammatical-gender directions from inanimate nouns with spelling, frequency, length, and
> concreteness controlled, verify them behaviorally, and then push nouns along them while
> reading out English adjective probabilities rated for gender by humans. We find: (1) the
> directions read gender, not spelling, at the noun's own token (28/35 layers at 8B) but not
> at a following token; (2) one direction partly generalizes across languages — it sorts
> flipped pairs and even Russian *-ь* nouns by each language's gender, with direction overlap
> well above a shuffle null; (3) steering a Spanish noun toward feminine moves English
> adjective choice toward feminine-rated words at 8B (about a third of a woman-man push) but
> not at 14B, with no growth in size; (4) even *unsteered*, a feminine article on a noun read
> as a thing pulls toward feminine-rated adjectives on every Qwen3 size tested but not on
> EuroLLM — the closest in-language analog of the bridge effect we find. We document two
> measurement artifacts (a noise-biased length estimator and a permutation test that could
> never fail) that would have produced false confirmations, and we report a pre-registered
> check that invalidated two of our own pooled readouts. Claims are scoped to model behavior,
> not human cognition.

**Contribution sentence** (what a reviewer should remember): a mechanism-level,
pre-registration-heavy map of grammatical→social gender bleedthrough in open-weight LLMs,
featuring (a) the strongest controlled steering evidence to date for the *concept* at 8B,
(b) a size-null at 14B, (c) an unsteered article-pull effect (P30) that survives a thing-
reading, and (d) a methods cautionary tale (the estimator trap + the P29 self-invalidation).

**What we explicitly do NOT claim** (keep this list in the intro/discussion):
- anything about human cognition, linguistic relativity in humans, or even about what the
  model "believes" — claims are about learned associations in a text-trained model;
- that the effect is present at all sizes (8B yes, 14B no, 0.6–4B mixed/in-between);
- that the German steering half of the question is answered (no valid test — frame failure);
- that R3-POOL/P20-POOL evidence survive (they don't; P29);
- that P30's mechanism is social gender (text-class statistics remain a candidate).

---

## 1. Introduction (≈3 pages)

- **The hook** (borrow post-1 outline in `docs/blog/series-outline.md`): the bridge, the human
  claim, the failed replications (Mickan et al. 2014; Samuel et al. 2019 review); the LLM
  version, and why intervention beats correlation.
- **The question + why the model is a good testbed:** read internals; intervene; no
  demand characteristics — but note the mirror-image limitation: an LLM effect doesn't tell us
  about humans either way (`docs/project-outline.md` §0; explainer 05 limits).
- **Prior work (§ related work below).**
- **Design preview:** five gated phases + side tests; every measurement choice pre-registered;
  the confirmatory/exploratory split.
- **Contribution + scope statement.**

### Related work (anchor to `sources/SUMMARIES.md`)
1. Humans: Boroditsky et al. 2003; Phillips & Boroditsky 2003; Mickan et al. 2014 (failed
   replication, reuses their noun lists/adjective ratings!); Semenuks et al. 2017 (partial
   positive); Samuel et al. 2019 (systematic review); Elpers, Jensen & Holmes 2022 (registered
   replication: native-speaker null, artificial-category training positive).
2. Embeddings: Kann 2019; Gonen et al. 2019; Flint & Ivanova 2024 (closest prior — same
   question in fastText/BERT); Williams et al. 2021 (corpus statistics).
3. LLM internals: Brinkmann et al. 2025 (2501.06346, shared latent grammatical concepts incl.
   gender, SAE features steerable in MT — cites your mechanism); Shtedritski & Mihaylov 2024
   (2407.09704, output-level cross-lingual gender adjective bias — prompting-level, no
   intervention); He & Mesgarani 2026 (2609.29328, grammatical "grandmother neurons" rare —
   motivates distributed directions/SAEs).
4. Methods: LEACE (Belrose et al. 2023, your Phase 1 eraser — in `sources/`); activation
   steering (Turner et al. 2023; Li et al. 2023 ITI); **Buttigieg et al. 2026 (2610.08559,
   the confidence confound — cite both as motivation AND as controls implemented)**.

> **(open)** Decide one-sentence per reference or a table; your SUMMARIES.md already has the
> substance.

## 2. Methods (≈6–8 pages incl. tables)

Structure it *as the log actually happened* (this is your strongest structural asset):
1. **Ground rules** (pre-registration, gold labels from Wiktionary, drop-don't-review, held-out
   frozen & asserted, same-chip rule) — `docs/decisions.md` first rows + `CLAUDE.md` hard rules.
2. **Stimuli & lexicons**: Wiktionary-based lexicons, eligibility filters, matched/stratified
   design (why spelling control by *design* after Phase 1 showed erasure doesn't generalize),
   the flipped pairs and their ~10% gloss-error caveat, the person-word/name lists (P24 family).
3. **Extraction & readouts**: `<|endoftext|>\n ` input convention (attention-sink fix), LAST vs
   AFTER readout decided by a *gender-blind* pre-registered check (this deserves its own
   paragraph — reviewers love it), layer indexing.
4. **Directions & the estimator**: stratified within-cell directions, Frisch-Waugh residualization,
   the logistic probe (fixed config), **the split-half estimator and why it's needed** (this is
   a paper-worthy methods contribution: plug-in length/cosine bias; cross-fitted dot products;
   within-cell shuffle null; the copies-in-one-half bootstrap rule).
5. **Behavioural gates**: Phase 0 known checks, frame checks on two families, the Phase 5
   agreement-flip gate (dose α*, random-flip limit), the readout frame checks.
6. **Steering & readouts in Phase 5**: noun-only push, dose in state-norm units, R1/R1-EN/R2/
   R3 readouts, the Glasgow-Norms regressions with valence/arousal/size/frequency covariates
   (gender×valence r ≈ −0.47!), the 100-random-direction rank test, damage matching, number &
   social controls, cross-language push (P26).
7. **Pre-specified verdict rules** (majority of layers; Holm across co-primaries; language
   agreement rule).
8. **New-this-paper controls (addendum):** confidence-direction cosine, entropy/Kendall-τ rank
   preservation, baseline-logP covariate — see `outputs/confidence-direction-control-spec.md`; to
   be added to decisions.md + methods before submission.

Include a **table of all experiments** (phase/P-number, question, pre-reg status, verdict) —
you have everything needed in `docs/glossary.md` + explainers.

## 3. Results (organize like the explainers: per experiment, verdict-shaped)

Suggested order (chronological = your repo's own logic):

1. **Phase 0:** known rates by model/size (0.6B–14B + EuroLLM); multi-gender items (models
   lean to the frequent sense).
2. **Phase 1:** outcome (b) — spelling cannot be erased generally; consequences for design.
3. **Phases 2–3:** Spanish direction reads gender (28/35 @ 8B, 26/39 @ 14B, LAST only);
   German compounds follow the head (35/35, 39/39); markedness null (and *why* — the
   estimator trap, put the numbers here). Both **position-dependent** (vanish at AFTER).
4. **Phase 4:** partially shared across languages — T3 flipped pairs (paired AUC ≈ 0.9),
   T4 Russian (≈0.8), T1 strict transfer fails, T2 overlap (cosine ≈ 0.4) above null.
5. **Phase 5:** the headline table:
   | | 8B | 14B |
   |---|---|---|
   | R1 grammatical push | +0.020 (3/4 layers) | 0.000 (0/3) |
   | R1 random 95% | 0.012 | 0.008 |
   | R1 social push (control) | 0.064 | 0.047 |
   | R1-EN grammatical | +0.011 (3/4) | −0.001 (0/3) |
   with the German no-valid-test box, R2 (man/woman moves on both but noun is visible),
   R3-POOL **and its P29 invalidation** (report both: the positive, then the killer control —
   this is a feature of your paper, not a bug).
6. **P30 (unsteered in-language):** article pull on a thing-noun (object +0.080/+0.124 8B/14B;
   EuroLLM ≈ 0; size curve 1.7B +0.038, 4B +0.026). Discussion anchor.
7. **English side (P22b/P24 family):** Glasgow sanity (EN ρ 0.51–0.59 → rating transfer OK;
   Spanish invariant weak after pleasantness — a caveat on R3-family reads); ES direction
   sorts English person words/names/pairs by sex (0.85–0.92); overlap with social halves on
   14B (0.44→0.22) — triangulates the steering null.

### Claims × evidence × controls table (write-up dashboard)

| Claim | Evidence | Control it already has | Status |
|---|---|---|---|
| Directions read grammatical gender, not spelling | strat probe reads gender; exceptions/homographs | within-ending cells, residualised covariates, position check | strong |
| Gender lives on the noun's own token | LAST yes, AFTER no | gender-blind position rule | strong, must state |
| Partly shared across languages | T3/T4/T2 | shuffle-null (cross term), cognate robustness | strong-ish; state "partly" |
| Steering bleeds grammar→social at 8B | R1/R1-EN ranked vs random | 100 rand dirs, damage match, number & social controls | good *if* confidence checks pass |
| No bleed at 14B | R1 0/3 | social control moves (sensitive readout) | good null |
| Bare article pulls feminine-rated words (unsteered) | P30 object reading | thing-reading probe; EuroLLM null; size curve | good descriptive; mechanism open |
| R3-POOL/ P20-POOL evidence | — | **dead by P29** | do not claim |
| Confidence confound absent | — | **to run** (spec) | pending |

## 4. Discussion (≈2–3 pages)

- **What the pattern means:** grammar→social association exists in Qwen3's learned statistics;
  8B can be pushed, 14B sorts-but-can't-be-pushed (the P24-vs-Phase5 divergence: sorting and
  steering ask different things — explainer 06's point).
- **The size puzzle** (8B yes, 14B no): descriptions, not explanations; candidates (layers
  tested, dose, saturation, family-specific) — see outline §15's caveat that 14B wasn't
  testable at 70% depth where 8B's effect peaked.
- **P30's two readings** (social vs text-class statistics) and what would separate them (next
  experiments box).
- **The German half is unanswered** (frame failure; the pre-registration discipline that led
  to "no valid test" rather than post-hoc frame-switching — worth defending explicitly).
- **What this does NOT say about humans** (one firm paragraph; cite Elpers 2022's null +
  Boroditsky's causal-training positive as the human-side map).
- **Methodological lessons** (the estimator trap, the impossible permutation test, P29) as a
  stand-alone subsection — this is your differentiating contribution.

## 5. Limitations (lift from the explainers' "Limits worth stating" sections)

- Position-dependence; single-token readouts only; Glasgow norms applied via translation
  (Spanish weak — P22b); gloss mismatch (~10%); two sizes opposite answers; Spanish-only
  verdicts; layer-depth mismatch (8B vs 14B); steering = sufficiency, not usage; the
  confidence confound status (to be reported per the spec); fp16 Mac vs bf16 pod runs.

## 6. Next experiments box (inside Discussion, or appendix)

- Confidence controls (spec). · P30 mechanism test (text-class baseline). · Steered *text
  generation* (P21 data already browsable). · Gemma 3 replication (confirmed family). ·
  EuroLLM steering if a frame passes. · SAE feature-level analysis (reuse `src/gbleed/sae.py` +
  roadmap `outputs/sae-neuron-roadmap.md`).

## 7. Appendix plan

- Stimulus cards (per list: counts, filters, review sheets — `docs/screening/`, `docs/reports/`).
- Full per-layer tables (results/ folders), per the "report every layer" convention.
- The decision log as a table (date/decision/why → already `docs/decisions.md`).
- P21 top-50 browse samples as qualitative exhibits.
- Glossary (`docs/glossary.md`) as a reader aid.

## 8. Venue/packaging (open)

- **Primary:** arXiv preprint + workshop submission (e.g., BlackboxNLP (gender/bias themed),
  or a mechanistic-interpretability workshop — matches the methods-heavy style).
- **Blog first** (see `outputs/writing-pack-casual.md` + your `docs/blog/series-outline.md`),
  then the paper — the blog being the "exploratory map", the paper the confirmatory core.
- Suggested order of writing: methods §2.4 (estimator), results Phase 5 table, then the blog
  posts 4/5, then the full draft.

---

## Source map (where each section's material lives in the repo)

| Paper section | Repo source |
|---|---|
| Intro hook | `docs/blog/series-outline.md` post 1; `sources/SUMMARIES.md` |
| Related work | `sources/SUMMARIES.md`; `docs/reading-list.md` |
| Methods ground rules | `CLAUDE.md`; `docs/decisions.md` (top rows); `README.md` |
| Stimuli | `README.md` (lexicon section); `docs/screening/`; `docs/reports/` |
| Estimator | `docs/methods-summary.md`; `src/gbleed/estimators.py`; decisions 2026-10-06/07 entries |
| Phase results | `docs/explainers/00–06`; `results/runpod{,2}/` |
| Phase 5 steering | `docs/explainers/05`; `docs/design/phase5-design.md`; `src/gbleed/phase5.py` |
| P29/P30 | `docs/explainers/05` (P29/P30 sections); `docs/design/pooling-neutrality-prereg.md`, `p30-object-context-prereg.md`; outline §16 |
| Confidence controls | **this spec** (`outputs/confidence-direction-control-spec.md`) + 2610.08559 |
| Limits | each explainer's "Limits worth stating"; outline §15–16 |
| Narrative/chronology | `docs/project-outline.md` |
