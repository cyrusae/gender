# docs/: a map

*What's in this folder and where to look for what. Written 2026-10-26, when base-model data
collection closed and the write-up began. Code and commands are in the top-level `README.md`.*

## Start here

| If you want to… | Open |
|---|---|
| Get back up to speed after a break | [`project-outline.md`](project-outline.md): everything done so far, in order |
| See the whole study on one page | [`methods-summary.md`](methods-summary.md) |
| Look up a term | [`glossary.md`](glossary.md) (*(tool)* marks software terms) |
| Find out why something was done a certain way | [`decisions.md`](decisions.md): every choice, dated, newest last |
| Understand one phase's method and numbers | [`explainers/`](#explainers-how-each-method-works) |
| Write | [`writeups/`](#writeups-drafts-for-readers) (the current drafts), then `feynman/writing-pack-casual.md` |

## The core files (kept current)

- **[`project-outline.md`](project-outline.md)**: the chronological "what we've done". Updated
  after every phase, pod session or other major step.
- **[`decisions.md`](decisions.md)**: the decisions log (date, decision, why, alternatives).
  Tentative entries say so; instruct-study entries are tagged **[instruct]**. This is the
  source for the methods section.
- **[`glossary.md`](glossary.md)**: every term, grouped by area.
- **[`methods-summary.md`](methods-summary.md)**: a one-page summary of question, methods and
  results.

## `explainers/`: how each method works

One per phase or major add-on. Each says what the measurement is, why it's built that way, how
to read the numbers, and which limits to state in a write-up.

| file | covers |
|---|---|
| [00](explainers/00-phase0-behavioural-check.md) | Phase 0: does the model know each noun's gender? (the margin, "known") |
| [01](explainers/01-phase1-spelling-eraser.md) | Phase 1: erasing *-a*/*-o* spelling (LEACE), outcome (b) |
| [02](explainers/02-phase2-spanish-gender-direction.md) | Phase 2: Spanish gender direction, tested on exceptions and homographs |
| [03](explainers/03-phase3-german-three-genders.md) | Phase 3: German three genders, compounds, markedness |
| [04](explainers/04-phase4-shared-gender.md) | Phase 4: is gender one direction across languages? (flipped pairs, Russian) |
| [05](explainers/05-phase5-steering.md) | Phase 5: steering a noun's gender; P29/P30 (the Spanish object test) |
| [06](explainers/06-english-social-gender.md) | P23/P24: does the gender direction point at "woman"? (English lists) |
| [07](explainers/07-controls-and-german-objects.md) | P31 (German object test), confidence control, German frequency axis |
| [I0](explainers/I0-instruct-pilot0.md) | Instruct study, pilot 0: turning things into characters |

## `design/`: plans and pre-registrations

Written **before** the data they describe. A title marked "ADOPTED, PI date" is binding; later
changes are amendments inside the file or entries in `decisions.md`.

**Big plans**
- [`phases-3-5-plan.md`](design/phases-3-5-plan.md): the plan that extended the original design doc to Phases 3–5.
- [`phase4-design.md`](design/phase4-design.md) (Q1–Q6) and [`phase5-design.md`](design/phase5-design.md) (P1–P28).
- [`runpod-session-plan.md`](design/runpod-session-plan.md): pod sessions 1 and 2, plans and outcomes.
- [`new-language-playbook.md`](design/new-language-playbook.md): checklist for adding French, Dutch etc. (each step comes from something that went wrong once).

**Pre-registrations, in the order they were run**

| file | question |
|---|---|
| [`suffix-followup-prereg.md`](design/suffix-followup-prereg.md) | German suffix follow-up (D1–D4) |
| [`pooling-neutrality-prereg.md`](design/pooling-neutrality-prereg.md) | P29: is the pooled adjective readout neutral to the article? (failed) |
| [`p30-object-context-prereg.md`](design/p30-object-context-prereg.md) | P30: does *la* pull toward feminine adjectives on an object? (Spanish) |
| [`p31-german-object-context-prereg.md`](design/p31-german-object-context-prereg.md) | P31: the German version of P30 |
| [`confidence-ctl-prereg.md`](design/confidence-ctl-prereg.md) | Confidence/flattening control for every steering result |
| [`frequency-axis-prereg.md`](design/frequency-axis-prereg.md) | Why the German direction points toward rarer words |
| [`register-axis-prereg.md`](design/register-axis-prereg.md) | Is that frequency axis partly register? |
| [`rarity-lean-heldout-prereg.md`](design/rarity-lean-heldout-prereg.md) | Does the rarity lean reach new simple nouns? |
| [`gemma-slim-prereg.md`](design/gemma-slim-prereg.md) | Gemma 3 replication (+ companion note [`gemma-outlier-dimension.md`](design/gemma-outlier-dimension.md)) |
| [`russian-direction-prereg.md`](design/russian-direction-prereg.md) | A Russian gender direction: does it align, does it lean? |
| [`instruct-replication-prereg.md`](design/instruct-replication-prereg.md) | The base-model study rerun on instruct models, stage A (phase 2 queue) |

[`confidence-direction-control-spec.md`](design/confidence-direction-control-spec.md) is an
identical copy of the research-agent spec in `feynman/`; the adopted version is
`confidence-ctl-prereg.md`.

## `screening/`: how words are chosen

Reference docs for the word lists: every filter, in order, why it exists and what it costs.
Its own index is [`screening/README.md`](screening/README.md) (lexicon, known checks, Spanish,
German, flipped pairs, Russian, nonce words).

## `reports/`: review sheets and their outcomes

Sheets generated for the PI to review in English, plus the record of what was dropped.

- `flipped-pairs-review.md` + `flipped-pairs-flags.csv`: the German–Spanish pairs (all 519 read; drops and reasons).
- `phase3-reference-set.md`: German neuter reference set.
- `russian-pool-review.md`: Russian *-ь* nouns.
- `*-review.csv`: English person words, names and gender pairs; Spanish invariant adjectives; German diminutives; Phase 5 number list.
- `tokenization-qwen3.md`: how many tokens each noun takes in Qwen3.

## `writeups/`: drafts for readers

| file | what it is |
|---|---|
| [`2026-10-06-progress-so-far.md`](writeups/2026-10-06-progress-so-far.md) | Early progress post, Phases 0–2 (out of date) |
| [`2026-10-22-methods-results-draft.md`](writeups/2026-10-22-methods-results-draft.md) | **Main draft**: methods and results for a reader new to the project |
| [`2026-10-23-methods-paper-style.md`](writeups/2026-10-23-methods-paper-style.md) | Same methods in conventional paper layout (companion, not a replacement) |

Related: [`blog/series-outline.md`](blog/series-outline.md), bullet outlines for a blog series
(numbers as of 2026-10-09).

## Background and side material

- **[`feynman/`](feynman/)**: briefs from an outside research agent (2026-10-21): a repo
  exploration, a reading list, paper deep dives (Elpers 2022, Buttigieg 2026, Shtedritski &
  Mihaylov 2024), an exploratory paper skeleton, an SAE/neuron roadmap, and a casual writing
  pack. Useful input, **not** project records: where they disagree with `decisions.md` or the
  explainers, those win.
- **[`conversations/`](conversations/)**: saved chat answers worth keeping: the 2026-10-06 methods
  overview (source of the Phases 3–5 plan), GPU speed-ups, and raking (weighting a lopsided
  sample).
- [`reading-list.md`](reading-list.md): references that came up but aren't in `sources/`.
  (Papers we have are summarised in `sources/SUMMARIES.md`.)
- [`model_candidates.md`](model_candidates.md): the 2026-10-05 survey of candidate models.

## Elsewhere in the repo

- The original design (phases 0–5): `Grammatical Gender Bleedthrough in LLMs Project Design.pdf` (repo root).
- Result summaries: `results/` (pod results under `results/runpod/`, `results/runpod2/`, `results/p29/`).
- Instruct study's own design notes: `instruct/docs/` and rules in `instruct/CLAUDE.md`.
- Bulky data (activations, sweeps, dumps): the external drive, `/Volumes/Lexar/gbleed`.
