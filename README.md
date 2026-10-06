# Grammatical Gender Bleedthrough in LLMs

Does a model's representation of grammatical gender, learned from inanimate nouns,
overlap with its representation of social gender? Study design:
`Grammatical Gender Bleedthrough in LLMs Project Design.pdf` (gated phases 0–5).
Background literature: `sources/SUMMARIES.md`.

## Status

| phase | question | status | explainer |
|---|---|---|---|
| 0 | Does the model know the genders? | done: Qwen3 0.6B/1.7B/4B (+ EuroLLM-1.7B) | [00](docs/explainers/00-phase0-behavioural-check.md) |
| 1 | Can a spelling (-a/-o) eraser be built? | done: outcome (b), erasure doesn't generalise to new words | [01](docs/explainers/01-phase1-spelling-eraser.md) |
| 2 | Spanish gender direction: gender or spelling? | in progress (stimuli built, known-check running) | — |
| 3 | German gender direction | not started | — |
| 4 | Shared across languages? | not started | — |
| 5 | Bleed into social gender? | not started | — |

Model family: **Qwen3 base** (0.6B/1.7B/4B on the Mac for development; full sequence plus
8B/14B on RunPod, one GPU type, bf16). Every choice that shapes results, with the reason, is
in [`docs/decisions.md`](docs/decisions.md); terms are in [`docs/glossary.md`](docs/glossary.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned.

```sh
uv sync                 # .venv with torch, transformers, concept-erasure, scikit-learn, wordfreq, nltk
uv run pytest           # stimulus-file validation, heuristics, analysis-logic guards
uv run ruff check . && uv run ruff format .
```

Gated models (Llama, Gemma, ...) need `uv run hf auth login` and accepting the terms on the
model's Hugging Face page. Qwen3 is ungated.

## Conventions (see CLAUDE.md for the full list)

- **Gold labels never come from a language model.** Genders come from Wiktionary.
- **Measurement choices are fixed on grammatical grounds and pre-registered** in
  `decisions.md` before results; analyses added afterwards are labelled exploratory.
- **Held-out test sets are frozen before training**, split by stem, and asserted disjoint.
- **Cross-model comparisons use the shared set**: items known by every model compared.
- **Stimulus lists are versioned** (`_vN`); a list with results is never edited.
- **Commit before runs**: result metadata records `git_commit` and `git_dirty`.
- **Strict filtering**: anything that would need a manual check is dropped; anything a person
  must review is checkable in English (glosses).

## Pipeline at a glance

```sh
# Noun lexicons (once; downloads ~1 GB per language to data/raw/)
uv run gbleed lexicon                          # data/lexicon/{de,es}_nouns.csv, pairs, multi-gender candidates
uv run gbleed classics                         # Wiktionary-check the hand-picked classic pairs

# Phase 0
uv run gbleed sample-phase0                    # -> data/stimuli/phase0_vN.csv
uv run gbleed phase0 MODEL [MODEL ...] --stimuli data/stimuli/phase0_v3.csv
uv run gbleed phase0-compare [--by freq_bin|set|de_suffix|es_exception]
uv run gbleed multi-check MODEL [MODEL ...]    # same-spelling two-gender items (See, mar)

# Phase 1
uv run gbleed phase1-stimuli                   # verbs / nonce / nouns (v1)
uv run gbleed phase1 MODEL [MODEL ...] [--skip-extract]

# Phase 2
uv run gbleed phase2-stimuli                   # pools -> data/stimuli/phase2_pool_v2.csv
uv run gbleed phase0 Qwen/Qwen3-0.6B-Base Qwen/Qwen3-1.7B-Base Qwen/Qwen3-4B-Base \
    --stimuli data/stimuli/phase2_pool_v2.csv --out results/phase2_known
uv run gbleed phase2-finalize                  # keep nouns known by all three; freeze splits
uv run gbleed phase2 MODEL [MODEL ...] [--skip-extract]
```

Model runs take minutes to an hour; run them in the background and write to a log.

Phase 2 also uses human concreteness ratings (Brysbaert, Warriner & Kuperman 2014), downloaded on
first use to `data/raw/norms/` from a mirror and checked against a pinned checksum (`src/gbleed/norms.py`).

## Noun lexicons (`data/lexicon/`)

Built from English Wiktionary via kaikki.org (dump date and SHA-256 in
`data/lexicon/metadata.json`; the full `{de,es}_nouns.csv` tables are git-ignored and rebuilt by
`gbleed lexicon`). One row per noun lemma:

| column | meaning |
|---|---|
| `gender` / `genders` | `m`, `f`, `n`, or `multi`; `multi_type` = `meaning_split` (*der See* lake / *die See* sea) or `free_variation` (*el/la mar*), with `gloss_m` / `gloss_f` |
| `gloss`, `concept_en` | first English gloss and its normalised head (used for pairing) |
| `zipf`, `zipf_en` | frequency in the language and in English (wordfreq; 3 ≈ once per million words) |
| `animacy`, `animacy_reason` | `inanimate` / `animate` / `uncertain`, and why |
| `concrete` | dominant WordNet sense is a physical thing |
| `marked`, `regions` | register/region tags of the first sense; letter and number names |
| `also_pos` | same spelling is another word or a given name (*aber*, *de*, *Charlotte*) |
| `also_form` | same spelling is an inflected form of another word (*camino* < *caminar*, *Plane* = pl. of *Plan*) |
| `de_suffix` | German gender-predicting suffix (`-ung`, `-heit`, ...) |
| `es_ending`, `es_regular`, `es_exception` | Spanish -o/-a regularity; exception type `greek_ma` / `clipping` / `other` |
| `initial_a_f` | feminine noun starting with stressed a-/ha- (takes *el*: *el agua*) |

**Eligible** stimuli: single-gender m/f, inanimate, unmarked, not regional, no `also_pos` or
`also_form`, not more frequent in English, and (Spanish) not `initial_a_f`.

**Reliability.** Genders come straight from Wiktionary tags and are reliable. Animacy and
concreteness are heuristics (Wiktionary categories and tags of the first sense, gendered
counterpart forms like *director/directora*, English gloss patterns, WordNet) that err toward
excluding; each row says why, in English.

**German–Spanish pairs** (`pairs_de_es.csv`) are matched on the first English gloss. `strict`
pairs need no checking: each noun is ≥ 10× more frequent than any other noun with the same gloss,
the two are within 10× of each other in frequency, and neither is a DE–ES cognate or overlaps
with English (`en_same`: ≈ the English word; `en_homograph`: a common English word). `en_cognate`
(similar to the English word, e.g. *Lunge*~lung) is kept but flagged so Phase 4 can compare
cognate and non-cognate pairs. Even strict pairs are ~10% loose (*Platte/apartamento* "flat"):
skim the final Phase 4–5 pairs in English before use.

## Hand-picked items (`data/stimuli/`)

| file | contents |
|---|---|
| `classics_spec.csv` → `classics.csv` | 19 classic flipped pairs (Kann 2019 after Boroditsky & Schmidt; Mickan et al. 2014; design doc); `gbleed classics` adds Wiktionary genders and a `check` column (e.g. *luna*/*estrella* are also women's names; *Bank* is an English word) |
| `multi_gender_review.csv` → `multi_gender_spec.csv` | PI-reviewed same-spelling two-gender items (German Wiktionary + RAE checked): 22 kept, 10 dropped, each with its reason |
| `exceptions_spec.csv` | Phase 2 exceptions (-ma, clippings, *día*/*mano*), with documented overrides for *moto* and *mano* |
| `phase0_seed.csv` | the original hand-written seed list (labels `claude-unverified`); pipeline testing only |

## Phase 0: does the model know the genders?

Each noun is scored by how well it fits after the masculine vs the feminine article
(log-probability of the noun and the rest of the sentence *given* the article), in two frames:

| | German | Spanish |
|---|---|---|
| frame 1 | `Das hat etwas mit dem/der X zu tun.` | `Esto tiene que ver con el/la X.` |
| frame 2 | `Hier ist ein/eine X.` | `Aquí hay un/una X.` |

A noun is **known** (`passed`) when both frames favour the right article by at least
`--min-margin` nats (default 1.0: the right article makes the noun ≥ ~2.7× more likely; a *nat*
is a natural-log unit). Near-ties are `unsure`, one frame right and one confidently wrong is
`conflict`. The headline number is `known_bal`, the balanced share of known nouns. A few-shot
`noun: article` quiz is also run as a diagnostic only. Results: `results/phase0/<model>/`.

**Multi-gender check** (`multi-check`): for *See*/*mar*-type items both articles are correct, so
the question is whether the model accepts both (|margin| < 2.3 nats in both frames). Most items
don't: models lean toward the more frequent variant or sense. Results:
`results/multigender/<model>/`.

## Phase 1: spelling eraser

LEACE erasers for -o vs -a, fit on regular -ar verb pairs (*hablo/habla*, where the ending marks
person, not gender) and tested on Spanish-looking nonce pairs (*flitra/flitro*) split by stem.
Words are fed as `<|endoftext|> word` (position 0 is an attention sink). Probes are always trained
on words the eraser never saw, and report ROC AUC as well as accuracy. Result: the verb eraser
leaves nonce endings fully readable at every layer and size, and even nonce-fitted erasers leave
new nonce words at AUC ~0.8, because the -a/-o contrast is largely word-specific. Results:
`results/phase1/<model>/`.

## Phase 2: Spanish gender direction (in progress)

Because erasure doesn't generalise, the main spelling control is the training set itself:

| set | role |
|---|---|
| `matched` | **train**: nouns whose ending carries no gender information by construction (equal m/f per final two letters; no -o/-a or gender-predicting suffixes), excluding English-overlapping and sex-associated nouns |
| `regular` | train (comparison) and test: regular -o masc / -a fem nouns, same exclusions |
| `exception_*` | **test only**: -ma (*el problema*), clippings (*la foto*, *la moto*), others (*el día*, *la mano*) |
| `homograph` | **test only**: the same string as noun (*mi camino*) and verb (*yo camino*, *usted cuenta*) |
| multi | **test only**: Spanish *mar*-type items, with the article (*el mar* / *la mar*) |

Test sets keep `en_overlap` and `sex_assoc` flags for with/without sensitivity runs. The
pre-registered reading (including the amendment made before results) is in `decisions.md`.

## Layout

```
data/stimuli/     stimulus lists and hand-picked specs (versioned, tracked)
data/lexicon/     pairs, multi-gender candidates, metadata (full noun tables git-ignored)
data/raw/         Wiktionary dumps, NLTK data (git-ignored)
src/gbleed/       models, scoring, stimuli, lexicon, phase0, multigender, activations,
                  phase1(_stimuli), phase2(_stimuli), cli
results/          small result tables (tracked): phase0/, multigender/, phase1/, phase2_known/, phase2/
activations/      extracted hidden states (git-ignored; one device and precision per model)
docs/             glossary, decisions log, per-phase explainers, writeups/, model_candidates.md
sources/          papers (PDFs git-ignored) + SUMMARIES.md
```
