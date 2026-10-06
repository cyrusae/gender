# Grammatical Gender Bleedthrough in LLMs

Does a model's representation of grammatical gender, learned from inanimate nouns,
overlap with its representation of social gender? See
`Grammatical Gender Bleedthrough in LLMs Project Design.pdf` for the study design and
`sources/SUMMARIES.md` for the background literature.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned.

```sh
uv sync                 # creates .venv with torch, transformers, concept-erasure, sklearn, ...
uv run pytest           # validates stimulus files and prompt construction
```

Gated models (Llama, Gemma, ...) need a Hugging Face login and accepting the model's
terms on its HF page: `uv run hf auth login`.

## Phase 0: behavioural gender check

```sh
uv run gbleed phase0 Qwen/Qwen3-0.6B-Base other/model-id ...   # one or more models
uv run gbleed phase0 MODEL --dtype float32                      # if fp16 gives NaN/inf (e.g. Gemma)
uv run gbleed phase0 MODEL --stimuli data/stimuli/other.csv --langs de
uv run gbleed phase0-compare                                    # table of every model run so far
```

Each noun is scored two ways (details in `src/gbleed/phase0.py`):

- **meta**: few-shot `noun: article` list; next-token P(der) vs P(die), P(el) vs P(la),
  averaged over two shot orderings.
- **ctx**: sentence log-probability with each article, in frames where the wrong article
  can't be read as a plural or contraction
  (`Das hat etwas mit dem/der X zu tun.`, `Esto tiene que ver con el/la X.`).

Outputs go to `results/phase0/<model>/`: `items.csv` (per-noun scores, prediction,
tokenization) and `summary.json` (accuracies, balanced accuracies, per-gender accuracy,
run metadata incl. device/dtype/versions). `results/phase0/comparison.csv` is the
cross-model table. Use **balanced** accuracy: a model that always says *die* still scores
~50% raw.

## Layout

```
data/stimuli/     stimulus lists (long-format CSV, see src/gbleed/stimuli.py)
src/gbleed/       package: models.py, scoring.py, stimuli.py, phase0.py, cli.py
results/          small result tables (tracked)
activations/      extracted activations (git-ignored; one device per model)
docs/             notes, e.g. model_candidates.md
notebooks/        exploratory analysis
sources/          papers (PDFs git-ignored) + SUMMARIES.md
```

## Stimulus labels

`data/stimuli/phase0_seed.csv` is a **seed list for testing the pipeline**; its gold
genders were written from model knowledge and are marked `claude-unverified`. Real lists
should take gold genders from a dictionary source (e.g. Wiktionary via kaikki.org) and
record that in the `source` column.
