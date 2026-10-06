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

Each noun is scored by comparing the log-probability of the same sentence with the
masculine vs the feminine article, in two frames (details in `src/gbleed/phase0.py`):

| | German | Spanish |
|---|---|---|
| frame 1 | `Das hat etwas mit dem/der X zu tun.` | `Esto tiene que ver con el/la X.` |
| frame 2 | `Hier ist ein/eine X.` | `Aquí hay un/una X.` |

A noun is **known** (`passed`) when both frames favour the right article by at least `--min-margin` nats (default 1.0, i.e. the right article makes the noun at least ~2.7× more likely than the wrong one; a *nat* is a natural-log unit, see `docs/glossary.md`). Near-ties are `unsure`, not wrong; one frame right and the other confidently wrong is `conflict`. The headline number is `known_bal`, the balanced share of known nouns (a model that always says *die* would otherwise score ~50%). A few-shot `noun: article` quiz is also run, as a diagnostic only.

Outputs go to `results/phase0/<model>/`: `items.csv` (per-noun margins, verdicts, status, tokenization) and `summary.json` (rates, breakdowns by frequency bin / set / suffix / exception type, run metadata incl. device/dtype/versions). `phase0-compare` (optionally `--by freq_bin` etc.) builds the cross-model tables.

## Layout

```
data/stimuli/     stimulus lists (long-format CSV, see src/gbleed/stimuli.py)
src/gbleed/       package: models.py, scoring.py, stimuli.py, phase0.py, cli.py
results/          small result tables (tracked)
activations/      extracted activations (git-ignored; one device per model)
docs/             glossary.md, decisions.md, explainers/ (one per phase), model_candidates.md
notebooks/        exploratory analysis
sources/          papers (PDFs git-ignored) + SUMMARIES.md
```

## Noun lists from Wiktionary

Gold genders come from Wiktionary (English Wiktionary, via the kaikki.org extracts),
not from any language model.

```sh
uv run gbleed lexicon                 # downloads ~1 GB per language to data/raw/ (once), then
                                      # writes data/lexicon/{de,es}_nouns.csv + pairs_de_es.csv
uv run gbleed sample-phase0           # -> data/stimuli/phase0_v3.csv
uv run gbleed sample-phase0 --per-cell 40 --seed 1 --out data/stimuli/phase0_v4.csv
```

`data/lexicon/{lang}_nouns.csv` has one row per noun lemma:

| column | meaning |
|---|---|
| `gender` / `genders` | `m`, `f`, `n`, or `multi` (e.g. *der/die See*, *el/la mar*: useful later as spelling-constant tests) |
| `gloss`, `concept_en` | first English gloss and its normalised head (used for pairing) |
| `zipf` | word frequency (wordfreq; 3 ≈ once per million words) |
| `animacy`, `animacy_reason` | `inanimate` / `animate` / `uncertain`, and why (heuristic: see below) |
| `marked`, `regions` | register tags (archaic, slang...) and region tags of the first sense; letter/number names |
| `also_pos` | the same spelling is also a verb/adjective/etc. (frequency is then unreliable; also the Phase 2 homograph pool) |
| `de_suffix` | German suffix that predicts gender (`-ung`, `-heit`, ...) |
| `es_ending`, `es_regular`, `es_exception` | Spanish -o/-a regularity; exceptions typed `greek_ma`, `clipping`, `other` |
| `initial_a_f` | feminine noun starting with a-/ha- (may take *el*, as in *el agua*) |

`eligible` stimuli are single-gender m/f, inanimate, unmarked, non-regional, not
homographs, and (Spanish) not `initial_a_f`. `sample-phase0` draws `--per-cell` of them per
language × frequency bin (low 2.5–3.5, mid 3.5–4.5, high ≥ 4.5) × gender, plus up to
`--max-pairs` flipped and same-gender control translation pairs.

**What is and isn't reliable:**
- *Gender*: taken directly from Wiktionary's tags. Reliable.
- *Animacy*: heuristic (Wiktionary categories of the first sense, the
  "by-personal-gender" tag, English gloss patterns, WordNet's category for the gloss).
  It errs toward excluding. Each row says why, in English, so it can be spot-checked
  without knowing German.
- *Translation pairs*: matched on the first English gloss, so some are loose
  (e.g. *Zeit/vez* both gloss as "time"). `pairs_de_es.csv` keeps both glosses: skim
  the flipped pairs before Phase 4–5. Pairs that are near-identical spellings are flagged
  `cognate` and left out of sampling.

**Hand-picked items.**
- `data/stimuli/classics_spec.csv`: the 19 classic flipped pairs from Kann (2019, after
  Boroditsky & Schmidt), Mickan et al. (2014) and the design doc. Edit this file to add
  or remove items; `uv run gbleed classics` looks up their genders in Wiktionary and writes
  `classics.csv` with a `check` column (e.g. *estrella*'s person sense, *disco*'s second
  gender). `sample-phase0` always includes them (`set = classic`).
- `data/lexicon/multi_gender_candidates.csv`: same spelling with both m and f
  (*der/die See*, *el/la cometa*, *el/la mar*), classified `meaning_split` vs
  `free_variation`, with the gloss for each gender. A pool for hand-picking
  spelling-constant tests; the classification is rough, so read the glosses.

`data/stimuli/phase0_seed.csv` is the original hand-written seed list (labels marked
`claude-unverified`); it's only for testing the pipeline.
