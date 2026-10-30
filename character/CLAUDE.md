# character/ — character study (instruct models; scoped rules)

This folder is a separate study: what instruct-tuned (chat) models *say* about things, read as
characters. It lives on branch `character` (worktree `worktrees/gender-character`) so it doesn't
collide with base-model work on `main`. The root `CLAUDE.md` applies here **except where this
file overrides it**. Status of these overrides: **ADOPTED by the PI (2026-10-29)** (see
`character/docs/design.md` §Rule changes).

## Overrides of root rules (inside `character/` only)

- **Instruct models allowed.** Root "Base (pretrained) models only" does not apply here. Still no
  4-bit/8-bit quantization; bf16 on RunPod; record backend (vLLM/transformers) and version.
- **Model judges of outputs are allowed, as measurement instruments, not as gold labels.**
  Root "gold linguistic labels never come from a language model" still holds for every *input*
  label (noun genders: Wiktionary via `gbleed lexicon`; English gender associations: Glasgow Norms).
  Coding *generated text* may use a model classifier only if:
  1. a deterministic coder (regex) can't measure the construct (pronouns are always regex);
  2. the classifier, its prompt/labels and version are frozen and pre-registered before outcome
     data are coded;
  3. it is validated against a human-coded English sample (the PI can code English outputs) and
     agreement is reported;
  4. what it sees is fixed in its pre-registration. The default is blind to condition (the
     paragraph only). A judge that also sees the noun or the prompt (e.g. "how well does this
     fulfil the prompt?") must say why in its pre-registration, and its scores are reported as
     unblinded.
  Its gender biases are a known confound: never use it for the primary gender outcome.
- **Sampling, not log-probs, is the default measure.** Sampling settings come from each model's
  published recommended defaults (`configs/`), never tuned on outcomes.

- **Pilot iteration.** On reserved pilot nouns only, prompts may be revised on any grounds,
  including outcomes. Before any non-pilot noun is run, the prompt is frozen and pre-registered,
  and the write-up says which pilot results informed it. Results on pilot nouns are never
  reported as findings.

- **Serving precision.** Any run on our own GPUs, and any result that feeds activation or
  steering work: full precision (or the model's training precision) only. API completions:
  full or training precision is strongly preferred. Every call pins one provider with no
  fallbacks and records the provider and its stated precision. A quantized or unknown-precision
  endpoint may be used for a finding only when no full-precision endpoint exists for that model
  (e.g. closed models from major labs). The model and its precision tier must be named in the
  pre-registration before any non-pilot noun is run. Those results carry a caveat (unknown
  precision, possible hidden system layers, possible silent updates), use a dated model version
  where one exists, and are labelled by tier in every comparison. Pilot runs on reserved nouns
  may use any endpoint, labelled "variety tier".

## Unchanged and especially important here

- Don't tune prompts on outcomes. Prompt wording is fixed on design grounds before a run; a
  prompt that fails a *manipulation* check (e.g. the model writes about the object, not a
  person) is fixed on principle and logged.
- **Pilot nouns are burned**: anything in `stimuli/reserved_nouns.csv` never appears in a later
  instruct stimulus list. Builders must assert this.
- Version stimulus/prompt files (`_vN`); don't edit one that has results.
- Log decisions in the root `docs/decisions.md` with a `[character]` tag (entries before 2026-10-27 use `[instruct]`); terms in
  `docs/glossary.md`; explainers as `docs/explainers/C0-…md`.

## Layout

- `configs/` prompt templates, conditions, per-model sampling settings (versioned)
- `stimuli/` noun lists (genders from the Wiktionary lexicon, English ratings from Glasgow)
- `gen.py` generation (vLLM on the pod, transformers on the Mac for smoke tests)
- `code_outputs.py` deterministic coding (pronouns, gendered nouns, CoT flags) + summaries
- `runpod/` pod scripts; `results/` outputs (jsonl + summaries)
