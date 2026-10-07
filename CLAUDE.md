# CLAUDE.md

Research project: does an LLM's representation of grammatical gender (learned from inanimate nouns) overlap with social gender? Design: `Grammatical Gender Bleedthrough in LLMs Project Design.pdf` (phases 0–5, each gated). Background papers: `sources/SUMMARIES.md`. Usage and layout: `README.md`.

## Working with the PI

- The PI is relying on Claude for the technical side and is **writing this up**. Explain each new method as it's reached, in chat *and* in `docs/explainers/NN-<phase>.md` (what it measures, why it's built that way, how to read the numbers, limits worth stating in a write-up). Add new terms to `docs/glossary.md`, marking which tool each belongs to.
- Log every choice that shapes results in `docs/decisions.md` (date, decision, why, alternatives). Flag tentative ones as tentative.
- The PI **doesn't read German**. Never rely on them to check German content. Anything they need to review must be checkable in English (show glosses).
- Preference: **drop anything that would need a manual check** rather than keep it and ask for review. Small, clean, automatically filtered sets beat large ones needing review. Hand-picked items (classics) are the exception, and they still get automatic checks.
- Report findings that cut against what you just built (e.g. a measure that turns out noisy).

## Hard rules

- **Gold linguistic labels never come from a language model**, including Claude. Genders come from Wiktionary (kaikki.org dumps) via `gbleed lexicon`. Hand-written labels must be marked unverified (`source = claude-unverified`).
- **Don't tune measurement choices on a model's scores** (frames, thresholds, prompts). Fix them on linguistic/grammatical grounds, then *check* on at least two model families (currently Qwen3 + EuroLLM). If a check reveals a flaw, fix the flaw on principle and say so.
- **One device and precision per model** for activations; never compare activations across hardware. Final cross-size runs all go on RunPod (CUDA, bf16); the Mac (M1, 16 GB, MPS, fp16) is for development.
- Base (pretrained) models only; no 4-bit/8-bit quantization.
- Stimulus lists are versioned (`data/stimuli/phase0_vN.csv`). Don't edit a list that has results; sample a new version and rerun.
- Cross-model comparisons use the **shared set** (items known by every model compared).
- **Held-out test sets are frozen before training and never trained on.** Every stimulus file for
  Phases 1–5 carries a `split` column (train/test), assigned once with a fixed seed, by stem.
  Test sets: Phase 1 nonce stems; Phase 2 exceptions (-ma, clippings, día/mano), homographs,
  el/la mar; Phase 3 suffix-marked nouns and der/die See; Phases 4–5 flipped pairs + classics.
  Training code must assert no test item appears in its training data. Candidate extra test:
  feminine *el agua*-type nouns (currently excluded) for Phase 2.

## Commands

```sh
uv sync; uv run pytest; uv run ruff check . && uv run ruff format .
uv run gbleed lexicon                 # rebuild lexicons from data/raw/kaikki (~40 s)
uv run gbleed classics                # Wiktionary-check data/stimuli/classics_spec.csv
uv run gbleed sample-phase0 --out data/stimuli/phase0_vN.csv
uv run gbleed phase0 MODEL [MODEL...] --stimuli data/stimuli/phase0_vN.csv
uv run gbleed phase0-compare [--by freq_bin|set|de_suffix|es_exception]
```

- Model runs and downloads take minutes: run them in the background, writing to a log.
- Commit before runs so `git_commit` in result metadata is meaningful (`git_dirty` ignores `results/`).
- Ruff config is strict and `ruff format` rewrites code. Re-read a file before scripted string-replacement edits, and check that the edit actually applied (it has failed silently in this repo before).

## Current state (update as it changes)

- Phase 0 done: two sentence frames, margin = log P(noun + rest | article), "known" = both frames ≥ 1 nat. Details: `docs/explainers/00-phase0-behavioural-check.md`.
- Model family: **Qwen3 base**, 0.6B/1.7B/4B (Mac) → 0.6B/1.7B/4B/8B/14B (RunPod). No 32B base; 30B-A3B is MoE (avoid). Repeat the full 0.6B/1.7B/4B sequence on RunPod.
- Phase 0 on v3 (de/es known): 0.6B 89/89%, 1.7B 90/92%, 4B 94/94%, EuroLLM-1.7B 97/98%. Shared set across the three Qwen sizes: 204 de / 207 es of 257; 0.6B is the bottleneck (open question: keep it once 8B/14B exist?). Optional later: EuroLLM-1.7B as a cross-family replication of Phases 2–3.
- Phase 1 (0.6B/1.7B/4B): outcome (b); linear erasure of -a/-o generalises poorly to new words.
- Phase 2 (1.7B/4B, fixed inputs): primary (adjusted, ending-matched) inconclusive; rank-2-erased regular-noun direction tracks gender on masculine exceptions (all 4B layers) and homographs (most 4B layers); feminine -o exceptions look masculine. Explainer 02. Readout position: LAST stays primary (gender-blind check); at AFTER spelling mostly vanishes and the 4B erased-direction result does not hold (1.7B does). Confirmatory readout amendment is TENTATIVE pending PI sign-off (decisions.md).
- Inputs: `<|endoftext|>` + newline + word (the first token after the separator can be a sink); extraction warns on outlier norms.
- Phases 3–5 plan: `docs/design/phases-3-5-plan.md` (agreement-flip gate first; dose-response + KL damage; epicenes; erasure arm). Markedness hypothesis pre-registered in decisions.md.
- RunPod: `runpod/` scripts (trial done on A40; see decisions.md). Global volume is GeeseFS object storage: code/venv/model cache on local disk.

## Gotchas learned

- **Never test an eraser with a probe trained on the eraser's fit data**: chance by construction (LEACE equalises class means → zero optimal weights). Erase-all-then-cross-validate goes *below* chance. Report AUC, not just accuracy.
- Position 0 is an attention sink (~180× norm), and so can be the first token after `<|endoftext|>` (single-token words got one dimension ~2,500): prefix words with `<|endoftext|>` + newline.

- Qwen3 adds no BOS token: never put the scored word or article first in a sentence.
- German *die* is also plural; *ein*/*dem* also serve neuter (so their priors favour masculine). Spanish *a el* → *al*, *de el* → *del*; feminine stressed-a nouns take *el* (*el agua*).
- `wordfreq` lowercases, so *Aber* gets *aber*'s frequency. Homographs corrupt frequencies.
- fp16 on MPS: log-prob differences below ~0.05 nat are rounding noise; in bf16 (RunPod) noise is larger, a few tenths of a nat. Gemma breaks in fp16.
- English-gloss matching of German–Spanish pairs is ~10% wrong even under strict rules (*Platte/apartamento* "flat"). Review final Phase 4–5 pairs in English before use.
