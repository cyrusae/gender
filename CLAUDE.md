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
- Phase 2 (1.7B/4B, fixed inputs): primary (adjusted, ending-matched) inconclusive; rank-2-erased regular-noun direction tracks gender on masculine exceptions (all 4B layers) and homographs (most 4B layers); feminine -o exceptions look masculine. Explainer 02. Readout position: LAST stays primary (gender-blind check); at AFTER spelling mostly vanishes and the 4B erased-direction result does not hold (1.7B does). Confirmatory runs: LAST primary, AFTER pre-registered secondary; a gender result is "position-robust" only if it holds at both (adopted, decisions.md). Phase 2 stimuli are v3 (homograph verb frame `siempre {w}`, behaviourally checked on Qwen3 + EuroLLM-1.7B; EuroLLM-1.7B is back on disk).
- Inputs: `<|endoftext|>` + newline + word (the first token after the separator can be a sink); extraction warns on outlier norms.
- Phase 2 v4 + Phase 3 (exploratory, 1.7B/4B): Spanish stratified primary reads gender (22/27, 31/35 layers at LAST; not at AFTER); German compounds follow the head (all layers at LAST); markedness not supported (the plain length estimator would have falsely confirmed it). Explainers 02 (v4 section) and 03. Shared estimators: `estimators.py`. Next: RunPod confirmatory 8B/14B.
- Phases 3–5 plan: `docs/design/phases-3-5-plan.md`. **Adopted designs (2026-10-08):** `docs/design/phase4-design.md` (Q1–Q6), `phase5-design.md` (P1–P15), `suffix-followup-prereg.md` (D1–D4). Markedness hypothesis pre-registered in decisions.md.
- RunPod confirmatory session done 2026-10-08 (A100, six models, ~$1.77). **Pod-data analysis runs in a separate worktree `../gender-pod` (branch `pod-analysis`)**; job queue + logs in the session scratchpad; results go to `results/runpod/` when done. First confirmatory: 8B Phase 2 at LAST gender 28/35 layers (AFTER, 14B pending); Phase 3 reruns pending (bootstrap fix). Faster Newton solver on branch `batched-solver` (`../gender-solver`) for Phases 4–5. Next pod round (Phases 4–5 + suffix_ctrl extraction): 8B/14B.
- Word-list screening, every filter: `docs/screening/`. Flipped pairs: 389 usable (63 English cognates kept, sensitivity without) (`docs/reports/flipped-pairs-flags.csv`). Russian *-ь* pool (245) on branch `russian` (`../gender-ru`); needs Russian known-check frames.
- RunPod: `runpod/` scripts. Global volume is GeeseFS object storage: code/venv/model cache on local disk.

## Gotchas learned

- **Bootstrap + split-half: keep copies of a resampled item in one half** (else halves share noise; `items=` in `split_half_geometry`).
- **Held-out rules need code, not prose**: every builder asserts against every earlier training set (the Phase 2 builder didn't: 37 flipped pairs leaked).
- **Don't edit a job list a running bash `while read` loop is reading** (it buffers); start a new runner instead.

- **Lengths and cosines of mean differences are noise-biased** (smaller group looks longer; shared reference pushes cosines up): use split-half estimates + a within-cell shuffle null (`estimators.py`).
- **Two models on the Mac's GPU at once slow down ~30×**: run GPU steps one at a time; CPU analyses can run in parallel (set `OMP_NUM_THREADS=1` etc.).

- **Never test an eraser with a probe trained on the eraser's fit data**: chance by construction (LEACE equalises class means → zero optimal weights). Erase-all-then-cross-validate goes *below* chance. Report AUC, not just accuracy.
- Position 0 is an attention sink (~180× norm), and so can be the first token after `<|endoftext|>` (single-token words got one dimension ~2,500): prefix words with `<|endoftext|>` + newline.

- Qwen3 adds no BOS token: never put the scored word or article first in a sentence.
- German *die* is also plural; *ein*/*dem* also serve neuter (so their priors favour masculine). Spanish *a el* → *al*, *de el* → *del*; feminine stressed-a nouns take *el* (*el agua*).
- `wordfreq` lowercases, so *Aber* gets *aber*'s frequency. Homographs corrupt frequencies.
- fp16 on MPS: log-prob differences below ~0.05 nat are rounding noise; in bf16 (RunPod) noise is larger, a few tenths of a nat. Gemma breaks in fp16.
- English-gloss matching of German–Spanish pairs is ~10% wrong even under strict rules (*Platte/apartamento* "flat"). Review final Phase 4–5 pairs in English before use.
