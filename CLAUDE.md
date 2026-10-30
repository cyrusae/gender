# CLAUDE.md

Research project: does an LLM's representation of grammatical gender (learned from inanimate nouns) overlap with social gender? Design: `Grammatical Gender Bleedthrough in LLMs Project Design.pdf` (phases 0–5, each gated). Background papers: `sources/SUMMARIES.md`. Usage and layout: `README.md`.

## Working with the PI

- The PI is relying on Claude for the technical side and is **writing this up**. Explain each new method as it's reached, in chat *and* in `docs/explainers/NN-<phase>.md` (what it measures, why it's built that way, how to read the numbers, limits worth stating in a write-up). Add new terms to `docs/glossary.md`, marking which tool each belongs to.
- Log every choice that shapes results in `docs/decisions.md` (date, decision, why, alternatives). Flag tentative ones as tentative.
- The PI **doesn't read German**. Never rely on them to check German content. Anything they need to review must be checkable in English (show glosses).
- Preference: **drop anything that would need a manual check** rather than keep it and ask for review. Small, clean, automatically filtered sets beat large ones needing review. Hand-picked items (classics) are the exception, and they still get automatic checks.
- Report findings that cut against what you just built (e.g. a measure that turns out noisy).
- Keep `docs/project-outline.md` (the chronological "what we've done" outline) current: add to it
  whenever a phase, a cloud session or another major step finishes.
- The PI has ADHD and is tracking many threads: when several are open, give one consolidated list
  (running / waiting on the PI / next) with a recommendation.

## Hard rules

- **Gold linguistic labels never come from a language model**, including Claude. Genders come from Wiktionary (kaikki.org dumps) via `gbleed lexicon`. Hand-written labels must be marked unverified (`source = claude-unverified`).
- **Don't tune measurement choices on a model's scores** (frames, thresholds, prompts). Fix them on linguistic/grammatical grounds, then *check* on at least two model families (currently Qwen3 + EuroLLM). If a check reveals a flaw, fix the flaw on principle and say so.
- **One device and precision per model** for activations; never compare activations across hardware. Final cross-size runs all go on RunPod (CUDA, bf16); the Mac (M1, 16 GB, MPS, fp16) is for development.
- Base (pretrained) models only; no 4-bit/8-bit quantization. (Exceptions: the character study in `character/`, branch `character`, scoped rules in `character/CLAUDE.md`; the instruct replication, `docs/design/instruct-replication-prereg.md`.)
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
### Token hygiene (tool output is the main context cost in this repo)

- Scripts write full results to files and print a short summary: verdict, a few key numbers,
  the output path, a shape check (≤ ~10 lines). Query the file for specifics afterwards.
- Filter by default: `| tail -n 20`, `grep -E` for errors/end markers, `pytest -q | tail -1`,
  `2>/dev/null` for known-noise stderr (HF warnings), `ssh -q` (drops the post-quantum banner).
- `ls` is aliased to `eza --color=always` (ANSI codes, which have broken paths): use
  `command ls`, or `find … | wc -l` when a count is enough.
- Progress bars: `gbleed.models.progress` prints plain one-line status at most once a minute in
  logs; `GBLEED_QUIET=1` turns it off. Strip `\r` (`tr '\r' '\n' | tail -2`) when reading logs.
- Small edits with the Edit tool, not inline Python scripts; confirm with `git diff --stat`.
- Don't re-check status on every monitor event; look closer only on finish or failure.

- Ruff config is strict and `ruff format` rewrites code. Re-read a file before scripted string-replacement edits, and check that the edit actually applied (it has failed silently in this repo before).

## Current state (update as it changes)

**Phase 1 (base-model study) is closed (2026-10-26); the write-up is next.** Full history:
`docs/decisions.md` (one row per decision/result), `docs/project-outline.md` ("Where things
stand"), explainers `docs/explainers/00–07`, preregs `docs/design/`. Drafts:
`docs/writeups/2026-10-22-methods-results-draft.md` (outsider) and `2026-10-23-methods-paper-style.md`.

- **Models:** Qwen3 Base 0.6B/1.7B/4B (Mac, fp16) and 8B/14B (pod, bf16); EuroLLM-1.7B (Mac);
  Gemma 3 4B/12B pt (pod, bf16; slim replication, `gemma-slim-prereg.md`). 30B-A3B exploratory only.
- **Headlines:** gender direction tracks grammar not spelling, at the noun's own token; German
  compounds follow the head; no markedness (the plug-in estimator would have falsely confirmed it);
  partial cross-language sharing (ES/DE; Russian one-way, Spanish strongest); Spanish steering
  effect only on Qwen3-8B, "language-specific" by the prereg (not "bleed": German never valid); German Phase 5 never valid; German f−m direction ↔
  rarer words is a Qwen3 training-sample property (absent on Gemma and held-out nouns).
- **Gemma:** one outlier residual dimension (443 / 2339) needs the pre-registered fix for any
  geometry (`gemma-outlier-dimension.md`); Gemma prompts need exactly one `<bos>`.
- **Lists:** frozen versioned files in `data/stimuli/`; known flaws logged, not edited (German
  complex masculines, Spanish months). New languages: `docs/design/new-language-playbook.md`
  (`gbleed lang-audit`).
- **Phase 2 queue (own preregs):** French → Dutch; Russian next steps (*папа*-type, common-gender
  nouns, animacy); **instruct replication** stage A (`instruct-replication-prereg.md`, adopted
  2026-10-26: raw + chat formats A/U, no steering yet; code `chat_format.py`, `gbleed chat-extract`).
- **Instruct character study:** separate, `instruct/` with its own `instruct/CLAUDE.md`; merge
  with `git merge --no-edit instruct` (`docs/decisions.md` uses `merge=union`).
- **Data:** bulky files on `/Volumes/Lexar/gbleed` (exFAT) via `src/gbleed/paths.py`
  (`GBLEED_STORE`, `GBLEED_ACT_DIR`, …). Qwen 8B/14B activations:
  `GBLEED_ACT_DIR=/Volumes/Lexar/gbleed/gender-s2/activations`; Gemma under
  `/Volumes/Lexar/gbleed/gemma-pod*/extract/`. Old worktrees' results archived in
  `archive/worktrees/` there; only `worktrees/gender-instruct` remains.
- **RunPod:** `runpod/` scripts; HF token file `~/.config/huggingface/gender-token` (pass into the
  pod process env only, never print it).

## Gotchas learned

- **Bootstrap + split-half: keep copies of a resampled item in one half** (else halves share noise; `items=` in `split_half_geometry`).
- **Held-out rules need code, not prose**: every builder asserts against every earlier training set (the Phase 2 builder didn't: 37 flipped pairs leaked).
- **Detect a pod run's end by a log marker, never `ssh pod 'pgrep -f NAME'`** (it matches its own remote shell: the pod idled 16.5 h). Keep timed check-ins while a pod bills.
- **Data on the external drive: never hard-code `activations/` or `results/phase5/`**; use `paths.act_dir()` / `paths.p5_dir()` / `paths.kaikki()`. The drive is exFAT: macOS writes a `._<name>` sidecar beside every file (provenance xattr), so wrap directory listings there in `paths.visible(...)`. A "not mounted" warning followed by file-not-found means the drive is unplugged.
- **End markers must be unique strings, matched as whole lines**: a copy loop grepping `P29 done` also matched the Python line `... P29 done: 2400 prompts` and quit early. Match `^=== .* P29 done$`.
- **Don't create pods from the PyTorch template as-is**: `runpod-torch-v280` attaches a 50 GB persistent `/workspace` volume, and on 2026-10-23 two such pods (US-MO-1, EUR-IS-2) hung at "Initializing" (storage "provisioned · usage unavailable", no container, empty logs). Pass `image` directly with only a container disk (`mounts: {}`), and exclude US-MO-1 (a third pod, no volume, hung there too); OC-AU-1 worked twice. Readiness = a command's output from inside the container, not status RUNNING.
- **Match the pod's host CUDA to the pinned stack before creating it**: vLLM 0.27–0.31 require
  torch 2.13, which exists only for CUDA 12.9/13.0, so a host whose driver tops out at 12.8 fails at
  CUDA init (2026-10-26: H100 NVL pod terminated after ~18 min, ~$0.80). Set `gpu.minCudaVersion`
  from the stack, and make the setup's CUDA check fatal (`set -uo` without `-e` let the script
  run on). Check capacity at that CUDA version, not just the GPU type.
- **Remote `pkill -f`/`pgrep -f` over ssh match the ssh command itself** (its `bash -c` line contains the
  pattern), killing the session; and killing the runner leaves the `VLLM::EngineCore` child holding GPU
  memory. Kill by PID, and check `nvidia-smi --query-compute-apps` before restarting.
- **Pod disk: models accumulate.** A 6-model session filled 200 GB (HF xet blobs are sharded dirs).
  Size `disk` to the sum of the weights + 20%, or `rm -rf $HF_HOME/hub` between models.
- **Blackwell + vLLM 0.31.0: gpt-oss-20b fails (FlashInfer kernels unavailable)** even with
  `VLLM_ATTENTION_BACKEND=TRITON_ATTN`; EuroLLM-9B has a 4,096-token context (cap `max_tokens`).
- **A pod can't stop itself**: the pod's injected `RUNPOD_API_KEY` (in `/proc/1/environ`, not in ssh sessions) is rejected by `runpodctl` ("Unauthorized"). Stopping is the Mac side's job; keep this session attended while a pod bills.
- **A local `pgrep -f PATTERN` inside a Monitor script matches the script itself** (its command line contains PATTERN): wait on an output file or log marker instead.
- **German `strat3` simplex filter gap:** compounds with an agent/instrument head (*Wasserkocher*) and numeral nouns (*Vierer*) passed Phase 3's "simplex" rule; `frequency_axis.complexity_rule` catches them. Any new German set should apply it.
- **Don't edit a job list a running bash `while read` loop is reading** (it buffers); start a new runner instead.

- **Lengths and cosines of mean differences are noise-biased** (smaller group looks longer; shared reference pushes cosines up): use split-half estimates + a within-cell shuffle null (`estimators.py`).
- **Mac memory: 8 parallel analysis jobs (4 of them 30B-A3B) + a GPU run filled 23 GB of swap** (thrashing, battery drain on the 18 W charger). Cap parallel jobs by memory, not cores: `scripts/queue.sh JOBS_FILE [SLOTS] [MIN_FREE_PCT]` starts a job only when enough memory is free.
- **Two models on the Mac's GPU at once slow down ~30×**: run GPU steps one at a time; CPU analyses can run in parallel (set `OMP_NUM_THREADS=1` etc.).

- **Never test an eraser with a probe trained on the eraser's fit data**: chance by construction (LEACE equalises class means → zero optimal weights). Erase-all-then-cross-validate goes *below* chance. Report AUC, not just accuracy.
- Position 0 is an attention sink (~180× norm), and so can be the first token after `<|endoftext|>` (single-token words got one dimension ~2,500): prefix words with `<|endoftext|>` + newline.

- Qwen3 adds no BOS token: never put the scored word or article first in a sentence.
- German *die* is also plural; *ein*/*dem* also serve neuter (so their priors favour masculine). Spanish *a el* → *al*, *de el* → *del*; feminine stressed-a nouns take *el* (*el agua*).
- `wordfreq` lowercases, so *Aber* gets *aber*'s frequency. Homographs corrupt frequencies.
- fp16 on MPS: log-prob differences below ~0.05 nat are rounding noise; in bf16 (RunPod) noise is larger, a few tenths of a nat. Gemma breaks in fp16.
- English-gloss matching of German–Spanish pairs is ~10% wrong even under strict rules (*Platte/apartamento* "flat"). Review final Phase 4–5 pairs in English before use.
