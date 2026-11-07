# Handoff: picking the project back up (written 2026-10-29, updated 2026-11-07)

For a fresh Claude session (after `/clear` or a restart) or for the PI. Read this, then
`CLAUDE.md`, then the newest rows of `docs/decisions.md`. Verify anything here against the files:
this is a snapshot, and the repo is the record.

## 1. State at handoff

- **Phase 1 (base-model study) is closed** (2026-10-26). Analysis cutoff for the write-up:
  2026-10-27 (decisions row). Phase 2 = new languages + instruct replication.
- **Nothing is running.** No pods. RunPod credit ≈ $11.70 at last check (confirm in console).
- **Experiments are paused** until the PI's usage resets (PI, 2026-10-29). Housekeeping only.
- Last commits: `b48e782` (quiet progress output, `GBLEED_QUIET=1`), `ba25ae5` (CLAUDE.md token
  hygiene). Tests: 131 pass.

## 2. Waiting on the PI

1. **Instruct stage A on the other three siblings**: Gemma-3-12B-it, Qwen3-8B, Qwen3-14B
   (`docs/design/instruct-replication-prereg.md`). ~$1 each on an L40S pod (`runpod/instruct_stageA.sh`)
   plus a few hours of Mac analysis each (`scripts/instruct_stageA_analysis.py MODEL FORMAT JOB`,
   30 jobs per model via `scripts/queue.sh`). The 4B-it pilot is done and recorded in the prereg.
2. Stage B (steering) for instruct models: decided later by the prereg's rule; not started.

## 3. Documentation updates

**Done 2026-11-07** (PI away; commits 2689b02 … 455730b): CLAUDE.md state (character branch and
worktree, stage A pilot); project outline §18–21 plus a Nov 7 status; glossary (outlier
dimension, estimator limit, partial Spearman, known-check, matched pairs, AoA, SAE, size control,
chat formats, boundary merge); explainers 08 (Gemma + outlier dimension), 09 (rarity held-out +
Russian), 10 (instruct stage A); READMEs and docs index; `writeups/supplement-table.md` (compiled
from result files; Gemma fixed T2 counts 31/33 and (12B, valid layers) 42/44, corrected from 45/47 after the Feynman draft's check); a first draft of abstract,
introduction and discussion (`writeups/2026-11-07-abstract-intro-discussion-draft.md`);
methods-summary lines.

**Still open:**
1. PI review of the abstract/intro/discussion draft (framing is the PI's call).
2. Citations still unverified: Elpers Exp 2 n, Samuel 2019 primary figures; no references yet for
   cross-fitting estimators, LEACE limits, anisotropy, outlier dimensions (find and verify before
   citing).
3. The paper-style draft (`2026-10-23-methods-paper-style.md`) has no abstract/intro/discussion of
   its own; decide whether it gets the same ones or is merged.
4. Instruct results stay out of the phase-1 write-up unless the PI says otherwise.

## 4. Where things live

- Code `src/gbleed/`, CLI `gbleed`, run Python with `uv run`. Preregs `docs/design/`; decisions
  `docs/decisions.md` (table; `merge=union`); results `results/<study>/`.
- Bulky data on `/Volumes/Lexar/gbleed` (must be mounted). Qwen 8B/14B activations:
  `GBLEED_ACT_DIR=/Volumes/Lexar/gbleed/gender-s2/activations`. Gemma: `gemma-pod*/extract/`.
  Instruct stage A: `instruct-stageA/`. Rarity: `rarity-pod/`. Old worktrees: `archive/worktrees/`.
- Only remaining worktree: `worktrees/gender-character` (character study session's, branch `character`).
- HF token: `~/.config/huggingface/gender-token`, passed into the pod process env only.
- EDA exports: `eda/` (CC0-safe, no words) and `results/word_level/` (words + Glasgow, not CC0).

## 5. Habits that prevent known mistakes

- **Stage files by name**; never `git commit -a` / `add -A`. The PI keeps uncommitted edits
  (notably `sources/SUMMARIES.md`) that have been swept into commits three times.
- Pods: create from `image` directly with `mounts: {}`, avoid US-MO-1, quote the hourly price
  first, readiness = command output from inside the container, detect the end by an anchored log
  marker, verify checksums before terminating, keep the session attended while billing.
- Check in with a plan before building or running new work (memory: check-in-before-building).
- Monitors expire after 30 min and don't survive restarts: re-arm, and after any restart check
  for orphaned jobs (`ps`, log end markers) before starting anything.
- Token hygiene: see `CLAUDE.md`.
