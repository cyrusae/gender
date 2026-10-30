# Handoff: picking the project back up (written 2026-10-29)

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

## 3. Documentation updates to prioritise (in this order)

1. **`CLAUDE.md` "Current state" is partly stale.** It still says the character study lives in
   `instruct/`, merged with `git merge --no-edit instruct`; the branch was renamed `character`
   (PI m06125) and the hard-rules line already says `character/`. Check the actual directory and
   branch name, then fix. Also add one line: instruct stage A pilot done (4B-it: all Phase 3–4
   verdicts as base; English-instruction formats collapse cross-language measures, exploratory).
2. **`docs/project-outline.md` "Where things stand"**: confirm it covers, in order, Gemma 12B
   AFTER check, rarity held-out, Russian direction (+ size follow-up S), the cosine
   nondeterminism fix (`sorted(set(...))`, c4b0fd1), the internal-review reconciliation (e87906b),
   and the instruct stage A pilot. Add what's missing.
3. **Explainers** (`docs/explainers/00–07`): the late work has no explainer yet, as far as I know.
   Likely needed: one for the Gemma replication + outlier dimension (source:
   `docs/design/gemma-outlier-dimension.md`), one for the rarity held-out and Russian tests, one
   for the instruct replication (chat formats raw / A-en / A-something / A-lang / U; known-check
   flips on rare words). Check the folder before writing.
4. **`docs/glossary.md`**: check for outlier dimension / variance share, validity gate, estimator
   limit ("> 1" clipped labels), partial Spearman, chat formats A/U, end-of-turn position.
5. **Write-up** (`docs/writeups/2026-10-22-methods-results-draft.md` outsider,
   `2026-10-23-methods-paper-style.md`): still missing abstract, introduction, discussion,
   citation verification, and a supplement table (model × phase × N × device × result). Instruct
   results are not in either draft and belong to phase 2, not this write-up, unless the PI says otherwise.

## 4. Where things live

- Code `src/gbleed/`, CLI `gbleed`, run Python with `uv run`. Preregs `docs/design/`; decisions
  `docs/decisions.md` (table; `merge=union`); results `results/<study>/`.
- Bulky data on `/Volumes/Lexar/gbleed` (must be mounted). Qwen 8B/14B activations:
  `GBLEED_ACT_DIR=/Volumes/Lexar/gbleed/gender-s2/activations`. Gemma: `gemma-pod*/extract/`.
  Instruct stage A: `instruct-stageA/`. Rarity: `rarity-pod/`. Old worktrees: `archive/worktrees/`.
- Only remaining worktree: `worktrees/gender-instruct` (character study session's).
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
