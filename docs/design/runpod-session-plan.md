# RunPod sessions: plans and outcomes

*Session 1 (confirmatory Phases 0–3): **done** 2026-10-08, outcome below the plan. Session 2
(Phases 4–5 + suffix follow-up): **plan** at the end of this file.*

---

# Session 1: plan (adopted 2026-10-08, PI decisions)

*Script: `runpod/session.sh`. Background: `runpod/README.md` (trial lessons), `docs/decisions.md`.*

## What the session is for

The confirmatory runs of everything pre-registered so far: Phase 2 (v4, stratified primary),
Phase 3, both readout positions, on **Qwen3-8B and 14B**. The 0.6B/1.7B/4B sequence is repeated on
the same hardware, so every size is compared on one GPU type and one precision (bf16), never
across hardware. **Qwen3-30B-A3B-Base** (mixture of experts: 30.5B parameters, ~3B active per
token; 48 layers, width 2,048) is added as an **exploratory** architecture data point, not a rung
of the dense size ladder (Qwen3-32B exists only post-trained). It needs ~61 GB, which is why the
session moved from the A40 to the A100; one GPU type for the whole Qwen session (also the Gemma
plan's GPU). Only GPU work happens on the pod (behavioural checks and activation extraction);
all analysis runs afterwards on the Mac.

## Hardware and cost

- **1 × A100 80 GB (SXM or PCIe), Secure Cloud, $1.59/h** (live price 2026-10-08). Same GPU type
  for all six models.
- **Budget: $9 of credits = absolute hard stop (~5.6 h).** `session.sh` won't start a new model
  after `MAX_HOURS=4.5` (~$7.15); the confirmatory models run before the exploratory one.
- **Template**: Runpod PyTorch 2.8.0, host CUDA ≥ 13.0. **Container disk 150 GB** (one model at
  a time: 30B-A3B weights ~61 GB + activations ~4 GB + environment ~8 GB; each model's weights
  and activations are deleted from local disk once saved to the volume).
- **No volume** (the global volume isn't visible to the API, and Claude creates the pod): outputs
  stay on the pod's disk and the Mac downloads each model's archive as soon as it's complete, so a
  stop when credits run out loses at most the model in progress.
- **Estimated ~2.5–3.5 h → ~$4–5.50.** Dense ladder ~1.5–2 h (A100 memory bandwidth ~2.5× the
  A40's), 30B-A3B ~0.75–1.25 h (61 GB download; the MoE layers are slow in Hugging Face code),
  ~15 min setup and transfer.

## Steps (per model, in `session.sh`)

1. download weights to local disk
2. Phase 0 (v3 list), multi-gender acceptance (needed by the Phase 2 analysis)
3. known checks on the frozen lists: Spanish v4 pool (Phase 0 frames), German three-way frames
4. verb-frame behavioural check (`siempre ___`)
5. activations: readout check (LAST + AFTER), Phase 1 (LAST), Phase 2 (LAST, AFTER, with Phase 1
   sets at AFTER), Phase 3 (both positions)
6. 4B only: Phase 0 in fp32 (the pre-registered bf16 vs fp32 check)
7. save that model's activations to the volume, delete them and the weights locally

Then all behavioural results go to the volume as one archive.

## Afterwards, on the Mac

- Download the archives (~23 GB of activations in total; ~1–2 GB per small model, ~6–9 GB for
  8B/14B).
- Unpack into a **separate git worktree at the analysis commit** (the bundled commit, or a later one
  that changes only analysis code: the initial-*a* amendment), not the main checkout, so pod
  activations never mix with Mac activations (0.6B–4B exist on both) and the analysis runs with
  exactly the code that extracted them. Copy in the gitignored lexicon tables
  (`data/lexicon/*_nouns.csv`; the analyses read glosses from them).
- Run the Phase 2 and Phase 3 analyses there (CPU, in parallel, one thread each), plus the readout
  analysis. Estimated several hours for 8B/14B (probe refits scale with width); overnight is fine.
- Copy result tables into the main repo under `results/runpod/`, log in `decisions.md`, update the
  explainers.

## Decisions for the PI

1. **Frozen word lists for 8B/14B** (recommended). The lists were selected with Mac 1.7B + 4B
   known checks; the larger models are expected to know at least as many. Analyse the frozen
   lists as pre-registered, report each model's known rate on them, and add a sensitivity analysis
   that drops nouns that model doesn't know. Alternative: re-select per model (changes the item
   sets between models; breaks the shared-set rule).
2. **Repeat 0.6B/1.7B/4B on the pod**: same-hardware comparison across all sizes, and a
   Mac-vs-pod replication of the exploratory results.
3. **A100 Secure** (PI): 30B-A3B added as exploratory; $9 hard stop.
4. **Suffix follow-up** (exploratory, leave-one-suffix-out): run on the Mac first; not part of
   this session unless it's pre-registered before the pod starts.

## Who does what

- Claude (PI decision 2026-10-08): create the pod via the API, bundle and copy the code, run
  setup and the session, download archives as they complete, **terminate the pod as soon as the
  downloads are verified**, then run the analyses. Spend stops at the $9 credit balance (no
  auto-reload); a little over is acceptable, but no pre-paying.

## Not in this session

- Qwen's official residual-stream SAEs exist for Qwen3-1.7B-Base, 8B-Base and 30B-A3B-Base
  (`Qwen/SAE-Res-Qwen3-*`, found 2026-10-08): an SAE follow-up within the same family is possible
  (previously assumed to need Gemma Scope). Later decision.

---

# Session 1: outcome (2026-10-08)

**Cost: $1.80 billed** ($1.78 GPU + $0.02 disk; pod uptime 4,007 s = 1.11 h at $1.59/h), against
the plan's estimate of $4–5.50. Remaining credit ≈ $7.20 of the $9.

**What ran:** all six models (0.6B, 1.7B, 4B, 8B, 14B, 30B-A3B), every step, no failures. Dense
ladder (0.6B–14B) ~22 min of steps in total. All archives downloaded and verified (size +
listing) before the pod was terminated; later re-verified file by file (SHA-256) after unpacking,
and the archives deleted (manifest: `~/GitHere/runpod-2026-10-08/`).

**What went differently from the plan:**
- **30B-A3B load took ~25 min** (CPU-bound conversion of the mixture-of-experts weights in
  transformers 5), and `session.sh` reloaded the model for every step. Fixed mid-session with
  `runpod/one_process.py` (one load, all steps in one process).
- **Disk filled** because the Hugging Face cache now stores weights in a shared `blobs/` folder,
  so deleting a model's folder freed nothing. Fixed by deleting through `huggingface_hub`'s
  cache API (`scan_cache_dir().delete_revisions`).
- **bf16 vs fp32 precision check: 98.6% same status**, below the pre-registered 99% bar (7
  near-threshold flips, no bias). Pre-registered fallback for session 2: fp32 output layer for
  behavioural checks.
- **The Mac analysis was the real bottleneck**, not the pod: the first estimate was ~60 h of CPU.
  Speedups (row-space probe fits, exact rank AUC, batched split-half geometry) brought it to
  roughly a day; two bugs were found on the way (AFTER-layer-0 zero-variance crash; bootstrap
  copies split across halves), both fixed and logged.

**Results (confirmatory, LAST):** Spanish stratified primary reads gender in 28/35 (8B) and
26/39 (14B) layers; German compounds follow their head in 35/35 and 39/39; markedness not
supported. At AFTER the Spanish result vanishes (8B: 3/35). Details: `docs/decisions.md`.

**Lessons for session 2:**
1. Run each model's steps in **one process** (`one_process.py` pattern) from the start.
2. Delete weights through the cache API, never by folder.
3. Budget the **Mac analysis time** as carefully as the pod time; measure throughput first.
4. Pod time was cheap relative to the plan; the estimate was conservative (A100 + extraction-only
   is fast).

---

# Session 2: plan (draft 2026-10-08; not yet adopted)

## What it's for

Confirmatory runs for everything designed and adopted on 2026-10-08, on **Qwen3-8B and 14B**:
- **Phase 4** (`docs/design/phase4-design.md`): activations for the 389 flipped pairs (both
  languages), fresh held-out German/Spanish draws, and the Russian *-ь* list (174 nouns), plus
  the Russian known check on 8B/14B.
- **Phase 5** (`docs/design/phase5-design.md`, P1–P17): the agreement gate, baseline, dose-response
  with damage, specificity, social-gender steering, the English-only readouts, the nonce sets
  (R-NONCE 600 words, English-style 100), noun-only and every-position steering.
- **Suffix follow-up** (`docs/design/suffix-followup-prereg.md`): `suffix_ctrl` known check +
  extraction (83 nouns), for the confirmatory S1.
- **Behavioural checks** in the pre-registered fallback precision (fp32 output layer), including
  a **rerun of session 1's known checks for 8B/14B** (minutes). Only behavioural statuses near the
  1-nat cutoff can change; activations (read before the output layer), the frozen word lists and
  session 1's confirmatory results don't. If statuses shift, the per-model sensitivity analysis
  (drop nouns the model doesn't know) is redone on the Mac with the corrected statuses.

30B-A3B is skipped (P7): its 25-minute load isn't worth it for steering.

## Before renting (prerequisites)

1. **Mac exploratory run first** (1.7B/4B, P8 with EuroLLM for the gate and baseline): it debugs
   the steering code and **measures throughput** per condition, so the pod estimate below becomes
   a real number. Nothing is rented until this runs end to end.
2. Code built and committed: Phase 4 builders (held-out draws, which go to the PI with counts
   first; exclusion of every Phase 2/3 training noun, asserted), Phase 4 analysis, Phase 5
   steering/readout/damage code, a `one_process`-style session driver.
3. Phase 5 frames' behavioural checks done on the Mac (the gate frames, R1/R2 frame checks).
4. The PI reloads credits once the measured estimate is known.

## Hardware: A100 or H100 (decided on the day)

- **Default: A100 80 GB Secure** (~$1.59/h on 2026-10-08), the same chip as session 1.
- **H100** if cheaper *per run*: published bf16 compute ~3× and memory bandwidth 1.7× an A100's;
  real speedups typically 1.5–3×. **Benchmark one fixed batch** (~5 min) at the start and switch
  if the speedup doesn't beat the price ratio. Check live prices that day.
- **Same-chip rule:** whichever chip is used, every comparison within a phase happens on it. The
  session re-extracts the Phase 2/3 training activations and refits the directions on that chip
  (minutes per model), so Phases 4–5 are single-chip; session 1's confirmatory results stay on the
  A100. Bonus check: cosine between session-1 and session-2 directions per model (expected ≈ 1).

## Cost estimate (napkin, before measuring)

| part | A100 time |
|---|---|
| Phase 5, 8B (~500 steered items × ~600 conditions × readouts, single-token adjectives, P16) | ~3–4 h |
| Phase 5, 14B | ~5–7 h |
| Phase 4 + suffix_ctrl extraction, re-extraction of Phase 2/3 training sets, known checks | ~0.5 h |
| setup, downloads, transfer | ~0.5 h |
| **total** | **~9–12 h ≈ $14–19 at $1.59/h** |

Trims if needed: every-position steering at the working dose only (~−25%); one layer instead of
two. The Mac exploratory run replaces these guesses with measured throughput.

## Mechanics (from session 1's lessons)

- Claude creates the pod via the API, bundles the code at a committed analysis commit, runs a
  one-process-per-model driver, downloads each model's outputs as soon as they're complete, and
  **terminates the pod as soon as downloads are verified**.
- Outputs stay on the pod's disk (no volume), downloaded per model, so a stop loses at most the
  model in progress.
- Budget guard as in session 1: no new model starts past a set hour limit; the PI sets the hard
  stop (credit balance; no pre-paying).
- Behavioural results and activations are small this time (steering outputs are scores, not
  activations), except the Phase 2/3 re-extraction (~6–9 GB per model, as session 1).

## Not in session 2

- 30B-A3B (P7); the erasure arm, epicene social-gender directions and the genderless-language
  control (Phase 5 later rounds); the German suffix nonces (decision after the confirmatory S1);
  Gemma (planned separately); SAE analyses (run on the Mac from saved activations).
