# RunPod confirmatory session: plan (adopted 2026-10-08, PI decisions)

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
- **Volume**: global volume `whispering_crimson_ermine` at `/workspace` (outputs only).
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
- Unpack into a **separate git worktree at the bundled commit**, not the main checkout, so pod
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

- PI: create the pod in the console (the global volume isn't visible to the API), with the SSH
  key already registered; pass `HF_TOKEN` as a pod secret if needed (Qwen isn't gated); terminate
  the pod at the end.
- Claude: bundle the committed code, copy it, run setup and the session, watch progress, save to
  the volume, download, and run the analyses.

## Not in this session

- Qwen's official residual-stream SAEs exist for Qwen3-1.7B-Base, 8B-Base and 30B-A3B-Base
  (`Qwen/SAE-Res-Qwen3-*`, found 2026-10-08): an SAE follow-up within the same family is possible
  (previously assumed to need Gemma Scope). Later decision.
