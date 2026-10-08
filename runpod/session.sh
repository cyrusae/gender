#!/usr/bin/env bash
# Confirmatory session (plan: docs/design/runpod-session-plan.md). GPU steps only: every model on
# the same GPU type, bf16; analysis happens on the Mac (a worktree at BUNDLE_COMMIT.txt).
# Run from the repo root after setup.sh:  nohup bash runpod/session.sh > /dev/null 2>&1 < /dev/null &
# Watch:  tail -f logs/session.log      Resume after a failure: MODELS="..." bash runpod/session.sh
# Local smoke test: KEEP_MODELS=1 HF_HOME=~/.cache/huggingface OUT=/tmp/x MODELS=Qwen/Qwen3-0.6B-Base bash runpod/session.sh
set -euo pipefail
export HF_HOME=${HF_HOME:-/root/hf}   # local disk (the global volume is object storage)
export PATH="$HOME/.local/bin:$PATH"
# Confirmatory 8B/14B before the exploratory MoE (30B-A3B), so a time stop only costs exploratory data.
MODELS=${MODELS:-"Qwen/Qwen3-0.6B-Base Qwen/Qwen3-1.7B-Base Qwen/Qwen3-4B-Base Qwen/Qwen3-8B-Base Qwen/Qwen3-14B-Base Qwen/Qwen3-30B-A3B-Base"}
MAX_HOURS=${MAX_HOURS:-4.5}  # don't start a new model after this much elapsed time (budget guard)
START=$(date +%s)
OUT=${OUT:-/root/out}  # pod-local; the Mac downloads each model archive as soon as it appears
mkdir -p logs "$OUT"
step() {  # step NAME CMD...: timed, logged, stops the session on failure
  local name=$1; shift
  local t=$(date +%s)
  echo "=== $(date +%T) start: $name"
  "$@"
  echo "=== $(date +%T) done:  $name ($(( $(date +%s) - t )) s)"
}
{
  echo "=== $(date) session start on $(uv run python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")'), commit $(cat BUNDLE_COMMIT.txt)"
  for M in $MODELS; do
    if [ "$(( $(date +%s) - START ))" -gt "$(python3 -c "print(int($MAX_HOURS*3600))")" ]; then
      echo "=== $(date +%T) MAX_HOURS ($MAX_HOURS h) reached: not starting $M"; break
    fi
    S=$(echo "$M" | tr '/' '_' | sed 's/_/__/')
    step "$M download" uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('$M')"
    # behavioural checks (frozen lists; nothing is re-selected)
    step "$M phase0 v3"          uv run gbleed phase0 "$M" --stimuli data/stimuli/phase0_v3.csv
    step "$M multi-check"        uv run gbleed multi-check "$M"
    step "$M known es (v4 pool)" uv run gbleed phase0 "$M" --stimuli data/stimuli/phase2_pool_v4.csv --out results/phase2_known
    step "$M known de (3-way)"   uv run gbleed phase3-known "$M"
    step "$M verb-frame check"   uv run gbleed frame-check "$M"
    # activations (LAST and AFTER)
    step "$M readout extract"    uv run gbleed readout "$M" --extract-only
    step "$M phase1 extract"     uv run gbleed phase1 "$M" --extract-only
    step "$M phase2 extract last"  uv run gbleed phase2 "$M" --extract-only
    step "$M phase2 extract after" uv run gbleed phase2 "$M" --position after --extract-only
    step "$M phase3 extract"     uv run gbleed phase3 "$M" --extract-only
    if [ "$M" = "Qwen/Qwen3-4B-Base" ]; then  # pre-registered precision check
      step "$M phase0 v3 fp32" uv run gbleed phase0 "$M" --dtype float32 --stimuli data/stimuli/phase0_v3.csv --out results/phase0_fp32
    fi
    # uncompressed: float activations barely compress, and gzip cost ~150 s even for 0.6B
    step "$M save archive" tar cf "$OUT/$S.tar.part" activations/"$S" BUNDLE_COMMIT.txt
    mv "$OUT/$S.tar.part" "$OUT/$S.tar"  # complete archives only appear under their final name
    if [ -z "${KEEP_MODELS:-}" ]; then  # free local disk (KEEP_MODELS=1 for local smoke tests)
      rm -rf activations/"$S" "$HF_HOME"/hub/models--"$(echo "$M" | tr '/' '-' | sed 's/-/--/')"
    fi
  done
  step "results archive" tar czf "$OUT/results.tgz.part" results logs BUNDLE_COMMIT.txt
  mv "$OUT/results.tgz.part" "$OUT/results.tgz"
  echo "=== $(date) session done"
} 2>&1 | tee -a logs/session.log
