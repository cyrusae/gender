#!/usr/bin/env bash
# Instruct replication stage A (docs/design/instruct-replication-prereg.md): gates + raw and chat-format
# extraction, no steering. Analyses run on the Mac. Needs HF_TOKEN (Gemma) in the environment only.
# Run from the repo root after setup.sh:  nohup bash runpod/instruct_stageA.sh > /dev/null 2>&1 < /dev/null &
# End markers: "=== STAGEA all done" / "=== STAGEA FAILED" in logs/stageA.log.
set -Eeuo pipefail
export HF_HOME=${HF_HOME:-/root/hf}
export GBLEED_DTYPE=${GBLEED_DTYPE:-bfloat16}
export GBLEED_STORE=${GBLEED_STORE:-local}
export PATH="$HOME/.local/bin:$PATH"
MODELS=${MODELS:-"google/gemma-3-4b-it"}
OUT=${OUT:-/root/out}
mkdir -p logs "$OUT"
step() { local name=$1; shift; local t=$(date +%s); echo "=== $(date +%T) start: $name"; "$@"; echo "=== $(date +%T) done:  $name ($(( $(date +%s) - t )) s)"; }
fetch() { uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('$1')"; }
{
  trap 'echo "=== STAGEA FAILED at line $LINENO ($(date +%T))"' ERR
  echo "=== $(date) stageA start on $(uv run python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")'), commit $(cat BUNDLE_COMMIT.txt)"
  for M in $MODELS; do
    S=$(echo "$M" | tr '/' '_' | sed 's/_/__/')
    step "$M download"          fetch "$M"
    step "$M phase0 v3"         uv run gbleed phase0 "$M" --stimuli data/stimuli/phase0_v3.csv
    step "$M known"             uv run gbleed gemma-known "$M"
    step "$M known pairs"       uv run gbleed p5 p4-known "$M"
    step "$M phase2 last"       uv run gbleed phase2 "$M" --extract-only
    step "$M phase2 after"      uv run gbleed phase2 "$M" --extract-only --position after
    step "$M phase3 extract"    uv run gbleed phase3 "$M" --extract-only
    step "$M phase4 extract"    uv run gbleed phase4 "$M" --extract-only
    step "$M number extract"    uv run gbleed p5 number-extract "$M"
    step "$M chat extract"      uv run gbleed chat-extract "$M"
    step "$M archive" tar cf "$OUT/$S.tar" activations/"$S" activations_chat results BUNDLE_COMMIT.txt logs
  done
  echo "=== STAGEA all done $(date)"
} 2>&1 | tee -a logs/stageA.log
