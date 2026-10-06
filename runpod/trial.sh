#!/usr/bin/env bash
# Trial run on Qwen3-0.6B: measures timings, and runs the planned bf16-vs-fp32 precision check.
# Run from the repo root after setup.sh. Watch with:  tail -f logs/trial.log
set -euo pipefail
export HF_HOME=${HF_HOME:-/root/hf}   # local disk: the global volume is object storage (slow for many small files)
export PATH="$HOME/.local/bin:$PATH"
M=Qwen/Qwen3-0.6B-Base
mkdir -p logs
{
  echo "=== $(date) trial start on $(uv run python -c 'import torch; print(torch.cuda.get_device_name(0))')"
  t=$(date +%s); uv run gbleed phase0 $M --stimuli data/stimuli/phase0_v3.csv --out results_pod/phase0_bf16
  echo "=== phase0 bf16: $(( $(date +%s) - t )) s"
  t=$(date +%s); uv run gbleed phase0 $M --dtype float32 --stimuli data/stimuli/phase0_v3.csv --out results_pod/phase0_fp32
  echo "=== phase0 fp32: $(( $(date +%s) - t )) s"
  t=$(date +%s); uv run gbleed phase1 $M --extract-only
  echo "=== phase1 extract only: $(( $(date +%s) - t )) s"
  echo "=== $(date) trial done"
} 2>&1 | tee logs/trial.log
tar czf trial_results.tgz results_pod results/phase1 activations logs
echo "Download trial_results.tgz, then stop the pod."
