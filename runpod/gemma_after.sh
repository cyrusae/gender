#!/usr/bin/env bash
# Gemma-12B Phase 4 AFTER check: Spanish phase2_bare_after (CUDA bf16, as the 12B pod run). Needs HF_TOKEN in the environment.
# Run from the repo root after setup.sh:  nohup bash runpod/gemma_after.sh > /dev/null 2>&1 < /dev/null &
# End markers (wait on these, never on pgrep): "=== GAFTER done" / "GAFTER FAILED" in logs/gemma_after.log.
# No pod-side self-stop: the pod's injected RUNPOD_API_KEY is rejected by runpodctl (tested
# 2026-10-17, "Unauthorized"), so stopping the pod is the Mac side's job (watch the markers).
set -euo pipefail
export HF_HOME=${HF_HOME:-/root/hf}
export GBLEED_STORE=${GBLEED_STORE:-local}  # no external drive on the pod: data stays in the repo (gbleed/paths.py)
export PATH="$HOME/.local/bin:$PATH"
MODELS=${MODELS:-"google/gemma-3-12b-pt"}
export GBLEED_DTYPE=bfloat16
: "${HF_TOKEN:?HF_TOKEN required}"
mkdir -p logs
{ (
  set -eE  # a failing step ends the run with a FAILED marker, so the watcher can't wait forever
  trap 'echo "=== $(date +%T) GAFTER FAILED at line $LINENO"' ERR
  echo "=== $(date) GAFTER start on $(uv run python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")'), commit $(cat BUNDLE_COMMIT.txt)"
  for M in $MODELS; do
    t=$(date +%s)
    echo "=== $(date +%T) start: $M"
    uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('$M')"
    uv run gbleed phase2 "$M" --extract-only --position after
    uv run python -c "from huggingface_hub import scan_cache_dir as s; c=s(); [c.delete_revisions(*[r.commit_hash for r in repo.revisions]).execute() for repo in c.repos if repo.repo_id=='$M']"
    echo "=== $(date +%T) done: $M ($(( $(date +%s) - t )) s)"
  done
  mkdir -p /root/out && tar czf /root/out/gemma_after.tgz activations/*/phase2_bare_after.* logs BUNDLE_COMMIT.txt
  echo "=== $(date) GAFTER done"
) } 2>&1 | tee -a logs/gemma_after.log
