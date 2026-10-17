#!/usr/bin/env bash
# P30 (docs/design/p30-object-context-prereg.md): unsteered object/person-context scoring on 8B/14B.
# Run from the repo root after setup.sh:  nohup bash runpod/p30.sh > /dev/null 2>&1 < /dev/null &
# End markers (wait on these, never on pgrep): "=== P30 done" / "P30 FAILED" in logs/p30.log.
# No pod-side self-stop: the pod's injected RUNPOD_API_KEY is rejected by runpodctl (tested
# 2026-10-17, "Unauthorized"), so stopping the pod is the Mac side's job (watch the markers).
set -euo pipefail
export HF_HOME=${HF_HOME:-/root/hf}
export GBLEED_STORE=${GBLEED_STORE:-local}  # no external drive on the pod: data stays in the repo (gbleed/paths.py)
export PATH="$HOME/.local/bin:$PATH"
MODELS=${MODELS:-"Qwen/Qwen3-8B-Base Qwen/Qwen3-14B-Base"}
mkdir -p logs
{ (
  set -eE  # a failing step ends the run with a FAILED marker, so the watcher can't wait forever
  trap 'echo "=== $(date +%T) P30 FAILED at line $LINENO"' ERR
  echo "=== $(date) P30 start on $(uv run python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")'), commit $(cat BUNDLE_COMMIT.txt)"
  for M in $MODELS; do
    t=$(date +%s)
    echo "=== $(date +%T) start: $M"
    uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('$M')"
    uv run gbleed p5 p30 "$M"
    uv run python -c "from huggingface_hub import scan_cache_dir as s; c=s(); [c.delete_revisions(*[r.commit_hash for r in repo.revisions]).execute() for repo in c.repos if repo.repo_id=='$M']"
    echo "=== $(date +%T) done: $M ($(( $(date +%s) - t )) s)"
  done
  echo "=== $(date) P30 done"
) } 2>&1 | tee -a logs/p30.log
