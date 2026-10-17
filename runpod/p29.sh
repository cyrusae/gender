#!/usr/bin/env bash
# P29 (docs/design/pooling-neutrality-prereg.md): unsteered agreement-neutrality scoring on 8B/14B.
# Run from the repo root after setup.sh:  nohup bash runpod/p29.sh > /dev/null 2>&1 < /dev/null &
# End markers (wait on these, never on pgrep): "=== P29 done" / "P29 FAILED" / "WATCHDOG" in logs/p29.log.
# Backstop: after MAX_MINUTES the pod stops itself (runpodctl, if the pod has it), finished or not.
set -euo pipefail
export HF_HOME=${HF_HOME:-/root/hf}
export GBLEED_STORE=${GBLEED_STORE:-local}  # no external drive on the pod: data stays in the repo (gbleed/paths.py)
export PATH="$HOME/.local/bin:$PATH"
MODELS=${MODELS:-"Qwen/Qwen3-8B-Base Qwen/Qwen3-14B-Base"}
MAX_MINUTES=${MAX_MINUTES:-120}
mkdir -p logs
(
  sleep $((MAX_MINUTES * 60))
  echo "=== $(date +%T) WATCHDOG: ${MAX_MINUTES} min reached, stopping the pod" >> logs/p29.log
  if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
    runpodctl stop pod "$RUNPOD_POD_ID" >> logs/p29.log 2>&1
  else
    echo "=== WATCHDOG: no runpodctl / RUNPOD_POD_ID: cannot stop" >> logs/p29.log
  fi
) &
{ (
  set -eE  # a failing step ends the run with a FAILED marker, so the watcher can't wait forever
  trap 'echo "=== $(date +%T) P29 FAILED at line $LINENO"' ERR
  echo "=== $(date) P29 start on $(uv run python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")'), commit $(cat BUNDLE_COMMIT.txt)"
  echo "watchdog: ${MAX_MINUTES} min; runpodctl: $(command -v runpodctl || echo none); pod id: ${RUNPOD_POD_ID:-none}"
  for M in $MODELS; do
    t=$(date +%s)
    echo "=== $(date +%T) start: $M"
    uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('$M')"
    uv run gbleed p5 p29 "$M"
    uv run python -c "from huggingface_hub import scan_cache_dir as s; c=s(); [c.delete_revisions(*[r.commit_hash for r in repo.revisions]).execute() for repo in c.repos if repo.repo_id=='$M']"
    echo "=== $(date +%T) done: $M ($(( $(date +%s) - t )) s)"
  done
  echo "=== $(date) P29 done"
) } 2>&1 | tee -a logs/p29.log
