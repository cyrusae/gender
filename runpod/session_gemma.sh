#!/usr/bin/env bash
# Gemma slim (prereg: docs/design/gemma-slim-prereg.md): G0 gates, known-check filter, and the GPU
# steps for G1 (R1 Phase 3-5 verdicts, R2 frequency overlap) on gemma-3-4b-pt / gemma-3-12b-pt.
# One chip (A100 80GB), bf16 on CUDA (models.py default). Analyses (R2 A/C/D, G2 SAE) run on the
# Mac from the copied-back activations. 27B is off unless the PI OKs it (MODELS="...").
# Needs HF_TOKEN in the environment (Gemma licence accepted); set at pod time only, never committed.
# Run from the repo root after setup.sh:  nohup bash runpod/session_gemma.sh > /dev/null 2>&1 < /dev/null &
# Watch: tail -f logs/session_gemma.log   End markers: "=== GEMMA all done" / "=== GEMMA FAILED"
# Resume: FROM="<model> <step name>" MODELS="..." bash runpod/session_gemma.sh
set -Eeuo pipefail
export HF_HOME=${HF_HOME:-/root/hf}
export GBLEED_STORE=${GBLEED_STORE:-local}  # no external drive on the pod (gbleed/paths.py)
export PATH="$HOME/.local/bin:$PATH"
MODELS=${MODELS:-"google/gemma-3-4b-pt google/gemma-3-12b-pt"}
MAX_HOURS=${MAX_HOURS:-5}  # budget guard: don't start a new model after this
START=$(date +%s)
OUT=${OUT:-/root/out}
mkdir -p logs "$OUT"
step() {
  local name=$1; shift
  if [ -n "${FROM:-}" ]; then
    if [ "$name" != "$FROM" ]; then echo "=== skip: $name"; return; fi
    FROM=""
  fi
  local t=$(date +%s)
  echo "=== $(date +%T) start: $name"
  "$@"
  echo "=== $(date +%T) done:  $name ($(( $(date +%s) - t )) s)"
}
fetch() { uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('$1')"; }
drop() { uv run python -c "from huggingface_hub import scan_cache_dir as s; c=s(); [c.delete_revisions(*[r.commit_hash for r in repo.revisions]).execute() for repo in c.repos if repo.repo_id=='$1']"; }
{
  trap 'echo "=== GEMMA FAILED at line $LINENO ($(date +%T))"' ERR
  [ -n "${HF_TOKEN:-}" ] || { echo "=== GEMMA FAILED: HF_TOKEN not set"; exit 1; }
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi dmon -s um -d 10 > logs/gpu_dmon.log 2>&1 & DMON_PID=$!
  fi
  echo "=== $(date) gemma start on $(uv run python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")'), commit $(cat BUNDLE_COMMIT.txt)"
  # G0 technical gates (BOS, sink, softcap, scorer = forward, SAE offset): gate of record
  step "G0 gates" uv run gbleed gemma-g0 $MODELS
  for M in $MODELS; do
    if [ "$(( $(date +%s) - START ))" -gt "$(python3 -c "print(int($MAX_HOURS*3600))")" ]; then
      echo "=== MAX_HOURS reached: not starting $M"; break
    fi
    S=$(echo "$M" | tr '/' '_' | sed 's/_/__/')
    step "$M download"            fetch "$M"
    # known-check filter on the frozen lists (+ portability report)
    step "$M known"               uv run gbleed gemma-known "$M"
    step "$M known pairs"         uv run gbleed p5 p4-known "$M"
    # activations for R1 (Phases 3-5) and R2 (frequency overlap)
    step "$M phase2 extract last" uv run gbleed phase2 "$M" --extract-only
    step "$M phase3 extract"      uv run gbleed phase3 "$M" --extract-only
    step "$M phase4 extract"      uv run gbleed phase4 "$M" --extract-only
    step "$M english extract"     uv run gbleed p5 english-extract "$M"
    step "$M number extract"      uv run gbleed p5 number-extract "$M"
    # Phase 5 gate frame check, readout check, gate + sweeps (one model load)
    step "$M phase5"              uv run gbleed p5 phase5-all "$M"
    step "$M save archive" tar cf "$OUT/$S.tar.part" activations/"$S" results BUNDLE_COMMIT.txt logs
    mv "$OUT/$S.tar.part" "$OUT/$S.tar"
    if [ -z "${KEEP_MODELS:-}" ]; then rm -rf activations/"$S"; drop "$M"; fi
  done
  gbleed_known_report() { uv run gbleed gemma-known $MODELS --report-only; }
  step "portability report" gbleed_known_report
  step "results archive" tar czf "$OUT/results.tgz.part" results logs BUNDLE_COMMIT.txt
  mv "$OUT/results.tgz.part" "$OUT/results.tgz"
  [ -n "${DMON_PID:-}" ] && kill "$DMON_PID" 2>/dev/null || true
  echo "=== GEMMA all done $(date)"
} 2>&1 | tee -a logs/session_gemma.log
