#!/usr/bin/env bash
# Session 2 (plan: docs/design/runpod-session-plan.md): Phases 4-5 + suffix follow-up on 8B/14B,
# one chip for everything (the Phase 2/3 training activations are re-extracted here, so the
# steering directions are fitted on this chip). GPU steps only; analysis on the Mac.
# Run from the repo root after setup.sh:  nohup bash runpod/session2.sh > /dev/null 2>&1 < /dev/null &
# Watch: tail -f logs/session2.log    Resume after a failure: MODELS="..." bash runpod/session2.sh
#   (add FROM="<model> <step name>" to start at that step, e.g. FROM="Qwen/Qwen3-8B-Base known ru")
# Local end-to-end test (from an unpacked bundle, not the main checkout):
#   KEEP_MODELS=1 GBLEED_SMOKE=1 HF_HOME=~/.cache/huggingface OUT=/tmp/x \
#   MODELS=Qwen/Qwen3-0.6B-Base SMALL= bash runpod/session2.sh
set -euo pipefail
export HF_HOME=${HF_HOME:-/root/hf}
export PATH="$HOME/.local/bin:$PATH"
MODELS=${MODELS:-"Qwen/Qwen3-8B-Base Qwen/Qwen3-14B-Base"}
SMALL=${SMALL:-"Qwen/Qwen3-0.6B-Base Qwen/Qwen3-1.7B-Base Qwen/Qwen3-4B-Base"}  # wug test only
MAX_HOURS=${MAX_HOURS:-14}  # don't start a new model after this (budget guard)
START=$(date +%s)
OUT=${OUT:-/root/out}
mkdir -p logs "$OUT"
step() {
  local name=$1; shift
  # FROM="<model> <step>": skip steps until that one (resume after a failure without redoing)
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
  # machine checks (throttling, cores, RAM) and GPU utilization every 10 s for the whole session
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi -q -d PERFORMANCE | grep -iE "throttle|slowdown|active" | head -20 || true
    nvidia-smi dmon -s um -d 10 > logs/gpu_dmon.log 2>&1 & DMON_PID=$!
  fi
  echo "cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu)"; free -h 2>/dev/null | head -2 || true
  echo "=== $(date) session 2 start on $(uv run python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")'), commit $(cat BUNDLE_COMMIT.txt)"
  read -r -a MLIST <<< "$MODELS"
  PF_PID=""
  for idx in "${!MLIST[@]}"; do
    M=${MLIST[$idx]}
    if [ "$(( $(date +%s) - START ))" -gt "$(python3 -c "print(int($MAX_HOURS*3600))")" ]; then
      echo "=== MAX_HOURS reached: not starting $M"; break
    fi
    S=$(echo "$M" | tr '/' '_' | sed 's/_/__/')
    if [ -n "$PF_PID" ]; then wait "$PF_PID" || true; PF_PID=""; fi
    step "$M download" fetch "$M"
    NEXT=${MLIST[$((idx + 1))]:-}
    if [ -n "$NEXT" ]; then fetch "$NEXT" > "logs/prefetch.log" 2>&1 & PF_PID=$!; fi
    # behavioural checks: session 1's known checks again with the fp32 output layer (plan)
    step "$M phase0 v3"           uv run gbleed phase0 "$M" --stimuli data/stimuli/phase0_v3.csv
    step "$M known v3 fp32-head"  env GBLEED_FP32_HEAD=1 uv run gbleed phase0 "$M" --stimuli data/stimuli/phase0_v3.csv --out results/phase0_fp32head
    step "$M known es fp32-head"  env GBLEED_FP32_HEAD=1 uv run gbleed phase0 "$M" --stimuli data/stimuli/phase2_pool_v4.csv --out results/phase2_known_fp32head
    step "$M known de fp32-head"  env GBLEED_FP32_HEAD=1 uv run python -c "from gbleed import phase3_known as k; k.score('$M', out_root='results/phase3_known_fp32head')"
    step "$M known pairs"         uv run gbleed p5 p4-known "$M"
    step "$M known ru"            uv run gbleed p5 ru-known "$M"
    step "$M known suffix_ctrl"   uv run gbleed p5 suffix-known "$M"
    step "$M wug"                 uv run gbleed p5 wug "$M"
    # activations on this chip (directions are refitted from these)
    step "$M phase2 extract last"  uv run gbleed phase2 "$M" --extract-only
    step "$M phase2 extract after" uv run gbleed phase2 "$M" --position after --extract-only
    step "$M phase3 extract"       uv run gbleed phase3 "$M" --extract-only
    step "$M phase4 extract"       uv run gbleed phase4 "$M" --extract-only
    step "$M suffix extract"       uv run gbleed p5 suffix-extract "$M"
    step "$M english extract"      uv run gbleed p5 english-extract "$M"
    step "$M number extract"       uv run gbleed p5 number-extract "$M"
    step "$M diminutives extract"  uv run gbleed p5 diminutives-extract "$M"
    # Phase 5: frame checks, in-language baseline, gate + sweeps at every passing layer
    step "$M p22 axis"             uv run gbleed p5 p22 "$M"
    # gate frame check, readout check, P20, gate + sweeps: one process, one model load
    step "$M phase5"               uv run gbleed p5 phase5-all "$M"
    step "$M save archive" tar cf "$OUT/$S.tar.part" activations/"$S" results BUNDLE_COMMIT.txt logs
    mv "$OUT/$S.tar.part" "$OUT/$S.tar"
    if [ -z "${KEEP_MODELS:-}" ]; then rm -rf activations/"$S"; drop "$M"; fi
  done
  for M in $SMALL; do step "$M wug" uv run gbleed p5 wug "$M"; done
  step "results archive" tar czf "$OUT/results.tgz.part" results logs BUNDLE_COMMIT.txt
  mv "$OUT/results.tgz.part" "$OUT/results.tgz"
  [ -n "${DMON_PID:-}" ] && kill "$DMON_PID" 2>/dev/null || true
  echo "=== $(date) session 2 done"
} 2>&1 | tee -a logs/session2.log
