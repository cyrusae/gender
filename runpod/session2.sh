#!/usr/bin/env bash
# Session 2 (plan: docs/design/runpod-session-plan.md): Phases 4-5 + suffix follow-up on 8B/14B,
# one chip for everything (the Phase 2/3 training activations are re-extracted here, so the
# steering directions are fitted on this chip). GPU steps only; analysis on the Mac.
# Run from the repo root after setup.sh:  nohup bash runpod/session2.sh > /dev/null 2>&1 < /dev/null &
# Watch: tail -f logs/session2.log    Resume after a failure: MODELS="..." bash runpod/session2.sh
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
  local t=$(date +%s)
  echo "=== $(date +%T) start: $name"
  "$@"
  echo "=== $(date +%T) done:  $name ($(( $(date +%s) - t )) s)"
}
fetch() { uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('$1')"; }
drop() { uv run python -c "from huggingface_hub import scan_cache_dir as s; c=s(); [c.delete_revisions(*[r.commit_hash for r in repo.revisions]).execute() for repo in c.repos if repo.repo_id=='$1']"; }
{
  echo "=== $(date) session 2 start on $(uv run python -c 'import torch; print(torch.cuda.get_device_name(0))'), commit $(cat BUNDLE_COMMIT.txt)"
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
    # Phase 5: frame checks, in-language baseline, gate + sweeps at every passing layer
    step "$M p22 axis"             uv run gbleed p5 p22 "$M"
    step "$M gate frame check"     uv run gbleed p5 gate-framecheck "$M"
    step "$M readout check"        uv run gbleed p5 readout-check "$M"
    step "$M p20"                  uv run gbleed p5 p20 "$M"
    step "$M phase5 session"       uv run gbleed p5 session "$M"
    step "$M save archive" tar cf "$OUT/$S.tar.part" activations/"$S" results BUNDLE_COMMIT.txt logs
    mv "$OUT/$S.tar.part" "$OUT/$S.tar"
    rm -rf activations/"$S"; drop "$M"
  done
  for M in $SMALL; do step "$M wug" uv run gbleed p5 wug "$M"; done
  step "results archive" tar czf "$OUT/results.tgz.part" results logs BUNDLE_COMMIT.txt
  mv "$OUT/results.tgz.part" "$OUT/results.tgz"
  echo "=== $(date) session 2 done"
} 2>&1 | tee -a logs/session2.log
