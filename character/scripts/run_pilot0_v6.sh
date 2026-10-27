#!/usr/bin/env bash
# Pilot 0 v6 on OpenRouter, one model at a time. Safe to re-run after an interruption:
# each gen.py call resumes its own output file (completed rows are kept and skipped).
# Run from character/:  bash scripts/run_pilot0_v6.sh >> results/pilot0_v6/run.log 2>&1
# End marker (whole line): ^=== PILOT0 V6 ALL DONE$
set -u
C=configs/pilot0_v6.json; N=stimuli/pilot0_nouns_v2.csv; O=results/pilot0_v6
run() {  # model thinking outname [extra args]
  local m=$1 t=$2 o=$3; shift 3
  echo "=== start $o $(date '+%F %T')"
  uv run python gen.py --config $C --nouns $N --model "$m" --backend openrouter --thinking "$t" \
    --out "$O/$o.jsonl" "$@" 2>&1 | grep -vE "Warning|warn\(" || true
  echo "=== end $o $(date '+%F %T')"
}
run meta-llama/llama-3.3-70b-instruct off llama33_70b
run openai/gpt-6-luna off gpt6_luna
run google/gemma-4-31b-it off gemma4_31b
run google/gemma-4-31b-it on gemma4_31b_think --only-conds free_noted
echo "=== PILOT0 V6 ALL DONE"
