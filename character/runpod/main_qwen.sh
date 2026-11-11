#!/usr/bin/env bash
# Main run (prereg v1 + amendments 1-3), Qwen3-14B on a pod via vLLM bf16. One PART per call:
#   off  = thinking off  (qwen3_nothinking sampling)        writer N=WN, char plain PN, char mods CN
#   on   = thinking on   (qwen3_thinking sampling)          same draws
#   ctrl = thinking off with qwen3_thinking sampling (amendment 3 control): writer plain+English-word
#          arms, character plain a/an only
# Resumable (gen.py skips rows already written; output is flushed after every chunk of prompts).
# Usage: bash character/runpod/main_qwen.sh PART [WN CN PN]     (defaults 10 2 10 = batch 1)
# Markers (whole lines): ^=== start <step>, ^=== FAILED exit N, ^=== QWEN MAIN <PART> DONE$ (only if
# every step exited 0) or ^=== QWEN MAIN <PART> INCOMPLETE$
set -uo pipefail
cd "$(dirname "$0")/.."
PART="$1"; WN="${2:-10}"; CN="${3:-2}"; PN="${4:-10}"
M=${M:-Qwen/Qwen3-14B}; O=${O:-results/main_v1}; mkdir -p "$O"
PY=${PY:-python}; BACKEND=${BACKEND:-vllm}; EXTRA=${EXTRA:-}
FAIL=0
step() { local name="$1"; shift; echo "=== start $name $(date '+%F %T')"
  $PY gen.py --model "$M" --backend "$BACKEND" "$@" $EXTRA 2>&1 \
    | grep --line-buffered -E "chunk|requested in|Traceback|Error|GEN DONE"
  local st=${PIPESTATUS[0]}; [ "$st" -ne 0 ] && FAIL=1 && echo "=== FAILED exit $st"
  echo "=== end $name $(date '+%F %T')"; }
W=(--config configs/main_writer_v1.json --nouns stimuli/main_nouns_v1.csv --translations stimuli/main_translations_v1.csv)
C=(--config configs/main_character_v1.json --nouns stimuli/main_nouns_v1.csv)
case "$PART" in
  off|on)
    step "writer-$PART" "${W[@]}" --thinking "$PART" --n "$WN" --out "$O/writer_qwen3-14b_$PART.jsonl"
    step "char-plain-$PART" "${C[@]}" --thinking "$PART" --mods-regex '\|plain_aan$' --n "$PN" \
      --out "$O/character_qwen3-14b_$PART.jsonl"
    step "char-mods-$PART" "${C[@]}" --thinking "$PART" --mods-regex '\|(?!plain_aan$)' --n "$CN" \
      --out "$O/character_qwen3-14b_$PART.jsonl" ;;
  ctrl)
    step writer-ctrl "${W[@]}" --thinking off --sampling-key qwen3_thinking --n "$WN" \
      --mods-regex '^(writer\|plain|writer\|[a-z]+:english|consider\|english)$' \
      --out "$O/writer_qwen3-14b_offthinkset.jsonl"
    step char-plain-ctrl "${C[@]}" --thinking off --sampling-key qwen3_thinking --n "$PN" \
      --mods-regex '\|plain_aan$' --out "$O/character_qwen3-14b_offthinkset.jsonl" ;;
  *) echo "PART must be off|on|ctrl"; exit 2 ;;
esac
if [ "$FAIL" -eq 0 ]; then echo "=== QWEN MAIN $PART DONE"; else echo "=== QWEN MAIN $PART INCOMPLETE"; fi
