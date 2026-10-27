#!/usr/bin/env bash
# Writer + Casey/Riley pilot (pilot0_v9) on four API models, one at a time, resumable.
# End marker (whole line): ^=== PILOT0 V9 ALL DONE$
cd "$(dirname "$0")/.."
for M in openai/gpt-oss-120b z-ai/glm-5.3-flash qwen/qwen3.8-flash openai/gpt-6-luna; do
  for K in writer casey; do
    echo "=== start $K ${M##*/} $(date '+%F %T')"
    uv run python gen.py --config configs/pilot0_${K}_v1.json --nouns stimuli/pilot0_nouns_v2.csv \
      --translations stimuli/pilot0_translations_v1.csv --model "$M" --backend openrouter \
      --out results/pilot0_v9/${K}_${M##*/}.jsonl 2>&1 | grep -E "cost|Traceback|Error"
    echo "=== end $K ${M##*/} $(date '+%F %T')"
  done
done
echo "=== PILOT0 V9 ALL DONE"
