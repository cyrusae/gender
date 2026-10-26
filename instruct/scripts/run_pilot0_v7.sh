#!/usr/bin/env bash
# Pilot 0 v7 model fishing (run from instruct/; resumable). End marker: ^=== PILOT0 V7 ALL DONE$
set -u
OUT=results/pilot0_v7; mkdir -p $OUT
for M in deepseek/deepseek-v4.1-flash qwen/qwen3.8-27b qwen/qwen3.8-flash mistralai/mistral-small-2603 openai/gpt-oss-120b anthropic/claude-haiku-5.5 z-ai/glm-5.3-flash; do
  N=${M##*/}; echo "=== start $N $(date '+%F %T')"
  uv run python gen.py --config configs/pilot0_v7.json --nouns stimuli/pilot0_nouns_v2.csv --model "$M" --backend openrouter --out $OUT/$N.jsonl 2>&1 | grep -vE "Warning|warn"
  echo "=== end $N $(date '+%F %T')"
done
echo "=== PILOT0 V7 ALL DONE"
