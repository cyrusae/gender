#!/usr/bin/env bash
# Main run, one batch for one model via OpenRouter (prereg v1). Resumable: gen.py skips samples
# already in the output, so a later batch with a larger N only adds the new draws.
# Usage: run_main_batch.sh MODEL WRITER_N CHAR_N [PLAIN_N]   e.g. openai/gpt-6-luna 10 2 10
# End marker (whole line): ^=== MAIN BATCH DONE <model-short> <writerN>/<charN>$
set -uo pipefail
cd "$(dirname "$0")/.."
M="$1"; WN="$2"; CN="$3"; PN="${4:-10}"; S="${M##*/}"; O=results/main_v1
mkdir -p "$O"
run() { echo "=== start $1 $(date '+%F %T')"; shift; uv run python gen.py "$@" --backend openrouter --model "$M" 2>&1 | grep -E "cost|Traceback|Error"; echo "=== end $(date '+%F %T')"; }
run writer --config configs/main_writer_v1.json --nouns stimuli/main_nouns_v1.csv \
  --translations stimuli/main_translations_v1.csv --n "$WN" --out "$O/writer_$S.jsonl"
run char-plain --config configs/main_character_v1.json --nouns stimuli/main_nouns_v1.csv \
  --mods-regex '\|plain_aan$' --n "$PN" --out "$O/character_$S.jsonl"
run char-mods --config configs/main_character_v1.json --nouns stimuli/main_nouns_v1.csv \
  --mods-regex '\|(?!plain_aan$)' --n "$CN" --out "$O/character_$S.jsonl"
echo "=== MAIN BATCH DONE $S $WN/$CN"
