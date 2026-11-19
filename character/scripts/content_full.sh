#!/usr/bin/env bash
# Full-run embedder scores (exploratory content measures), on the Mac CPU in the background.
# Every output of both studies x 5 transformer embedders: mask-all pass on all rows, keep-they pass
# on he/she paragraphs that contain they-forms (others score the same either way).
# Saves as it goes: chunks of $CH rows, each written atomically to results/content/full/; a chunk
# whose file exists is skipped, so a restart resumes. Waits (between chunks) while on battery;
# scripts/power_guard.sh pauses a running chunk within a minute of unplugging.
# Low priority (nice 15), $THREADS threads, one job at a time.
# End marker (whole line): ^=== CONTENT FULL DONE$  (only if every chunk exited 0)
set -uo pipefail
cd "$(dirname "$0")/.."
O=results/content/full; mkdir -p "$O"; CH=${CH:-10000}; THREADS=${THREADS:-4}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS TOKENIZERS_PARALLELISM=false
on_ac() { pmset -g batt | head -1 | grep -q "AC Power"; }
PY=(nice -n 15 uv run --with sentence-transformers python content_embed.py)
FAIL=0
for e in bge-m3 e5-instruct labse qwen3-emb-0.6b embeddinggemma; do
  for kind in writer character; do
    for f in gpt-6-luna qwen3-14b_off qwen3-14b_on; do
      in=results/main_v1/${kind}_$f.jsonl
      for pass in all kept; do
        flags=(); [ "$pass" = kept ] && flags=(--keep-they --they-only)
        n=$("${PY[@]}" "$in" --only "$e" ${flags[@]+"${flags[@]}"} --count --out /dev/null 2>/dev/null | tail -1)
        [[ "$n" =~ ^[0-9]+$ ]] || { echo "=== FAILED count $e $kind $f $pass"; FAIL=1; continue; }
        for ((k = 0; k * CH < n; k++)); do
          out="$O/${e}_${kind}_${f}_${pass}_$(printf %03d $k).csv"
          [ -s "$out" ] && continue
          while ! on_ac; do sleep 300; done
          if "${PY[@]}" "$in" --only "$e" ${flags[@]+"${flags[@]}"} --skip $((k * CH)) --take "$CH" \
               --out "$out.tmp" > "$O/last.log" 2>&1; then
            mv "$out.tmp" "$out"; echo "chunk $(date '+%F %T') $e $kind $f $pass $((k + 1))/$(((n + CH - 1) / CH))"
          else
            echo "=== FAILED $e $kind $f $pass chunk $k"; FAIL=1; rm -f "$out.tmp"
          fi
        done
      done
    done
  done
done
[ "$FAIL" -eq 0 ] && echo "=== CONTENT FULL DONE" || echo "=== CONTENT FULL INCOMPLETE"
