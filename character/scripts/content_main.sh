#!/usr/bin/env bash
# Step 3 of content_main.py: embedders on the fixed writer sample (CPU, fp32), mask-all and
# --keep-they; fastText also on the full writer outputs. Two jobs at a time, 4 threads each.
# End marker (whole line): ^=== CONTENT EMBED DONE$ (only if every job exited 0)
set -uo pipefail
cd "$(dirname "$0")/.."
O=results/content; FAIL=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
job() { uv run --with sentence-transformers python content_embed.py "$@" > "$O/log_$(basename "${@: -1}" .csv).txt" 2>&1 || { echo "=== FAILED $*"; return 1; }; }
S="$O/sample_luna_medium.jsonl $O/sample_luna_none.jsonl $O/sample_qwen_off.jsonl $O/sample_qwen_on.jsonl"
for pair in "bge-m3 e5-instruct" "labse qwen3-emb-0.6b" "embeddinggemma fasttext"; do
  set -- $pair
  job $S --only $1 --out $O/emb_$1.csv & p1=$!
  job $S --only $2 --out $O/emb_$2.csv & p2=$!
  wait $p1 || FAIL=1; wait $p2 || FAIL=1
  job $S --only $1 --keep-they --out $O/emb_$1_kept.csv & p1=$!
  job $S --only $2 --keep-they --out $O/emb_$2_kept.csv & p2=$!
  wait $p1 || FAIL=1; wait $p2 || FAIL=1
  echo "=== pair done: $pair $(date '+%F %T')"
done
for f in gpt-6-luna qwen3-14b_off qwen3-14b_on; do
  job results/main_v1/writer_$f.jsonl --only fasttext --out $O/emb_fasttext_all_$f.csv || FAIL=1
done
[ "$FAIL" -eq 0 ] && echo "=== CONTENT EMBED DONE" || echo "=== CONTENT EMBED INCOMPLETE"
