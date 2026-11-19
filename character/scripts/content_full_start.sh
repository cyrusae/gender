#!/usr/bin/env bash
# Wait for the sample embedder run to end, then start the full run (sleep-proof on AC only via
# caffeinate -s) and the battery guard. Detach with: nohup bash scripts/content_full_start.sh &
cd "$(dirname "$0")/.."
until grep -qE "^=== CONTENT EMBED (DONE|INCOMPLETE)$" results/content/embed_run.log 2>/dev/null; do sleep 120; done
caffeinate -s bash scripts/content_full.sh > results/content/full_run.log 2>&1 &
sleep 20
bash scripts/power_guard.sh
wait
