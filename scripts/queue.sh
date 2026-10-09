#!/bin/bash
# Run analysis jobs from a list, a few at a time, without overfilling the Mac's memory.
#   scripts/queue.sh JOBS_FILE [SLOTS] [MIN_FREE_PCT]
# Each line of JOBS_FILE is the arguments to `uv run gbleed` (read once at start, so editing the
# file afterwards has no effect). A job starts only when fewer than SLOTS (default 4) gbleed
# processes of this checkout are running AND macOS reports at least MIN_FREE_PCT (default 40)
# percent of memory free; jobs then get 2 minutes to load data before the next check.
# Logs: logs/queue/<n>_<job>.log; progress lines go to stdout.
# (2026-10-08: 8 parallel jobs incl. four 30B-A3B analyses filled 23 GB of swap.)
set -u
JOBS_FILE=$1; SLOTS=${2:-4}; MIN_FREE=${3:-40}
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs/queue
JOBS=(); while IFS= read -r l; do [ -n "$l" ] && JOBS+=("$l"); done < "$JOBS_FILE"  # bash 3.2
running() { pgrep -f "$ROOT/.venv/bin/gbleed " | wc -l | tr -d ' '; }
free_pct() { memory_pressure 2>/dev/null | awk '/free percentage/ {gsub("%",""); print $NF}'; }
n=0
for job in "${JOBS[@]}"; do
  n=$((n + 1))
  while [ "$(running)" -ge "$SLOTS" ] || [ "$(free_pct)" -lt "$MIN_FREE" ]; do sleep 30; done
  log="logs/queue/$(printf %03d $n)_$(echo "$job" | tr ' /' '__' | cut -c1-60).log"
  echo "$(date +%H:%M:%S) start $n (free $(free_pct)%): $job"
  ( uv run gbleed $job > "$log" 2>&1; rc=$?
    echo "$(date +%H:%M:%S) end $n rc=$rc tracebacks=$(grep -ac Traceback "$log"): $job" ) &
  sleep 120
done
wait; echo "$(date +%H:%M:%S) QUEUE DONE"
