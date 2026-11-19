#!/usr/bin/env bash
# Pause the content embedder (SIGSTOP) while the Mac is on battery; resume (SIGCONT) on AC.
# Checks every 60 s; exits when no content_full.sh is running. Usage: bash power_guard.sh &
PAT='content_embed\.py'
while pgrep -f 'content_full\.sh' >/dev/null; do
  if pmset -g batt | head -1 | grep -q "AC Power"; then
    pkill -CONT -f "$PAT" 2>/dev/null
  else
    pkill -STOP -f "$PAT" 2>/dev/null
  fi
  sleep 60
done
