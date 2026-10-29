#!/usr/bin/env bash
# Dead-man's switch: stop the evaluation runner if the Claude session's watcher process dies
# (session ended / tokens ran out). Results are resumable: rerun eval/run_all.sh to continue.
ROOT=/home/jayamadu/Documents/Robobrain-2.0
LOG=${1:-$ROOT/logs/run_all.log}
PIDF=$ROOT/logs/watcher.pid
while pgrep -f "^bash eval/run_(all|extra).sh" >/dev/null; do
  if [ -f "$PIDF" ] && ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    pgrep -f "^bash eval/run_(all|extra).sh" | xargs -r kill
    pgrep -f "rb2eval/bin/python run_(traj|spatial|planner).py" | xargs -r kill
    echo "=== STOPPED by guard: Claude session watcher gone $(date +%T). Resume: rerun the runner script (resumable)" >> "$LOG"
    exit 0
  fi
  sleep 60
done
