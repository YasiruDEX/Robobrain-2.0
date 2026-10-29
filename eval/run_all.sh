#!/usr/bin/env bash
# Full evaluation, priority order. Resumable: every script skips finished (id, cond) keys.
# Log markers: "=== START <job>", "] done (...)", "=== FAILED", "=== ALL JOBS FINISHED".
set -uo pipefail
cd /home/jayamadu/Documents/Robobrain-2.0/eval
PY=/home/jayamadu/miniconda3/envs/rb2eval/bin/python
R=../results
mkdir -p $R

job() {  # job <name> <cmd...>
  echo "=== START $1 $(date +%T)"
  shift
  "$@" || { echo "=== FAILED $*"; exit 1; }
}

# 0. sanity: 3 trajectory samples, all conditions; abort if nothing parses
job smoke $PY run_traj.py --data ../data/sharerobot_traj.jsonl --out $R/smoke --limit 3
$PY - <<'EOF' || { echo "=== FAILED sanity check"; exit 1; }
import json
t = [json.loads(l) for l in open("../results/smoke/traj.jsonl")]
s = [json.loads(l) for l in open("../results/smoke/steps.jsonl")]
nb, nv = sum(r["box"] is not None for r in s), sum(r["valid"] for r in t)
print(f"sanity: boxes {nb}/{len(s)}, valid trajectories {nv}/{len(t)}")
for r in t[:2]: print("  raw:", r.get("raw", "")[:120])
assert nb > 0 and nv > 0
EOF

# 1. headline: grounding-conditioned trajectory on ShareRobot-Bench (200 x 6 conditions)
job traj $PY run_traj.py --data ../data/sharerobot_traj.jsonl --out $R/traj
$PY summarize.py --out $R/traj > /dev/null && echo "] done (traj summary)"
# 2. grounding-conditioned affordance (200 x 4 conditions)
job affordance $PY run_spatial.py --task affordance --out $R/spatial
# 3. single-task references
job grounding $PY run_spatial.py --task grounding --out $R/spatial
job pointing $PY run_spatial.py --task pointing --out $R/spatial
# 4. task routing
job planner $PY run_planner.py --out $R/planner
echo "=== ALL JOBS FINISHED $(date +%T)"
