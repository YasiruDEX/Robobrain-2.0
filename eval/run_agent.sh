#!/usr/bin/env bash
# Agent-local benchmark (original app layer, Groq replaced by RoboBrain's own Qwen decoder).
# Waits for run_extra.sh to finish (GPU fits one model), then smoke test -> full runs. Resumable.
set -uo pipefail
cd /home/jayamadu/Documents/Robobrain-2.0/eval
PY=/home/jayamadu/miniconda3/envs/rb2eval/bin/python
R=../results

echo "=== WAITING for run_extra.sh $(date +%T)"
while ps -eo args | grep -q "^bash eval/run_extra.sh"; do sleep 60; done
echo "] done (wait) $(date +%T)"

job() {
  echo "=== START $1 $(date +%T)"
  shift
  "$@" || { echo "=== FAILED $*"; exit 1; }
}

job agent-smoke $PY run_agent.py --task trajectory --out $R/agent_smoke --limit 3
$PY - <<'EOF' || { echo "=== FAILED agent sanity"; exit 1; }
import json
rs = [json.loads(l) for l in open("../results/agent_smoke/trajectory_runs.jsonl")]
print("sanity:", [(r["plan_parsed"], r["plan_repaired"], r["tasks"], r["valid"]) for r in rs])
assert sum(r["plan_parsed"] for r in rs) >= 2, "planner JSON still failing"
assert len(rs) == 3
EOF
job agent-traj $PY run_agent.py --task trajectory
$PY summarize.py --out $R/agent > /dev/null && echo "] done (agent summary)"
job agent-afford $PY run_agent.py --task affordance
echo "=== ALL AGENT JOBS FINISHED $(date +%T)"
