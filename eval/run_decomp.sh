#!/usr/bin/env bash
# Decomposition study (progress/05_decomposition_plan.md): Groq vs inbuilt decoder on multi-stop tasks.
# Groq plan pre-pass (CPU/network) runs in the background while the GPU does Agent-local on VABench-V.
# Resumable: every step skips cached / finished ids. Launch from the repo root:
#   GROQ_API_KEY=$(cat ~/.config/groq/key) nohup setsid bash eval/run_decomp.sh > logs/run_decomp.log 2>&1 &
set -uo pipefail
cd /home/jayamadu/Documents/Robobrain-2.0/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/jayamadu/miniconda3/envs/rb2eval/bin/python
R=../results
GM=openai/gpt-oss-120b
[ -n "${GROQ_API_KEY:-}" ] || { echo "=== FAILED no GROQ_API_KEY"; exit 1; }

job() {
  echo "=== START $1 $(date +%T)"
  shift
  "$@" || { echo "=== FAILED $*"; exit 1; }
}
sanity() {  # sanity <runs.jsonl>: plans parsed >= 2/3 and >= 2/3 valid trajectories
  $PY - "$1" <<'EOF' || { echo "=== FAILED smoke sanity $1"; exit 1; }
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1])]
print("sanity:", [(r["plan_parsed"], r["tasks"], r["valid"]) for r in rs])
assert len(rs) == 3 and sum(bool(r["plan_parsed"]) for r in rs) >= 2 and sum(r["valid"] for r in rs) >= 2
EOF
  echo "] done (sanity $1)"
}

echo "=== START groq-plans (background) $(date +%T)"
$PY groq_plans.py --model $GM > ../logs/groq_plans.log 2>&1 &
GP=$!

job smoke-local $PY run_agent.py --task trajectory --planner local --dataset vabench --out $R/decomp_smoke/local --limit 3
sanity $R/decomp_smoke/local/trajectory_runs.jsonl
job local-vabench $PY run_agent.py --task trajectory --planner local --dataset vabench

echo "=== WAITING groq-plans $(date +%T)"
wait $GP || { echo "=== FAILED groq-plans (see logs/groq_plans.log)"; exit 1; }
grep "\] done (groq" ../logs/groq_plans.log

job smoke-groq $PY run_agent.py --task trajectory --planner groq --groq-model $GM --dataset vabench --out $R/decomp_smoke/groq --limit 3
sanity $R/decomp_smoke/groq/trajectory_runs.jsonl
job groq-vabench $PY run_agent.py --task trajectory --planner groq --groq-model $GM --dataset vabench
job groq-sharerobot $PY run_agent.py --task trajectory --planner groq --groq-model $GM --dataset sharerobot

echo "=== START summary $(date +%T)"
{
  echo "# Decomposition study (auto summary $(date +%F\ %T)); last = app-faithful, :concat = pre-registered headline on multi"
  for d in vabench sharerobot; do
    T=$R/decomp/summary_$d.jsonl
    cat $R/decomp/groq_$d/traj.jsonl > $T
    if [ $d = vabench ]; then src=$R/decomp/local_vabench/traj.jsonl; else src=$R/agent/traj.jsonl; fi
    grep '"cond": "Agent-local' $src >> $T
    echo; echo "## $d"
    $PY decomp_split.py --traj $T --dataset $d --conds Agent-local Agent-local:concat Agent-groq-120b Agent-groq-120b:concat
  done
} > $R/decomp/summary.md && cat $R/decomp/summary.md && echo "] done (summary)"
echo "=== DECOMP RUNNER FINISHED $(date +%T)"
