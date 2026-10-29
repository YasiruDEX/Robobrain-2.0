#!/usr/bin/env bash
# Follow-on to run_decomp.sh after the Groq daily token cap (user decision 2026-10-29 00:05): the Groq pre-pass was
# stopped, so Groq runs only on samples with a cached plan (--cached-only). Waits for run_decomp.sh (GPU) to exit.
#   GROQ_API_KEY=$(cat ~/.config/groq/key) nohup setsid bash eval/run_decomp_partial.sh > logs/run_decomp_partial.log 2>&1 &
set -uo pipefail
cd /home/jayamadu/Documents/Robobrain-2.0/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/jayamadu/miniconda3/envs/rb2eval/bin/python
R=../results
GM=openai/gpt-oss-120b

echo "=== WAITING for run_decomp.sh $(date +%T)"
while ps -eo args | grep -q "^bash eval/run_decomp.sh"; do sleep 30; done
grep -q "\] done (agent trajectory local vabench)" ../logs/run_decomp.log || { echo "=== FAILED local-vabench did not finish"; exit 1; }
echo "] done (wait) $(date +%T)"

job() {
  echo "=== START $1 $(date +%T)"
  shift
  "$@" || { echo "=== FAILED $*"; exit 1; }
}

job smoke-groq $PY run_agent.py --task trajectory --planner groq --groq-model $GM --dataset vabench --cached-only --out $R/decomp_smoke/groq --limit 3
$PY - <<'EOF' || { echo "=== FAILED smoke sanity"; exit 1; }
import json
rs = [json.loads(l) for l in open("../results/decomp_smoke/groq/trajectory_runs.jsonl")]
print("sanity:", [(r["plan_parsed"], r["tasks"], r["valid"]) for r in rs])
assert sum(bool(r["plan_parsed"]) for r in rs) >= 2 and sum(r["valid"] for r in rs) >= 2
EOF
echo "] done (sanity groq)"
job groq-vabench $PY run_agent.py --task trajectory --planner groq --groq-model $GM --dataset vabench --cached-only
job groq-sharerobot $PY run_agent.py --task trajectory --planner groq --groq-model $GM --dataset sharerobot --cached-only

echo "=== START summary $(date +%T)"
{
  echo "# Decomposition study (auto summary $(date +%F\ %T)); last = app-faithful, :concat = pre-registered headline on multi"
  echo "Groq VABench-V = cached-plan subset only (Groq free-tier daily token cap); deltas are paired on common ids."
  for d in vabench sharerobot; do
    T=$R/decomp/summary_$d.jsonl
    cat $R/decomp/groq_$d/traj.jsonl > $T
    if [ $d = vabench ]; then src=$R/decomp/local_vabench/traj.jsonl; else src=$R/agent/traj.jsonl; fi
    grep '"cond": "Agent-local' $src >> $T
    echo; echo "## $d (all ids)"
    $PY decomp_split.py --traj $T --dataset $d --conds Agent-local Agent-local:concat Agent-groq-120b Agent-groq-120b:concat
  done
} > $R/decomp/summary.md && cat $R/decomp/summary.md && echo "] done (summary)"
echo "=== DECOMP PARTIAL FINISHED $(date +%T)"
