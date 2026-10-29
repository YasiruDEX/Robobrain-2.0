#!/usr/bin/env bash
# Show experiment progress. Usage: bash eval/progress.sh   (live: watch -n 30 bash eval/progress.sh)
cd /home/jayamadu/Documents/Robobrain-2.0
count() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
bar() {  # bar <done> <total>
  local d=$1 t=$2 w=30 f=$(( $1 * 30 / $2 ))
  local fill=""; [ "$f" -gt 0 ] && fill=$(printf '#%.0s' $(seq 1 $f))
  printf "[%-${w}s] %4d/%d (%d%%)" "$fill" "$d" "$t" $(( d * 100 / t ))
}
running() { ps -eo args | grep -q "$1" && echo "running" || echo "not running"; }

echo "=== Experiment progress $(date +%T) ==="
echo
printf "VABench-V trajectories   "; bar "$(count results/vabench/traj.jsonl)" 2100; echo
echo "  conditions done: $(sed -n '/START vabench/,/START qwen/p' logs/run_extra.log | grep -oE '\] done \((B|C-[a-z]+)\)' | sed 's/] done (//;s/)//' | awk '!s[$0]++' | tr '\n' ' ') (of B C-text C-pred C-point C-shuffle C-gtend C-goal)"
printf "Qwen pass 1 (grounding)  "; bar "$(count results/traj_qwen/steps.jsonl)" 200; echo
printf "Qwen pass 1b (end loc.)  "; bar "$(count results/traj_qwen/goal_steps.jsonl)" 200; echo
printf "Qwen2.5-VL-3B (base)     "; bar "$(count results/traj_qwen/traj.jsonl)" 1400; echo
printf "Agent-local trajectory   "; bar "$(count results/agent/trajectory_runs.jsonl)" 200; echo
printf "Agent-local affordance   "; bar "$(count results/agent/affordance_runs.jsonl)" 200; echo
eta() {  # eta <file> <total> <start HH:MM:SS> <done at start>
  local n=$(count "$1") el=$(( $(date +%s) - $(date -d "$3" +%s) ))
  [ "$n" -gt "$4" ] && [ "$n" -lt "$2" ] && \
    echo "  ETA $(date -d "@$(( $(date +%s) + el * ($2 - n) / (n - $4) ))" +%H:%M) ($(( (n - $4) * 60 / el )) samples/min)"
}
eta results/agent/affordance_runs.jsonl 200 "$(grep -oP 'START agent-afford \K[0-9:]+' logs/run_agent.log | tail -1)" 4
echo
echo "runner (run_extra.sh):   $(running '^bash eval/run_extra.sh')"
echo "agent  (run_agent.sh):   $(running '^bash eval/run_agent.sh')"
echo "afford (run_afford.sh):  $(running '^bash eval/run_afford.sh')"
echo "stopper (after VABench): $(running '^bash eval/stop_after_vabench.sh')"
echo "GPU memory used:         $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
echo
grep -E "STOPPED|FAILED|Traceback|ALL JOBS FINISHED" logs/run_extra.log | tail -2 | sed 's/^/run_extra: /'
grep -E "OOM|FAILED|Traceback|ALL AGENT JOBS FINISHED" logs/run_agent.log | tail -3 | sed 's/^/agent:     /'
grep -q "STOPPED for checkpoint after VABench-V" logs/run_extra.log \
  && echo ">>> VABench-V finished and jobs stopped: GPU is free for your other task."

# Decomposition study (progress/05_decomposition_plan.md). Paths are fixed here; the Step 2 code writes to them.
echo
echo "=== Decomposition study (Groq vs inbuilt decoder) ==="
[ -f data/subsets.json ] && echo "Step 0 subsets.json:      done" || echo "Step 0 subsets.json:      missing"
grep -q "Step 1 DONE" progress/05_decomposition_plan.md && echo "Step 1 split (existing):  done" || echo "Step 1 split (existing):  not done"
for f in eval/groq_plans.py eval/run_decomp.sh; do
  printf "Step 2 %-18s %s\n" "$(basename $f):" "$([ -f $f ] && echo written || echo 'not written')"
done
printf "Groq plans gpt-oss-120b  "; bar "$(count results/groq_cache/openai_gpt-oss-120b.jsonl)" 391; echo
printf "Agent-Groq  VABench-V    "; bar "$(count results/decomp/groq_vabench/trajectory_runs.jsonl)" 300; echo
printf "Agent-local VABench-V    "; bar "$(count results/decomp/local_vabench/trajectory_runs.jsonl)" 300; echo
printf "Agent-Groq  ShareRobot   "; bar "$(count results/decomp/groq_sharerobot/trajectory_runs.jsonl)" 200; echo
echo "decomp (run_decomp.sh):  $(running '^bash eval/run_decomp.sh')"
[ -f logs/run_decomp.log ] && grep -E "=== START|\] done \(|FAILED|Traceback|out of memory|FINISHED" logs/run_decomp.log | tail -4 | sed 's/^/decomp:    /'
exit 0
