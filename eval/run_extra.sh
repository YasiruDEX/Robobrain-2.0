#!/usr/bin/env bash
# Extra experiments (resumable): C-goal on ShareRobot, VABench-V (all conds), base Qwen2.5-VL-3B.
set -uo pipefail
cd /home/jayamadu/Documents/Robobrain-2.0/eval
PY=/home/jayamadu/miniconda3/envs/rb2eval/bin/python
R=../results
QW=../weights/Qwen2.5-VL-3B-Instruct

job() {
  echo "=== START $1 $(date +%T)"
  shift
  "$@" || { echo "=== FAILED $*"; exit 1; }
}

# base model download runs alongside the GPU jobs
/home/jayamadu/miniconda3/envs/rb2eval/bin/huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir $QW > ../logs/qwen_download.log 2>&1 &
DL=$!

job goal $PY run_traj.py --data ../data/sharerobot_traj.jsonl --out $R/traj --conds C-goal
$PY summarize.py --out $R/traj > /dev/null && echo "] done (traj summary)"
job vabench $PY run_traj.py --data ../data/vabench_v.jsonl --out $R/vabench
$PY summarize.py --out $R/vabench > /dev/null && echo "] done (vabench summary)"
wait $DL || { echo "=== FAILED qwen download"; exit 1; }
echo "] done (qwen download)"
job qwen $PY run_traj.py --data ../data/sharerobot_traj.jsonl --out $R/traj_qwen --model $QW
$PY summarize.py --out $R/traj_qwen > /dev/null && echo "] done (qwen summary)"
echo "=== ALL JOBS FINISHED $(date +%T)"
