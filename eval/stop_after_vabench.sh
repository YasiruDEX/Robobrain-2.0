#!/usr/bin/env bash
# Stop run_extra.sh cleanly once VABench-V is finished (before the Qwen job does real work).
cd /home/jayamadu/Documents/Robobrain-2.0
until grep -qE "done \(vabench summary\)|START qwen|FAILED" logs/run_extra.log; do sleep 15; done
for p in $(ps -eo pid,args | awk '($2=="bash" && $3=="eval/run_extra.sh") || ($3=="run_traj.py") {print $1}'); do kill $p; done
echo "=== STOPPED for checkpoint after VABench-V $(date +%T)" >> logs/run_extra.log
