#!/usr/bin/env bash
# Resume the Agent-local affordance run after the 13:11 OOM (OOM now retried once, then counted as failure).
set -uo pipefail
cd /home/jayamadu/Documents/Robobrain-2.0/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== START agent-afford $(date +%T)"
/home/jayamadu/miniconda3/envs/rb2eval/bin/python run_agent.py --task affordance || { echo "=== FAILED agent-afford"; exit 1; }
echo "=== ALL AGENT JOBS FINISHED $(date +%T)"
