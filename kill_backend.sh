#!/bin/bash
# Kill all backend.py processes and clean GPU memory

echo "Killing all backend.py processes..."
pkill -f "python.*backend.py" || echo "No backend processes found"

echo "Waiting for processes to terminate..."
sleep 2

echo "Checking GPU status..."
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo ""
echo "GPU memory freed. You can now restart the backend."
