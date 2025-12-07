#!/bin/bash

# Ensure we are in the project root
cd "$(dirname "$0")"

# Activate Conda Environment
source /home/yasiru/miniconda3/etc/profile.d/conda.sh
conda activate /home/yasiru/miniconda3/envs/robobrain2-env

# Install requirements if needed
echo "Checking requirements..."
pip install -r requirements.txt

# Run the backend
echo "Starting backend..."
python backend.py
