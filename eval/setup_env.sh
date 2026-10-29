#!/usr/bin/env bash
# Creates the rb2eval conda env and downloads RoboBrain2.0-3B weights. Idempotent.
set -uo pipefail
CONDA=/home/jayamadu/miniconda3/bin/conda
ENV=/home/jayamadu/miniconda3/envs/rb2eval
ROOT=/home/jayamadu/Documents/Robobrain-2.0
echo "=== START setup $(date)"
if [ ! -x "$ENV/bin/python" ]; then
  $CONDA create -y -p "$ENV" python=3.10 || { echo "=== FAILED conda create"; exit 1; }
fi
PIP="$ENV/bin/pip"
$PIP install --quiet torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 || { echo "=== FAILED torch"; exit 1; }
echo "] done (torch)"
$PIP install --quiet transformers==4.50.0 accelerate==1.3.0 qwen-vl-utils==0.0.8 bitsandbytes==0.45.5 \
  numpy==1.26.4 pillow scipy pandas matplotlib datasets huggingface_hub opencv-python-headless tqdm || { echo "=== FAILED pip deps"; exit 1; }
echo "] done (deps)"
"$ENV/bin/python" -c "import torch,transformers,bitsandbytes;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'tf',transformers.__version__)" || { echo "=== FAILED import check"; exit 1; }
"$ENV/bin/huggingface-cli" download BAAI/RoboBrain2.0-3B --local-dir "$ROOT/weights/RoboBrain2.0-3B" || { echo "=== FAILED weights download"; exit 1; }
echo "] done (weights)"
du -sh "$ROOT/weights/RoboBrain2.0-3B"
echo "=== RUNNER FINISHED $(date)"
