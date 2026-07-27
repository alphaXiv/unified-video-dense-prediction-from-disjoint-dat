#!/usr/bin/env bash
set -euo pipefail

python -m pip install --quiet --disable-pip-version-check -r requirements.txt
export HF_HOME=/tmp/huggingface
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

torchrun --standalone --nproc_per_node=4 src/reproduce.py --config configs/reproduction.json

