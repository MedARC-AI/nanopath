#!/usr/bin/env bash
set -euo pipefail
cd /admin/home/paul/nanopath
uv sync
source /admin/home/paul/nanopath/.venv/bin/activate
mkdir -p /data/nanopath
CUDA_VISIBLE_DEVICES=0 python train.py configs/nano.yaml
