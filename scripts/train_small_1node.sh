#!/usr/bin/env bash
set -euo pipefail
cd /admin/home/paul/nanopath
uv sync
source /admin/home/paul/nanopath/.venv/bin/activate
mkdir -p /data/nanopath
torchrun --standalone --nproc_per_node=8 train.py configs/small.yaml
