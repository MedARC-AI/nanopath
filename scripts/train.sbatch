#!/usr/bin/env bash
#SBATCH --job-name=nanopath-nano
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --output=/admin/home/paul/nanopath/slurms/%j.out
#SBATCH --error=/admin/home/paul/nanopath/slurms/%j.err
#SBATCH --partition=main
#SBATCH --account=sophont
#SBATCH --qos=top # low | normal | top

set -euo pipefail

CONFIG_FILE="/admin/home/paul/nanopath/configs/nano.yaml"
NPROC_PER_NODE=1

cd /admin/home/paul/nanopath
mkdir -p slurms
mkdir -p /data/nanopath
uv sync
source /admin/home/paul/nanopath/.venv/bin/activate

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"

torchrun \
  --standalone \
  --nproc_per_node "${NPROC_PER_NODE}" \
  train.py \
  "${CONFIG_FILE}"
