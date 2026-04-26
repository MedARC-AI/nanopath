#!/usr/bin/env bash
#SBATCH --job-name=jepa-8gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/data/nanopath/slurm/%j.out
#SBATCH --error=/data/nanopath/slurm/%j.err
#SBATCH --partition=n
#SBATCH --account=sophont
#SBATCH --qos=top # low | normal | top

set -euo pipefail

# Full-node SLURM launcher: torchrun on all 8 GPUs, defaults to the 8-GPU
# config (configs/small_8gpu.yaml) which scales global_batch_size so each rank
# sees 16 samples. Pass a different YAML as the first arg to override.
# Logs land in /data/nanopath/slurm/<jobid>.{out,err}.
CONFIG_FILE="${1:-/admin/home/paul/nanopath/configs/small_8gpu.yaml}"
NPROC_PER_NODE=8

cd /admin/home/paul/nanopath
mkdir -p /data/nanopath
mkdir -p /data/nanopath/slurm
source /admin/home/paul/nanopath/.venv/bin/activate
export NANOPATH_SLURM_STDOUT="/data/nanopath/slurm/${SLURM_JOB_ID}.out"
export NANOPATH_SLURM_STDERR="/data/nanopath/slurm/${SLURM_JOB_ID}.err"

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME}"

torchrun \
  --standalone \
  --nproc_per_node "${NPROC_PER_NODE}" \
  train.py \
  "${CONFIG_FILE}"
