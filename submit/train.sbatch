#!/usr/bin/env bash
#SBATCH --job-name=baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --time=01:30:00
#SBATCH --output=/data/nanopath/slurm/%j.out
#SBATCH --error=/data/nanopath/slurm/%j.err
#SBATCH --partition=n
#SBATCH --account=sophont
#SBATCH --qos=top # low | normal | top

set -euo pipefail

# Default launcher for the checked-in nano baseline run.
CONFIG_FILE="/admin/home/paul/nanopath/configs/nano.yaml"
NPROC_PER_NODE=1

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
