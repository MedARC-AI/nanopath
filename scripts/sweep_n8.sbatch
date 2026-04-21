#!/usr/bin/env bash
#SBATCH --job-name=nanopath-sweep-n8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --nodelist=n-8
#SBATCH --time=08:00:00
#SBATCH --output=/admin/home/paul/nanopath/slurms/%j.out
#SBATCH --error=/admin/home/paul/nanopath/slurms/%j.err
#SBATCH --partition=main
#SBATCH --account=sophont
#SBATCH --qos=normal

set -euo pipefail

MANIFEST="${MANIFEST:-/admin/home/paul/nanopath/configs/sweeps/micro_mean_probe_f1_v1.yaml}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

cd /admin/home/paul/nanopath
mkdir -p slurms
mkdir -p /data/nanopath
uv sync
source /admin/home/paul/nanopath/.venv/bin/activate

echo "MANIFEST=${MANIFEST}"
echo "GPU_IDS=${GPU_IDS}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"

./.venv/bin/python scripts/run_sweep_local.py "${MANIFEST}" "${GPU_IDS}"
