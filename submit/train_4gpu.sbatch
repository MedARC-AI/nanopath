#!/usr/bin/env bash
#SBATCH --job-name=baseline_4gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/data/%u/nanopath/slurm/%j.out
#SBATCH --error=/data/%u/nanopath/slurm/%j.err
#SBATCH --partition=n
#SBATCH --account=sophont
#SBATCH --qos=normal # low | normal | top

set -euo pipefail

# Single-node SLURM launcher: torchrun on NPROC_PER_NODE GPUs, optional first
# arg overrides the YAML config (defaults to configs/leader.yaml). Adjust
# NPROC_PER_NODE and the #SBATCH resource lines above to scale up or down.
# SLURM logs land in /data/$USER/nanopath/slurm/<jobid>.{out,err}.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_FILE="${1:-${REPO_DIR}/configs/leader.yaml}"
NPROC_PER_NODE=4
JOB_ID="${SLURM_JOB_ID:-local-$$}"

cd "${REPO_DIR}"
export NANOPATH_DATA_DIR="${NANOPATH_DATA_DIR:-/data/${USER:-$(id -un)}/nanopath}"
mkdir -p "${NANOPATH_DATA_DIR}/slurm"
export NANOPATH_SLURM_STDOUT="${NANOPATH_DATA_DIR}/slurm/${JOB_ID}.out"
export NANOPATH_SLURM_STDERR="${NANOPATH_DATA_DIR}/slurm/${JOB_ID}.err"
exec >> "${NANOPATH_SLURM_STDOUT}" 2>> "${NANOPATH_SLURM_STDERR}"
source "${REPO_DIR}/.venv/bin/activate"

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "NANOPATH_DATA_DIR=${NANOPATH_DATA_DIR}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-local}"

torchrun \
  --standalone \
  --nproc_per_node "${NPROC_PER_NODE}" \
  train.py \
  "${CONFIG_FILE}"
