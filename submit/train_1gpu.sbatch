#!/usr/bin/env bash
#SBATCH --job-name=baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=n
#SBATCH --account=sophont
#SBATCH --qos=normal # low | normal | top

set -euo pipefail

# Single-GPU SLURM launcher: torchrun with NPROC_PER_NODE=1, optional first
# arg overrides the YAML config (defaults to configs/leader.yaml).
# Runtime logs are tee'd to ${NANOPATH_DATA_DIR}/slurm/<jobid>.{out,err}.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_FILE="${1:-${REPO_DIR}/configs/leader.yaml}"
NPROC_PER_NODE=1
JOB_ID="${SLURM_JOB_ID:-local-$$}"

cd "${REPO_DIR}"
export NANOPATH_DATA_DIR="${NANOPATH_DATA_DIR:-/data/${USER:-$(id -un)}/nanopath}"
mkdir -p "${NANOPATH_DATA_DIR}/slurm"
export NANOPATH_SLURM_STDOUT="${NANOPATH_DATA_DIR}/slurm/${JOB_ID}.out"
export NANOPATH_SLURM_STDERR="${NANOPATH_DATA_DIR}/slurm/${JOB_ID}.err"
exec > >(tee -a "${NANOPATH_SLURM_STDOUT}") 2> >(tee -a "${NANOPATH_SLURM_STDERR}" >&2)
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
