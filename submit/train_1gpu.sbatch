#!/usr/bin/env bash
#SBATCH --job-name=baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --partition=n
#SBATCH --account=sophont
#SBATCH --qos=normal # low | normal | top

set -euo pipefail

# Single-GPU SLURM launcher: torchrun with NPROC_PER_NODE=1, optional first
# arg overrides the YAML config (defaults to configs/leader.yaml).
# Logs land in repo-local slurm/<jobid>.{out,err}.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_FILE="${1:-${REPO_DIR}/configs/leader.yaml}"
NPROC_PER_NODE=1

cd "${REPO_DIR}"
# Worktrees share the venv from the main checkout at /admin/home/$USER/nanopath.
VENV_DIR="${REPO_DIR}/.venv"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  VENV_DIR="${HOME}/nanopath/.venv"
fi
source "${VENV_DIR}/bin/activate"

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-local}"

torchrun \
  --standalone \
  --nproc_per_node "${NPROC_PER_NODE}" \
  train.py \
  "${CONFIG_FILE}"
