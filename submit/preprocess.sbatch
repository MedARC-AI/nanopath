#!/usr/bin/env bash
#SBATCH --job-name=preprocess
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --partition=c
#SBATCH --account=sophont
#SBATCH --qos=normal

set -euo pipefail

# CPU-node launcher for preprocessing.py. Decodes 4M 224x224 JPEG tiles from
# the TCGA SVS sample list into data.dataset_dir; manifest.txt at the end.
# Reruns are resumable: existing JPEGs are skipped. Optional first arg is the
# YAML config (defaults to configs/leader.yaml).
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_FILE="${1:-${REPO_DIR}/configs/leader.yaml}"

cd "${REPO_DIR}"
# Worktrees share the venv from the main checkout at /admin/home/$USER/nanopath.
VENV_DIR="${REPO_DIR}/.venv"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  VENV_DIR="${HOME}/nanopath/.venv"
fi
source "${VENV_DIR}/bin/activate"

# OpenSlide reads from /block (NFS) and JPEG writes go to /data; tiles are
# IO-bound, so oversubscribe workers past nproc to overlap waits.
export PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-16}"

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-local}"
echo "nproc=$(nproc)  PREPROCESS_WORKERS=${PREPROCESS_WORKERS}"

python preprocessing.py "${CONFIG_FILE}"
