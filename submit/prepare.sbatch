#!/usr/bin/env bash
#SBATCH --job-name=prepare
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

# CPU-node launcher for prepare.py. First arg is the YAML config (defaults to
# configs/leader.yaml); any further args are passed through to prepare.py, e.g.
#   sbatch submit/prepare.sbatch configs/leader.yaml download=True
# Reruns are resumable: HF caches the parquet shards by content hash and
# already-populated probe roots are left alone.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_FILE="${1:-${REPO_DIR}/configs/leader.yaml}"
shift || true  # drop the consumed config arg; remaining "$@" forward to prepare.py

cd "${REPO_DIR}"
# Worktrees share the venv from the main checkout at /admin/home/$USER/nanopath.
VENV_DIR="${REPO_DIR}/.venv"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  VENV_DIR="${HOME}/nanopath/.venv"
fi
source "${VENV_DIR}/bin/activate"

# prepare.py's HF download (and the regen-only SVS-decode path) are both
# IO-bound, so oversubscribe workers past nproc to overlap network/disk waits.
export PREPARE_WORKERS="${PREPARE_WORKERS:-16}"

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-local}"
echo "nproc=$(nproc)  PREPARE_WORKERS=${PREPARE_WORKERS}"

python prepare.py "${CONFIG_FILE}" "$@"
