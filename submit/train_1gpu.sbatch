#!/usr/bin/env bash
#SBATCH --job-name=leader_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --partition=n
#SBATCH --account=sophont
#SBATCH --qos=normal # low | normal | top
#SBATCH --signal=B:USR1@900
set -euo pipefail

# Single-GPU SLURM launcher. Optional first arg overrides the YAML config
# (defaults to configs/leader.yaml). Any extra args after that are forwarded
# to train.py as overrides, e.g.
#   sbatch submit/train_1gpu.sbatch configs/leader.yaml output_dir=/data/$USER/nanopath/leader/myrun
# Logs land in repo-local slurm/<jobid>.{out,err}.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_FILE="${1:-${REPO_DIR}/configs/leader.yaml}"
[[ $# -gt 0 ]] && shift

cd "${REPO_DIR}"
# Worktrees share the venv from the main checkout at /admin/home/$USER/nanopath.
VENV_DIR="${REPO_DIR}/.venv"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  VENV_DIR="${HOME}/nanopath/.venv"
fi
source "${VENV_DIR}/bin/activate"

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-local}"

# SLURM sends USR1 to this batch shell; forward it to the python child so
# train.py saves, probes, and exits cleanly before wall-clock expiry.
python train.py "${CONFIG_FILE}" "$@" &
PY_PID=$!
trap 'kill -USR1 "$PY_PID" 2>/dev/null || true' USR1
while true; do
  wait "$PY_PID" && break
  status=$?
  kill -0 "$PY_PID" 2>/dev/null || exit "$status"
done
