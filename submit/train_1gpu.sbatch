#!/usr/bin/env bash
#SBATCH --job-name=main_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --partition=n
#SBATCH --qos=normal # low | normal | top

set -euo pipefail

# Single-GPU SLURM launcher. Run it directly so the login node can collect
# Labless metadata before scheduling the noninteractive GPU job, e.g.
#   ./submit/train_1gpu.sbatch configs/main.yaml output_dir=/data/$USER/nanopath/main/myrun
# Plain `sbatch submit/train_1gpu.sbatch ...` still trains, but cannot auto-submit.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_FILE="${1:-${REPO_DIR}/configs/main.yaml}"
[[ $# -gt 0 ]] && shift

cd "${REPO_DIR}"
# Worktrees share the venv from the main checkout at /admin/home/$USER/nanopath.
VENV_DIR="${REPO_DIR}/.venv"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  VENV_DIR="${HOME}/nanopath/.venv"
fi
source "${VENV_DIR}/bin/activate"
mkdir -p slurm

LAUNCH_INFO="$(python - "${CONFIG_FILE}" "$@" <<'PY'
import json, os, sys
from pathlib import Path
import yaml

cfg = yaml.safe_load(os.path.expandvars(Path(sys.argv[1]).read_text()))
for arg in sys.argv[2:]:
    key, _, value = arg.partition("=")
    if key == "output_dir":
        cfg["project"]["output_dir"] = os.path.expandvars(value)
output_dir = str(Path(cfg["project"]["output_dir"]).expanduser().resolve())
eligible = (
    bool(cfg["probe"]["enabled"])
    and int(cfg["probe"]["count"]) > 0
    and int(cfg["train"]["max_train_samples"]) == 1_000_000
    and int(cfg["train"]["max_train_flops"]) == 1_000_000_000_000_000_000
)
print(json.dumps({"output_dir": output_dir, "eligible": eligible}))
PY
)"
OUTPUT_DIR="$(python -c 'import json, sys; print(json.load(sys.stdin)["output_dir"])' <<<"${LAUNCH_INFO}")"
LABLESS_ELIGIBLE="$(python -c 'import json, sys; print("1" if json.load(sys.stdin)["eligible"] else "0")' <<<"${LAUNCH_INFO}")"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  AUTOSUBMIT_FILE=""
  if ! command -v sbatch >/dev/null; then
    echo "sbatch not found; run python train.py on a GPU machine or adapt submit/train_1gpu.sbatch for your scheduler." >&2
    exit 127
  fi
  if [[ "${LABLESS_ELIGIBLE}" == "1" && -t 0 ]]; then
    echo "This looks like a full Labless-eligible run. Leave either prompt blank to train without auto-submit."
    read -r -p "Labless run name (<=20 chars): " LABLESS_RUN_NAME || LABLESS_RUN_NAME=""
    read -r -p "Labless notes: " LABLESS_NOTES || LABLESS_NOTES=""
    if [[ -n "${LABLESS_RUN_NAME}" && -n "${LABLESS_NOTES}" && "${#LABLESS_RUN_NAME}" -le 20 ]]; then
      AUTOSUBMIT_FILE="${OUTPUT_DIR}.labless_autosubmit.json"
      set +e
      python labless/submit_to_labless.py login_only=true token_output="${AUTOSUBMIT_FILE}" run_name="${LABLESS_RUN_NAME}" notes="${LABLESS_NOTES}"
      LOGIN_STATUS=$?
      set -e
      if [[ "${LOGIN_STATUS}" != "0" ]]; then
        echo "Labless login did not complete; training will run without auto-submit."
        rm -f "${AUTOSUBMIT_FILE}"
        AUTOSUBMIT_FILE=""
      fi
    else
      echo "Labless auto-submit skipped; run name and notes are required, and run name must be <=20 chars."
    fi
  elif [[ "${LABLESS_ELIGIBLE}" == "1" ]]; then
    echo "No interactive stdin; training will run without Labless auto-submit."
  fi
  if [[ -n "${AUTOSUBMIT_FILE}" ]]; then
    set +e
    sbatch --export=ALL,LABLESS_AUTOSUBMIT_FILE="${AUTOSUBMIT_FILE}" "$0" "${CONFIG_FILE}" "$@"
    SBATCH_STATUS=$?
    set -e
    if [[ "${SBATCH_STATUS}" != "0" ]]; then
      rm -f "${AUTOSUBMIT_FILE}"
    fi
    exit "${SBATCH_STATUS}"
  fi
  exec sbatch "$0" "${CONFIG_FILE}" "$@"
fi

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-local}"
cleanup_labless_token() {
  if [[ -n "${LABLESS_AUTOSUBMIT_FILE:-}" && -f "${LABLESS_AUTOSUBMIT_FILE}" ]]; then
    rm -f "${LABLESS_AUTOSUBMIT_FILE}"
  fi
}
trap cleanup_labless_token EXIT
if [[ "${LABLESS_ELIGIBLE}" == "1" && -z "${LABLESS_AUTOSUBMIT_FILE:-}" ]]; then
  echo "Labless auto-submit is not armed. Run ./submit/train_1gpu.sbatch from the login node to enable the pre-run prompt."
fi

python train.py "${CONFIG_FILE}" "$@"
