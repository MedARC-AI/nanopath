#!/usr/bin/env bash
#SBATCH --job-name=prepare_chimera
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

# CPU-node launcher for the chimera tile cache. Two-stage:
#   1) `aws s3 sync` pulls 3A + 3B HE.tif + HE_mask.tif + CD.json from the public chimera-challenge
#      bucket into /data/chimera/raw (~350 GB, ~1 hr at typical bandwidth).
#   2) prepare_chimera_tiles extracts CHIMERA_TILES_PER_SLIDE foreground tiles per slide and
#      writes a single parquet shard at /data/chimera/chimera_tiles/data/chimera-00000.parquet
#      (~1.4 GB, ~10 min on this CPU node).
# Reruns of step 1 are resumable via aws s3 sync. Step 2 always rewrites the shard.
RAW_DIR="/data/chimera/raw"
TILE_DIR="/data/chimera/chimera_tiles"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"
VENV_DIR="${REPO_DIR}/.venv"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  VENV_DIR="${HOME}/nanopath/.venv"
fi
source "${VENV_DIR}/bin/activate"

echo "RAW_DIR=${RAW_DIR}"
echo "TILE_DIR=${TILE_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"

python -c "
from pathlib import Path
from prepare import fetch_chimera_raw, prepare_chimera_tiles
fetch_chimera_raw(Path('${RAW_DIR}'))
prepare_chimera_tiles(Path('${RAW_DIR}'), Path('${TILE_DIR}'))
"
