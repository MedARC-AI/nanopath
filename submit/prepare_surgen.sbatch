#!/usr/bin/env bash
#SBATCH --job-name=prepare_surgen
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

# CPU-node launcher for the SurGen extended-RAS tile cache. Stream pipeline:
#   for each of ~398 cases (KRAS or NRAS labelable, balanced ~48% positive), 8 workers
#   download one ~4 GB .czi from s3://path-datasets/SurGen/SR1482_WSIs, sample 256 random
#   tissue tiles via aicspylibczi mosaic reads, JPEG-encode, and DELETE the .czi before
#   moving on. Net transient peak ≈ 8 × 4 GB = ~32 GB on /data; permanent output is
#   /data/surgen/surgen_tiles/data/surgen-00000.parquet (~3-4 GB) + labels.csv.
RAW_DIR="/data/surgen/raw"
TILE_DIR="/data/surgen/surgen_tiles"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"
VENV_DIR="${REPO_DIR}/.venv"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  VENV_DIR="${HOME}/nanopath/.venv"
fi
source "${VENV_DIR}/bin/activate"

# path-datasets bucket lives on cloudflare R2; export the credentials from ~/.bashrc.
export AWS_ENDPOINT_URL=https://d367d26c39a5adf066cf2612d594f2c9.r2.cloudflarestorage.com
export AWS_ACCESS_KEY_ID="2e09c2091e6c62f62f46ff7512c5ee54"
export AWS_SECRET_ACCESS_KEY="c1b16d619c3181bebce313ae9af5b55f622f7b89363fd6c93a3a64a71af9add8"
export AWS_REGION="auto"

echo "RAW_DIR=${RAW_DIR}"
echo "TILE_DIR=${TILE_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"

python -c "
from pathlib import Path
from prepare import prepare_surgen_tiles
prepare_surgen_tiles(Path('${RAW_DIR}'), Path('${TILE_DIR}'))
"
