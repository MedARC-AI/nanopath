# NanoPath

Lean JEPA pathology pretraining code.

## Setup

```bash
cd /admin/home/paul/nanopath
uv sync
source /admin/home/paul/nanopath/.venv/bin/activate
mkdir -p /data/nanopath
```

## Required data

- training sample list: `/block/TCGA/sample_dataset_30.txt`
- Thunder probe datasets:
  - `/block/eva-data/bach`
  - `/block/eva-data/bracs`
  - `/block/eva-data/breakhis`
  - `/block/eva-data/mhist`
  - `/block/eva-data/patch_camelyon`
- Thunder repo: `/admin/home/paul/thunder`

## Main files

- `train.py`: training loop
- `model.py`: model and projector stack
- `dataloader.py`: TCGA sample-list loader
- `probe.py`: async Thunder linear probes
- `thunder_adapter.py`: Thunder checkpoint adapter
- `configs/*.yaml`: training configs
- `submit/train.sbatch`: single SLURM launcher

## Running

Single GPU from an allocated node:

```bash
cd /admin/home/paul/nanopath
source /admin/home/paul/nanopath/.venv/bin/activate
python train.py configs/nano.yaml
```

SLURM submit:

```bash
sbatch /admin/home/paul/nanopath/submit/train.sbatch
```

Edit [train.sbatch](/admin/home/paul/nanopath/submit/train.sbatch) before submit:

- `CONFIG_FILE`
- `NPROC_PER_NODE`
- `#SBATCH` resources if needed

## Outputs

- run outputs: `/data/nanopath/<family>/<project.name>`
- wandb: `/data/nanopath/wandb`
- sample-list cache: `/data/nanopath/cache`
- SLURM logs from `train.sbatch`: `/data/nanopath/slurm`
- Thunder scratch: `/tmp/nanopath-thunder`
- checked-in nano baseline: `/data/nanopath/nano/new-baseline-only-jepa`

Fresh non-resume launches delete and recreate `project.output_dir` before training starts.
Thunder probes are launched asynchronously from eval steps and continue chaining after the training job exits; monitor `project.output_dir/thunder/state.json` plus the Thunder SLURM logs for pending probe progress.
