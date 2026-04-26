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
- `probe.py`: inline downstream probes (cls KNN/SimpleShot/linear, pannuke MaskTransformer seg) run on the training GPU
- `thunder_adapter.py`: vendored MaskTransformer + multiclass dice loss
- `configs/small.yaml`: training config
- `submit/train.sbatch`: single SLURM launcher

## Running

Single GPU from an allocated node:

```bash
cd /admin/home/paul/nanopath
source /admin/home/paul/nanopath/.venv/bin/activate
python train.py configs/small.yaml
```

SLURM submit:

```bash
sbatch /admin/home/paul/nanopath/submit/train.sbatch
sbatch --job-name=nanopath-small /admin/home/paul/nanopath/submit/train.sbatch /admin/home/paul/nanopath/configs/small.yaml
```

The checked-in config is the current small EMA-probe comparison run: `small` model, `train.global_batch_size: 128`, `train.max_train_flops: 1000000000000000000`, validation every 500 steps, `train.warmdown_flop_fraction: 0.65`, `train.final_lr_frac: 0.05`, `train.ema_decay: 0.999`, `probe.model_weights: ema`, and `probe.count: 1` (set to 4 to probe at 25/50/75/100% of the FLOP budget). The default launcher requests 4 H100s (`NPROC_PER_NODE=4`) with `--time=04:00:00`. For paired comparisons, hold the seed fixed and compare `final_probe_score` plus the per-task `final_probe_linear_mean_f1` / `final_probe_knn_mean_f1` / `final_probe_fewshot_mean_f1` / `final_probe_seg_mean_jaccard` fields in `summary.json`.

Edit [train.sbatch](/admin/home/paul/nanopath/submit/train.sbatch) before submit only if you want to change the checked-in defaults:

- `NPROC_PER_NODE`
- `#SBATCH` resources if needed

## Outputs

- run outputs: `/data/nanopath/<family>/<project.name>`
- wandb: `/data/nanopath/wandb`
- sample-list cache: `/data/nanopath/cache`
- SLURM logs from `train.sbatch`: `/data/nanopath/slurm`
- Thunder probe scratch: `/tmp/nanopath-thunder`
- validated small screen baseline: `/data/nanopath/small/small-screen-bs24-seed-1337-20260424`

Fresh non-resume launches delete and recreate `project.output_dir` before training starts.
Training writes a single root checkpoint, `latest.pt`; later scheduled/final saves overwrite it and remove stale root `step_*.pt` checkpoints.
The learning rate schedule follows nanochat: linear warmup, constant LR, then linear warmdown to `train.final_lr_frac` over the last `train.warmdown_flop_fraction` of the FLOP budget.
Thunder probes run inside the same job and GPU allocation once training crosses evenly spaced FLOP milestones, using the EMA weights from the next validation checkpoint after each milestone and the final validation checkpoint at the end of the run. Training pauses while each probe runs, logs the probe metrics to wandb and `metrics.jsonl`, then resumes.
