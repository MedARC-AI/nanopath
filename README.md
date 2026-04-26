# NanoPath

Lean JEPA pathology pretraining harness. Designed for fast hacking on a
single H100: tweak a recipe, run a smoke, run the full small recipe, and
compare `final_probe_score` against the leaderboard.

The downstream probes (`probe.py`) are the comparison signal — JEPA-style
val losses across recipes aren't directly comparable, so we always rank by
mean F1 across five classification probes (bach, bracs, break_his, mhist,
pcam) plus pannuke segmentation Jaccard.

## Quickstart

```bash
git clone <repo> nanopath && cd nanopath
uv sync && source .venv/bin/activate
python download_data.py                      # ~XX GB, probe datasets only
python train.py configs/smoke.yaml           # ~5 min end-to-end smoke on 1 GPU
```

That's enough to verify your env and then `sbatch submit/train_4gpu.sbatch`
a real run.

## Layout

- `train.py` — pretraining loop (DDP via torchrun, JEPA + SIGReg, EMA, inline probe dispatch).
- `model.py` — `NanoPathFM` ViT backbone + `SIGReg`. Hack here for new objectives.
- `dataloader.py` — TCGA sample-list streaming loader. Hack here for new pretraining data.
- `probe.py` — inline downstream probes (KNN, SimpleShot, linear, pannuke MaskTransformer seg). Hack here for new probes.
- `seg_head.py` — vendored `MaskTransformer` + `multiclass_dice_loss` (used only by `probe.py`'s pannuke segmentation).
- `download_data.py` — auto-downloads the six probe datasets if missing.
- `data_splits/` — checked-in classification splits so probes work out of the box.
- `configs/{smoke,small}.yaml` — smoke is a few-minute sanity run; small is the leaderboard recipe.
- `submit/train_{1,4}gpu.sbatch` — SLURM launchers. The 4-GPU one is the canonical full run.

## Data

- **TCGA pretraining list** (`/block/TCGA/sample_dataset_30.txt`): obtained
  from the internal source — we **do not** auto-download. `train.py`
  errors loudly at config-load if it's missing.
- **Probe datasets** (bach / bracs / break_his / mhist / pcam / pannuke):
  `python download_data.py` pulls each one to its `DATASET_ROOTS[...]` path
  if not already present. mhist requires a one-time form access; the script
  prints instructions. break_his requires Kaggle credentials.
- **Classification split JSONs**: shipped under `data_splits/` in this repo;
  no Thunder install needed.

## Running

Smoke (single GPU, ~5 min, validates the full train+probe path):

```bash
python train.py configs/smoke.yaml
```

Full small recipe (4 H100s, ~1h, hits the leaderboard):

```bash
sbatch submit/train_4gpu.sbatch                       # default: configs/small.yaml
sbatch submit/train_4gpu.sbatch configs/your_recipe.yaml
sbatch submit/train_1gpu.sbatch configs/smoke.yaml    # single-GPU sweeps
```

Edit the `#SBATCH` lines or pass `sbatch --gpus-per-task=N --time=...` to
override resources. `submit/train_4gpu.sbatch` accepts an optional first
argument to point at a different config.

## Recipe summary

The checked-in `configs/small.yaml` is the current leaderboard run:
`small` model, `train.global_batch_size: 128`, `train.max_train_flops: 1e18`,
validation every 500 steps, `train.warmdown_flop_fraction: 0.65`,
`train.final_lr_frac: 0.05`, `train.ema_decay: 0.999`, `probe.model_weights: ema`,
`probe.count: 1` (set to 4 to probe at 25/50/75/100% of the FLOP budget).

The LR schedule follows nanochat: linear warmup → constant LR → linear
warmdown to `final_lr_frac` over the last `warmdown_flop_fraction` of the
FLOP budget.

For paired comparisons, hold the seed fixed and compare `final_probe_score`
plus the per-task `final_probe_linear_mean_f1` / `final_probe_knn_mean_f1` /
`final_probe_fewshot_mean_f1` / `final_probe_seg_mean_jaccard` from
`summary.json`. Treat anything below a 0.02 mean-F1 delta as noise.

## Outputs

- run outputs: `/data/nanopath/<family>/<project.name>` (wiped on fresh launch).
- wandb: `/data/nanopath/wandb`.
- sample-list cache: `/data/nanopath/cache`.
- SLURM logs: `/data/nanopath/slurm/<jobid>.{out,err}`.

Training writes a single rolling `latest.pt`; the best probe checkpoint is
also kept as `best_mean_probe_score.pt`. Probes run inline in the same job
once training crosses evenly spaced FLOP milestones (using EMA weights from
the next validation checkpoint), pause training while they run, log into
wandb + `metrics.jsonl`, then resume.
