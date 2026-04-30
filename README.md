# nanopath

![nanopath logo](nanopath_logo.png)

`nanopath` is a super lean experimental harness for training tile-level computational pathology foundation models, inspired by [nanochat](https://github.com/karpathy/nanochat). It runs on a single GPU (can also be run multi-gpu if you want faster, identical results), the code is minimal/hackable, and covers the full pretraining pipeline using the public TCGA dataset (12k WSIs) and built-in probe evals from the [Thunder benchmark](https://mics-lab.github.io/thunder/). 

This repository is intentionally made to be compatible with [autoresearch](https://github.com/karpathy/autoresearch)-style pursuits. We will continuously update our codebase and [Leaderboard](#leaderboard) to reflect the best performing model. The current leaderboard winner (`configs/leader.yaml`) takes ~1 hour on one H100 GPU (this includes the six downstream probes evaluated at the end of the same job).

**Want to get involved? Join us in the [MedARC Discord](https://discord.gg/tVR4TWnRM9) (find us in #path-fm)!**

## Quickstart

Install [uv](https://docs.astral.sh/uv/) first if you don't have it, then:

```bash
git clone https://github.com/MedARC-AI/nanopath.git && cd nanopath
uv sync && source .venv/bin/activate
wandb login

# first-time setup to verify pretrain + probe datasets exist
python prepare.py configs/smoke.yaml download=False   

# "smoke test": train + eval model in <10 min to check everything works
sbatch submit/train_1gpu.sbatch configs/smoke.yaml
# or directly: python train.py configs/smoke.yaml

# train and evaluate current first place model in the leaderboard 
sbatch submit/train_1gpu.sbatch configs/leader.yaml
# or directly: python train.py configs/leader.yaml
```

If you are a MedARC volunteer on our shared cluster, the checked-in configs already point at `/data/nanopath_parquet` for the tile shards and `/block/{eva-data,thunder-data}/<name>` for the probe datasets. `python prepare.py configs/leader.yaml download=False` should print all `[skip]` lines. 

For non-MedARC cluster users, run `python prepare.py configs/leader.yaml download=True` to download our 4M-tile dataset (200 parquet shards, ~120 GB) from the [`medarc/nanopath`](https://huggingface.co/datasets/medarc/nanopath) HF mirror and pull each probe dataset from its public source. That is the entire data setup; you do not need the original TCGA SVS files to train.

`pyproject.toml` pins `torch` / `torchvision` against the CUDA 12.9 wheel index. If your GPU/driver needs a different CUDA build (e.g. cu118 for older A100/V100 setups), edit the `torch` and `torchvision` lines in `pyproject.toml` before `uv sync`.

A successful model training prints periodic train/val lines, logs to wandb, and ends with final summary in `metrics.jsonl`. `configs/smoke.yaml` probe scores will be near-random since it is undertrained — use `configs/leader.yaml` for full runs.

## Leaderboard

Score is final `mean_probe_score`: unweighted mean of standard classification probe F1 aggregates (bach, bracs, break_his, mhist, pcam) and pannuke segmentation Jaccard. All metrics are computed on each dataset's validation split only; train splits solely fit the probe heads on top of the frozen backbone.

| # | mean | linear | KNN | few-shot | seg Jaccard | Description | wandb | Date | Contributors |
|---|------:|-------:|----:|---------:|------------:|-------------|-------|------|--------------|
| 1 | **0.5228** | 0.6832 | 0.6093 | 0.4490 | 0.3496 | LeJEPA baseline | [t72j3r8k](https://wandb.ai/paulscotti/nanopath/runs/t72j3r8k) | Apr 26 2026 | @PaulScotti |

### How to submit to leaderboard

The current `configs/leader.yaml` is the top performing leaderboard recipe. To get on the leaderboard you must outperform the existing top leaderboard `mean_probe_score` by at least 0.01. If you do so, open a PR to this repo with a description of your changes (please keep only the minimal necessary code changes that improve performance) and share your wandb run/report. [@PaulScotti](https://github.com/PaulScotti) will train a new model using your code on his H100 but with a different rng seed, while striving to reduce the submission to the smallest practical diff against the current codebase. If it still improves `mean_probe_score` by at least 0.01, we will update the README & leaderboard accordingly. **You don't need an H100 yourself to submit** — train on whatever hardware you have access to, share the run if you think it's a winner, and Paul handles H100 verification.

### What you must NOT change for a leaderboard submission

To keep entries comparable, the following are fixed across all submissions. Anything else (model architecture, training objective, optimizer, schedule shape, augmentation policy, EMA decay, masking, predictor design, dataset curation, using a pretrained image model ckpt, etc.) is fair game.

**Compute budget**
- `train.max_train_flops` (1e18). Compute budget is fixed to facilitate direct comparisons across training approaches; you can't buy higher score with more compute.

**Wall-clock budget**
- End-to-end `train.py` (training + inline probes) must complete in **≤3 hours on a single 80 GB H100**. Memory tricks like `train.activation_checkpointing: true` are fair game. `prepare.py`'s one-time data prep is excluded. We will verify any submissions using our own H100 hardware, so you do not yourself need to test on an H100.

**Activated parameter count**
- **≤150M activated backbone params**, where "backbone" is everything in `NanoPathFM` except `self.projector` (the projector is pretraining-only scaffolding and is discarded for downstream probes).
- For MoE / sparse architecture explorations, count parameters touched on a single token's forward pass.
- `train.py` already computes `backbone_activated_params` for easy verification.

**TCGA pretraining**
- TCGA (12K WSIs) is the only dataset allowed for pretraining, but you are free to revise how we select the tiles used for training.
- The six probe datasets (bach, bracs, break_his, mhist, pcam, pannuke) are eval-only. Their images, labels, and splits cannot be observed by the pretraining model — neither directly (training data) nor indirectly (distillation target, contrastive negatives, label-smoothing prior, etc.).

**Probe evaluation**
- All of `probe.py` and `seg_head.py`.
- `probe_data_splits/` — the checked-in classification splits.
- All probe config variables in `configs/leader.yaml`.

## Repository layout

### Primary files meant to be hacked
- `train.py` — the main pretraining loop (DDP via torchrun, JEPA + SIGReg, EMA, probe dispatch, wandb logging).
- `model.py` — `NanoPathFM` ViT backbone. Hack here for new model architectures / training objectives.
- `dataloader.py` — TCGA tile loader (parquet shards via pyarrow, mmap'd) and augmentation stack. Hack here for crop/color/HED augmentation tweaks.
- `configs/{smoke,leader}.yaml` — recipes (model shape, optimizer, schedule, augmentation knobs, probe config).

### Helper files
- `AGENTS.md` — guidelines for AI assistants and human contributors: design philosophy (minimal/hackable, nanochat-flavored), coding rules, experiment discipline, and cluster/storage conventions. Note some language is specific to the MedARC cluster.
- `prepare.py` — data prep (verify or download HF mirror + probe datasets); also hosts the SVS-decode + parquet-pack helpers for [regenerating the tile dataset from raw SVS](#regenerating-the-tile-dataset-from-raw-svs).
- `probe.py` — downstream probes (KNN, few shot, linear, segmentation).
- `submit/{prepare,train_1gpu,train_4gpu}.sbatch` — SLURM launchers (CPU node for prepare, GPU node for training).
- `seg_head.py` — `MaskTransformer` + `multiclass_dice_loss` (used by `probe.py`'s pannuke segmentation), vendored by Thunder.
- `probe_data_splits/` — checked-in classification splits for probes.
- `download_TCGA.sh` — manual utility, run by hand if you want the full 12K TCGA open-access SVS slide set (~13 TB) for forking the tile-extraction recipe. Not invoked by `prepare.py` and not needed for any standard training workflow.
- `LOG.md` — running notes on what has been tried, including negative results.
- `pyproject.toml` + `uv.lock` — Python dependency spec consumed by `uv sync`.

## Data

`prepare.py` prepares the necessary data for pretraining. Flag `download=True` to download all datasets to the folders specified by the .yaml, or flag `download=False` to simply verify if all datasets are already present.

Edit `data.dataset_dir` and every `probe.dataset_roots.*` in your config (`configs/leader.yaml` and `configs/smoke.yaml` if you also smoke-test) to your own correct paths.

```bash
# Pull the 4M-tile parquet dataset from the medarc/nanopath HF mirror into
# data.dataset_dir, and pull each missing probe dataset from its public
# source into the configured probe.dataset_roots paths.
python prepare.py configs/leader.yaml download=True

# Verify-only: confirms the parquet shards and every probe dataset listed
# in the config exist on disk where the YAML says they should.
python prepare.py configs/leader.yaml download=False
```

**What `download=True` does**
1. **TCGA tiles**: `huggingface_hub.snapshot_download` (filtered to `shard-*.parquet`) pulls the 200 parquet shards (~120 GB total, `{path: string, jpeg: binary}` rows with 64-row row groups) from [`medarc/nanopath`](https://huggingface.co/datasets/medarc/nanopath) into `data.dataset_dir`.
2. **Probe datasets** (bach / bracs / break_his / mhist / pcam / pannuke): for each empty root, fetches + unpacks the dataset from its public source into the configured path.

**Prerequisites**
- ~120 GB free wherever `data.dataset_dir` lives for the parquet shards (cluster default: `/data/nanopath_parquet`).
- ~30 GB free in total across the six `probe.dataset_roots` entries (pannuke ~13 GB + bach ~10 GB are the bulk; the rest are smaller).
- mhist requires a one-time form at https://bmirds.github.io/MHIST/. Drop the resulting `annotations.csv` + `images.zip` into `probe.dataset_roots.mhist`, then rerun `prepare.py … download=True` to unpack.

### Regenerating the tile dataset from raw SVS

`prepare.py` itself never touches raw SVS files — it always pulls the ready-made parquet shards from HF. If you want, however, you can download the full ~13 TB original SVS files from TCGA and pre-extract different tiles to pretrain on. Two-step workflow (decode SVS → JPEG dir + manifest, then pack into parquet shards):

```bash
# 1) Download the full 12K open-access TCGA SVS slide set (~13 TB).
bash download_TCGA.sh /data/TCGA 8

# 2) Decode + pack. prepare_tiles deterministically subsamples the sample list
#    to TARGET_TILE_COUNT (4M, hardcoded in prepare.py — bump it for a bigger
#    dataset) and writes JPEGs + manifest.txt under jpeg_dir; reruns are
#    resumable (existing JPEGs are EOF-validated and reused). pack_from_jpeg_dir
#    then walks the manifest, splits into NUM_SHARDS=200 chunks, and writes
#    shard-NNNNN.parquet files with 64-row row groups (the layout the
#    dataloader expects). Once it's done you can rm -rf the jpeg_dir.
python -c "
from pathlib import Path
from prepare import prepare_tiles, pack_from_jpeg_dir
jpeg_dir = Path('/data/nanopath_jpegs_tmp')
prepare_tiles(Path('/data/TCGA/sample_dataset_30.txt'), jpeg_dir, split_seed=42)
pack_from_jpeg_dir(jpeg_dir, jpeg_dir / 'manifest.txt', Path('/data/nanopath_parquet'))
"
```

Once `data.dataset_dir` contains `shard-*.parquet`, `prepare.py … download=False` will print `[skip] tiles` and only fetch the probe datasets. To publish a new variant of the dataset, push the resulting shards to a fresh HF dataset repo and update `HF_REPO_ID` in `prepare.py`.

## Running

Smoke (single GPU, ~8 min, validates the full train+probe path):

```bash
sbatch submit/train_1gpu.sbatch configs/smoke.yaml
# or directly: `python train.py configs/smoke.yaml`
```

Leader (full train+probe for the first place recipe in our Leaderboard)

```bash
sbatch submit/train_1gpu.sbatch configs/leader.yaml
# or directly: `python train.py configs/leader.yaml`
# can alternatively do `submit/train_4gpu.sbatch` for faster training
```

`configs/leader.yaml` is sized for an 80 GB H100 at `train.global_batch_size: 128`. On smaller cards you can set `train.activation_checkpointing: true` if you OOM. Smoke fits comfortably on any 24 GB+ GPU.

## Outputs

- run outputs: `project.output_dir` (default is `/data/$USER/nanopath/leader/...`). Final probe results log to `metrics.jsonl`.
- wandb: `/data/$USER/nanopath/wandb`.
- parquet tile shards: `data.dataset_dir` (defaults to `/data/nanopath_parquet`).
- probe datasets: `probe.dataset_roots` (defaults to `/block/{eva-data,thunder-data}/<name>`).
- SLURM logs: `slurm/<jobid>.{out,err}` in the repo.
- checkpoints: rolling `latest.pt` written every `train.save_every` steps under `project.output_dir`; smoke leaves none because `save_every: null`.

### Auto-resume / SLURM requeuing after a kill

`submit/train_*.sbatch` set `--requeue`, so SLURM auto-resubmits the job on preemption / walltime / `scancel`. The requeued job sees an existing `output_dir/latest.pt` and resumes from that checkpoint, same wandb run id, same step counter, same optimizer + EMA state, no `train.resume` config edit. You'll lose at most `train.save_every` steps of progress.

To start a run completely fresh instead, either delete the run's `project.output_dir` or change `project.output_dir` to a new path before launching. The `train.resume` config field still works as an explicit override (e.g. resuming from a different run's checkpoint) and takes priority over the auto-detect.

## Experiment log

See [LOG.md](LOG.md) for running notes on what has been tried in nanopath. Negative results included! Such logs help contributors avoid retrying dead ends.

## Acknowledgements

Inspired by [nanochat](https://github.com/karpathy/nanochat). Probe code, dataset splits, and the pannuke `MaskTransformer` head are adapted from the [Thunder benchmark](https://mics-lab.github.io/thunder/).
