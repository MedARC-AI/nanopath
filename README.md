# nanopath

![nanopath logo](imgs/nanopath_logo.png)

`nanopath` is a super lean experimental harness for training tile-level computational pathology foundation models, inspired by [nanochat](https://github.com/karpathy/nanochat). It runs on a single GPU, the code is minimal/hackable, and covers the full pretraining pipeline using the public TCGA dataset (12k WSIs) and built-in downstream probes spanning classification, segmentation, slide-level mutation/response, survival, and robustness.

This repository is intentionally made to be compatible with [autoresearch](https://github.com/karpathy/autoresearch)-style pursuits. We will continuously update our codebase and [Leaderboard](#leaderboard) to reflect the best performing model. The maintained training recipe (`configs/leader.yaml`) targets a single-H100 run with the 12-dataset downstream probe suite evaluated at the end; after the uncapped PathoBench-style retile, the final probe wall is about 31 minutes for DINOv2-S and about 50 minutes for ViT-G baselines.

**Want to get involved? Join us in the [MedARC Discord](https://discord.gg/tVR4TWnRM9) (find us in #path-fm)!**

## Quickstart

Install [uv](https://docs.astral.sh/uv/) first if you don't have it, then:

```bash
git clone https://github.com/MedARC-AI/nanopath.git && cd nanopath
uv sync && source .venv/bin/activate
wandb login

# MedARC cluster: verify the shared parquet/probe roots + DINOv2 weights
python prepare.py configs/smoke.yaml download=False
# Off-cluster: edit dataset paths in the YAML first, then fetch data + weights.
# Some raw probe sources are large; MHIST/CoNSeP/SurGen use our HF probe mirror
# and print upstream access-term/provenance notices before fetching.
# python prepare.py configs/smoke.yaml download=True

# smoke test: short training plus the full configured probe suite
sbatch submit/train_1gpu.sbatch configs/smoke.yaml
# or directly on a GPU machine: python train.py configs/smoke.yaml

# train and evaluate the maintained Nanopath recipe
sbatch submit/train_1gpu.sbatch configs/leader.yaml
# or directly on a GPU machine: python train.py configs/leader.yaml
```

If you are a MedARC volunteer on our shared cluster, the checked-in configs already point at `/data/nanopath_parquet` for the tile shards and the shared `/block/...` and `/data/...` roots for the probe datasets.

For non-MedARC cluster users, run `python prepare.py configs/leader.yaml download=True` to download our 4M-tile dataset (200 parquet shards, ~120 GB) from our [nanopath HF dataset](https://huggingface.co/datasets/medarc/nanopath), fetch/prepare the configured probe datasets, and fetch relevant pretrained weights for the configured `model.type`. You do not need the original TCGA SVS files to train.

`pyproject.toml` pins `torch` / `torchvision` against the CUDA 12.9 wheel index. If your GPU/driver needs a different CUDA build (e.g. cu118 for older A100/V100 setups), edit the `torch` and `torchvision` lines in `pyproject.toml` before `uv sync`.

A successful model training prints periodic train lines, logs to wandb, and ends with a final summary in `metrics.jsonl`. `configs/smoke.yaml` is simply meant to train briefly and then exercise the same downstream probe machinery as full runs; use `configs/leader.yaml` for leaderboard-scale runs.

## Leaderboard

![Nanopath progress plot](imgs/progress_plot.png)

Score is final `mean_probe_score`: the unweighted mean of the 12 dataset columns below. Tile classification datasets use the mean of linear / KNN / SimpleShot F1; PathoBench-derived slide classification datasets use balanced logistic linear-probe AUROC; segmentation datasets use macro Jaccard; SurGen uses AUROC; CRC survival uses Harrell's c-index; and PathoROB uses its robustness index. Historical six-probe rows were removed because they are not comparable to the current expanded benchmark. See `benchmarking/` for the full benchmark notes and test-split policy.

| # | mean | break_his | bracs | mhist | pcam | monusac | consep | pannuke | her2 | ucla_lung | surgen | crc_survival | pathorob | Description | wandb | Date | Contributors |
|---|-----:|----------:|------:|------:|-----:|--------:|-------:|--------:|-----:|----------:|-------:|-------------:|---------:|-------------|-------|------|--------------|
| 1 | **0.6209** | 0.7490 | 0.5534 | 0.7516 | 0.8830 | 0.3349 | 0.2218 | 0.4153 | 0.7692 | 0.7004 | 0.6487 | 0.5313 | 0.8926 | Untouched H-optimus-0 ViT-G/14-reg baseline (`baselines/hoptimus0_baseline.py`) | n/a | May 9 2026 | Bioptimus |
| 2 | 0.5681 | 0.5241 | 0.4867 | 0.7566 | 0.7888 | 0.2994 | 0.2255 | 0.4005 | 0.7348 | 0.6993 | 0.5990 | 0.5583 | 0.7438 | Untouched OpenMidnight ViT-G/14-reg baseline (`baselines/openmidnight_baseline.py`) | n/a | May 9 2026 | @PaulScotti |
| 3 | 0.5454 | 0.4967 | 0.4942 | 0.7610 | 0.7836 | 0.2367 | 0.2042 | 0.3607 | 0.7407 | 0.5827 | 0.6079 | 0.5221 | 0.7543 | Untouched Meta `dinov2_vits14_reg` baseline (`baselines/dinov2_small_baseline.py`) | n/a | May 9 2026 | @tmabraham |

### How to submit to the leaderboard

The current `configs/leader.yaml` is the maintained Nanopath training recipe, but its old six-probe score has been removed until it is re-run on this 12-probe benchmark. To get on the leaderboard you must outperform the existing top leaderboard `mean_probe_score` by at least 0.01. If you do so, open a PR to this repo with a description of your changes (please keep only the minimal necessary code changes that improve performance) and share your wandb run/report. [@PaulScotti](https://github.com/PaulScotti) will train a new model using your code on his 1 80GB H100, using a different rng seed and striving to reduce the submission to the smallest practical diff against the current codebase. If it still improves `mean_probe_score` by at least 0.01, we will update the README & leaderboard accordingly. **You don't need an H100 yourself to submit** — train on whatever hardware you have access to, share the run if you think it's a winner, and Paul handles H100 verification.

We also strongly welcome PRs that simplify the codebase — either by reducing lines of code (excluding commented-out lines intended for readability) or by reducing complexity (e.g. replacing the cosine LR scheduler with a constant LR) — without regressing `mean_probe_score`.

### What you must NOT change for a leaderboard submission

Anything not explicitly fixed below (e.g., model architecture, training objective, optimizer, lr scheduler, data augmentations, masking, dataset curation) is fair game for modification.

**Training ends at 1e18 total FLOPs OR after 45 min. elapsed on 1xH100**

Every leaderboard run is verified on the organizer's compute (1 80GB H100 gpu), bounded by two possible caps:

- **`train.max_train_flops` ≤ 1e18 training FLOPs**, measured directly from aten op shapes via `torch.utils.flop_counter.FlopCounterMode` on the first step (forward + backward + opt.step) and reused thereafter since per-step shapes are fixed. This counts everything that touches the GPU during a step — student backbone, EMA teacher forward, projection heads, masking, etc. — not just the backbone.
- **≤45 min. training on a single 80 GB H100 before the final probe window**, enforced by SLURM. `submit/train_1gpu.sbatch` runs with `--signal=USR1@900`, so SLURM sends `SIGUSR1` 15 minutes before the `--time` wall; `train.py`'s SIGUSR1 handler catches it as a clean stop signal, cuts training, and uses the remaining window for the final checkpoint save + downstream probe suite. The current full-grid final probe is ~31 minutes for DINOv2-S and ~50 minutes for ViT-G baselines, dominated by SurGen.

The above limits force submissions to be **simultaneously compute efficient and systems efficient**.

**TCGA as the only pretraining data**
- TCGA (12K WSIs) is the only dataset allowed for pretraining, but you are free to revise how we select the tiles used for training.
- The probe datasets cannot be used for pretraining, neither directly (training data) nor indirectly (distillation target, contrastive negatives, label-smoothing prior, etc.).

**Probe evaluation must be untouched**
- All of `probe.py`.
- `benchmarking/` — checked-in downstream split metadata.
- All probe config variables in `configs/leader.yaml`.

**Initializing model from a pretrained ckpt is OK only if not pathology-specific**
You can initialize the model using DINOv2 checkpoint (trained on natural images) but you can't initialize from, say, H-optimus or OpenMidnight checkpoints. We want to train a pathology foundation model so we shouldn't offload most of the training to someone else's pathology-specific model.

## Repository layout

### Primary files meant to be hacked
- `train.py` — main pretraining loop
- `model.py` — model architecture and training objectives
- `dataloader.py` — TCGA tile loader and data augmentations
- `configs/{smoke,leader}.yaml` — training recipes (e.g., hyperparameters)

### Helper files
- `AGENTS.md` — guidelines for AI assistants and human contributors: design philosophy (minimal/hackable, nanochat-flavored), coding rules, experiment discipline, and cluster/storage conventions. Note some language is specific to the MedARC cluster.
- `benchmarking/` — checked-in split metadata plus benchmark philosophy, dataset notes, source links, timing tables, and test-split policy.
- `prepare.py` — data prep: verify or download HF tile mirror + probe datasets + any pretrained weights.
- `probe.py` — downstream probes (KNN, few-shot, linear, segmentation, slide AUROC, survival, robustness).
- `submit/train_1gpu.sbatch` — SLURM launcher for single-GPU training.
- `download_TCGA.sh` — manual utility, run by hand if you want the full 12K TCGA open-access SVS slide set (~13 TB) for forking the tile-extraction recipe. Not invoked by `prepare.py` and not needed for any standard training workflow.
- `LOG.md` — running notes on what has been tried, including negative results.
- `pyproject.toml` + `uv.lock` — Python dependency spec consumed by `uv sync`.

## Data

`prepare.py` prepares the necessary data for pretraining and downstream probing. Flag `download=True` to fetch/prepare the configured datasets into the folders specified by the YAML; flag `download=False` to verify that all required paths are already populated.

Edit `data.dataset_dir` and every `probe.dataset_roots.*` in your config (`configs/leader.yaml` and `configs/smoke.yaml` if you also smoke-test) to your own correct paths.

```bash
# Pull the 4M-tile parquet dataset from the medarc/nanopath HF mirror,
# fetch/prepare probe datasets, and fetch pretrained models.
python prepare.py configs/leader.yaml download=True

# Verify-only: confirms the parquet shards, every probe dataset listed in
# the config, and any necessary pretrained weights all exist on disk where
# train.py expects them.
python prepare.py configs/leader.yaml download=False
```

**What `download=True` does**
1. **TCGA tiles**: `huggingface_hub.snapshot_download` (filtered to `shard-*.parquet`) pulls the 200 parquet shards (~120 GB total, `{path: string, jpeg: binary}` rows with 64-row row groups) from [`medarc/nanopath`](https://huggingface.co/datasets/medarc/nanopath) into `data.dataset_dir`.
2. **Probe datasets**: for each empty configured root, fetches/unpacks and, where needed, pre-extracts the probe data. BRACS, BreaKHis, PCam, PanNuke, UCLA Lung, HER2, CRC survival, PathoROB, and MoNuSAC come from their official public sources. MHIST, CoNSeP, and SurGen use the [`medarc/nanopath`](https://huggingface.co/datasets/medarc/nanopath) probe mirror for portable noninteractive setup; before fetching MHIST or CoNSeP, `prepare.py` prints that users must satisfy the official upstream form/access terms first. Slide-level probes cache uncapped 20x/512 tissue grids (`tiles/`, `tiles.parquet`, `surgen-*.parquet`, or `patches.parquet`) so `probe.py` never opens raw WSIs.
3. **DINOv2 backbone weights**: `torch.hub.load_state_dict_from_url` fetches the Meta checkpoint for `model.type` from `dl.fbaipublicfiles.com` into `~/.cache/torch/hub/checkpoints/`.

**Prerequisites**
- ~120 GB free wherever `data.dataset_dir` lives for the parquet shards (cluster default: `/data/nanopath_parquet`).
- Probe data disk varies by suite: the checked-in cluster paths are shared; off-cluster, expect large one-time downloads and preprocessing for PanNuke, UCLA Lung, HER2, CRC survival, and MoNuSAC. SurGen's official CZI regeneration path is multi-hour, so normal setup pulls our pre-extracted ~102 GB HF parquet cache instead. Reruns skip already-populated roots.
- ~330 MB free under `~/.cache/torch/hub/checkpoints/` for DINOv2 weights.
- `wget` on PATH for the BRACS FTP mirror. Python-side WSI/probe dependencies are installed by `uv sync`.

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

To publish a new variant of the dataset, push the resulting shards to a fresh HF dataset repo and update `HF_REPO_ID` in `prepare.py`.

## Running

Smoke (single GPU, short training plus the full probe suite, validates the train+probe path; H100 probe wall is ~31-50 minutes for the remeasured baselines):

```bash
sbatch submit/train_1gpu.sbatch configs/smoke.yaml
# or directly on a GPU machine: `python train.py configs/smoke.yaml`
```

Untouched DINOv2-S baseline (no training, full probe only):

```bash
python baselines/dinov2_small_baseline.py configs/leader.yaml output_dir=/data/$USER/nanopath/baselines/dinov2-small
```

Untouched OpenMidnight baseline (no training, full probe only):

```bash
python baselines/openmidnight_baseline.py configs/leader.yaml output_dir=/data/$USER/nanopath/baselines/openmidnight
```

Untouched H-optimus-0 baseline (no training, full probe only):

```bash
python baselines/hoptimus0_baseline.py configs/leader.yaml output_dir=/data/$USER/nanopath/baselines/hoptimus0
```

Maintained Nanopath recipe (full train+probe)

```bash
sbatch submit/train_1gpu.sbatch configs/leader.yaml
# or directly on a GPU machine: `python train.py configs/leader.yaml`
```

`configs/leader.yaml` is sized for an 80 GB H100 at `train.batch_size: 128`. On smaller cards you can set `train.activation_checkpointing: true` if you OOM. Smoke fits comfortably on any 24 GB+ GPU.

## Outputs

- run outputs: `project.output_dir` (default is `/data/$USER/nanopath/leader/...`). Final probe results log to `metrics.jsonl`.
- wandb: `/data/$USER/nanopath/wandb`.
- parquet tile shards: `data.dataset_dir` (defaults to `/data/nanopath_parquet`).
- probe datasets: `probe.dataset_roots` (defaults to shared `/block/...` and `/data/...` paths on the MedARC cluster).
- DINOv2 backbone weights: `~/.cache/torch/hub/checkpoints/` for the selected `model.type`.
- SLURM logs: `slurm/<jobid>.{out,err}` in the repo.
- checkpoints: rolling `latest.pt` written every `train.save_every` steps under `project.output_dir`, plus one final save at end of run. `save_every: null` (smoke) disables both; probes always get their own short-lived checkpoint regardless.

## Experiment log

See [LOG.md](LOG.md) for running notes on what has been tried in nanopath. Negative results included! Such logs help contributors avoid retrying dead ends.

## Acknowledgements

Inspired by [nanochat](https://github.com/karpathy/nanochat). The DINOv2 backbone weights are [Meta checkpoints](https://github.com/facebookresearch/dinov2) loaded by state-dict into our own clean ViT implementation. Tile-classification and segmentation probes follow the [Thunder benchmark](https://mics-lab.github.io/thunder/); slide classification and survival probes follow [PathoBench](https://huggingface.co/datasets/MahmoodLab/Patho-Bench) task metadata.
