# nanopath

![nanopath logo](imgs/nanopath_logo.png)

`nanopath` is a super lean experimental harness for training tile-level computational pathology foundation models, inspired by [nanochat](https://github.com/karpathy/nanochat). It trains in less than 45 minutes on a single GPU and covers the full pretraining pipeline using public TCGA dataset (12k WSIs) and built-in downstream probes spanning classification, segmentation, slide-level mutation/progression, survival, and robustness. The goal is to easily explore and iterate on research directions to see what works best on small-scale, then we scale up the best performing training recipes with larger compute.

This repository is intentionally made to be compatible with [autoresearch](https://github.com/karpathy/autoresearch)-style pursuits, and we even have a live autoresearch-style plot in [Leaderboard](#leaderboard). Nanopath models must finish training on a single H100 in <45 minutes, and are evaluated in the same job in <15 minutes across a downstream benchmark suite.

**Want to get involved? Join us in the [MedARC Discord](https://discord.gg/tVR4TWnRM9) (find us in #path-fm)!**

## Quickstart

Install [uv](https://docs.astral.sh/uv/) first if you don't have it, then:

```bash
git clone https://github.com/MedARC-AI/nanopath.git && cd nanopath
uv sync && source .venv/bin/activate
wandb login

# download pretraining & probe datasets & DINOv2 pretrained ckpt
python prepare.py configs/smoke.yaml download=True

# smoke test: short training plus the fixed full probe suite
sbatch submit/train_1gpu.sbatch configs/smoke.yaml
# or directly on a GPU machine: python train.py configs/smoke.yaml

# train and evaluate the current leader nanopath recipe
sbatch submit/train_1gpu.sbatch configs/leader.yaml
# or directly on a GPU machine: python train.py configs/leader.yaml

# publish the completed smoke run to the live labless plot
./labless/submit_to_labless.py output_dir=$RUN_DIR contributor=@yourgithub notes="what changed"
```

`pyproject.toml` pins `torch` / `torchvision` against the CUDA 12.9 wheel index. If your GPU/driver needs a different CUDA build, edit the `torch` and `torchvision` lines in `pyproject.toml` before `uv sync`.

A successful model training prints periodic train lines, appends metrics to `metrics.jsonl`, and writes the final comparison artifact to `summary.json`. `configs/smoke.yaml` is simply meant to pretrain briefly and run the fixed downstream probe suite to ensure everything works.

## Leaderboard

[![Nanopath progress plot](https://api.labless.dev/api/nano-projects/nanopath/plot.svg)](https://labless.dev/nano-projects/nanopath)

Score is final `mean_probe_score` across our 11-dataset benchmarking suite, assessing tile-level classification (linear probing, knn, few-shot), segmentation, slide-level classification (progression, mutation, survival), and robustness. These benchmarks are derived from [THUNDER](https://mics-lab.github.io/thunder/) and [PathoBench](https://github.com/mahmoodlab/patho-bench), with modifications to make them finish in under 15 minutes on an H100 gpu. We operate only on the train/validation splits for these datasets, entirely holding out the test splits defined in THUNDER/PathoBench, so these benchmark suites remain valid for `nanopath` models without overfitting. See [benchmarking/README.md](benchmarking/README.md) for more information.

### Nanopath models

| # | Description | final score | linear | knn | 16-shot | segmentation | progression | mutation | survival | robustness | Contributors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | DINOv2-S/14-reg leader recipe trained on TCGA to 1e18 FLOPs (`v18_s4242`) | 0.5563 | 0.7656 | 0.7046 | 0.6667 | 0.3000 | 0.6644 | 0.5843 | 0.5070 | 0.6142 | @PaulScotti |

### Baselines

| # | Name | Description | final score | linear | knn | 16-shot | segmentation | progression | mutation | survival | robustness |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | GenBio-PathFM | GenBio-PathFM ViT-G/16 baseline | **0.6268** | 0.8076 | 0.7626 | 0.6970 | 0.3301 | 0.7680 | 0.6375 | 0.5349 | 0.9412 |
| 2 | H-optimus-0 | H-optimus-0 ViT-G/14-reg baseline | 0.6190 | 0.7995 | 0.7676 | 0.6931 | 0.3261 | 0.7004 | 0.6584 | 0.5661 | 0.8926 |
| 3 | DINOv2-giant | Untouched Meta `dinov2_vitg14_reg` | 0.5627 | 0.7689 | 0.7208 | 0.5834 | 0.2845 | 0.6000 | 0.6174 | 0.5562 | 0.7985 |
| 4 | OpenMidnight | OpenMidnight ViT-G/14-reg baseline | 0.5494 | 0.7926 | 0.7135 | 0.4335 | 0.3064 | 0.6993 | 0.6091 | 0.4861 | 0.7438 |
| 5 | DINOv2-small | Untouched Meta `dinov2_vits14_reg` | 0.5304 | 0.6968 | 0.6249 | 0.5834 | 0.2675 | 0.5827 | 0.6225 | 0.5321 | 0.7543 |
| 6 | DINOv2-small random | `dinov2_vits14_reg` with random weights | 0.4268 | 0.5237 | 0.5066 | 0.4139 | 0.2682 | 0.6922 | 0.5648 | 0.5176 | 0.1905 |

Baseline rows are frozen reference checkpoints evaluated with the same probe suite. They help calibrate the plot, but pathology-specific baselines are not valid initialization points for nanopath leaderboard submissions.

### How to submit to the leaderboard

`configs/leader.yaml` is the currently winning `nanopath` training recipe. Submit any completed or failed run to labless:

```bash
RUN_DIR=/data/$USER/nanopath/leader/my-run
./labless/submit_to_labless.py output_dir=$RUN_DIR contributor=@yourgithub wandb_url=https://wandb.ai/... notes="what changed and why"
```

To top the leaderboard you must outperform this recipe on `mean_probe_score` by at least 0.01. Submit the run to labless; that public submission is the leaderboard claim, with git state, changed files, notes, metrics, hardware, and optional W&B attached. [@PaulScotti](https://github.com/PaulScotti) will inspect promising submissions, rerun the candidate on his 1 80GB H100 with a different rng seed, and update the README, `configs/leader.yaml`, and labless leader state if it still improves by at least 0.01. **You don't need an H100 or a PR to submit**—train on whatever hardware you have access to, and labless handles the public record and maintainer validation.

Code-cleanup PRs are still welcome when they simplify the codebase without changing benchmark peformance on the leader recipe. Leaderboard claims should go through labless instead of a pull request.

### What you must NOT change for a leaderboard submission

Anything not explicitly fixed below (e.g., model architecture, training objective, optimizer, lr scheduler, data augmentations, masking, dataset curation) is fair game for modification.

**Training ends at 1e18 total FLOPs OR after 45 min. elapsed on 1xH100**

Every leaderboard run is verified on the organizer's compute (1 80GB H100 gpu), bounded by two possible caps:

- **≤45 min. training on a single 80 GB H100 before probe benchmarking**. `submit/train_1gpu.sbatch` runs with `--signal=USR1@900`, so SLURM sends `SIGUSR1` 15 minutes before the `--time` wall; `train.py`'s SIGUSR1 handler catches it as a clean stop signal, cuts training, and uses the remaining window for the final checkpoint save + downstream probe suite.
- **`train.max_train_flops` ≤ 1e18 training FLOPs**, measured directly via `torch.utils.flop_counter.FlopCounterMode` on the first step (forward + backward + optimizer.step) and reused thereafter since per-step shapes are fixed. This counts everything that touches the GPU during a step (student backbone, EMA teacher forward, projection heads, masking, etc.).

The above limits force submissions to be **simultaneously compute efficient and systems efficient**.

**TCGA as the only pretraining data**
- TCGA (12K WSIs) is the only dataset allowed for pretraining, but you are free to revise how we select the tiles used for training.
- The probe datasets cannot be used for pretraining, neither directly (training data) nor indirectly (distillation target, contrastive negatives, label-smoothing prior, etc.).

**Probe evaluation must be untouched**
- All of `probe.py` and `benchmarking/`
- All probe config variables in `configs/leader.yaml`.

**Initializing model from a pretrained ckpt is OK only if not pathology-specific**
You can initialize the model using DINOv2 checkpoint (trained on natural images) but you can't initialize from, say, H-optimus-0 or OpenMidnight checkpoints. We want to train our own pathology foundation model, not offload most of the task to someone else's pathology-specific model.

### Labless for live tracking

Submit any informative completed runs to the live tracker:

```bash
RUN_DIR=/data/$USER/nanopath/leader/my-run
./labless/submit_to_labless.py output_dir=$RUN_DIR contributor=@yourgithub notes="what changed and why"
```

The script reads `summary.json` and `metrics.jsonl`, writes `labless_submission.json` into the run directory, and posts to `api.labless.dev`. The labless website, run log, and plot update automatically; new completed runs stay `pending` until maintainer validation. See [labless/README.md](labless/README.md) for details.

## Repository layout

### Primary files meant to be hacked
- `train.py` — main pretraining loop
- `model.py` — model architecture and training objectives
- `dataloader.py` — TCGA tile loader and data augmentations
- `configs/{smoke,leader}.yaml` — training recipes (e.g., hyperparameters)

### Helper files
- `AGENTS.md` — guidelines for design philosophy, coding rules, experiment discipline, cluster conventions, etc. Note this is Paul's personal `AGENTS.md` file and has instructions specific to our MedARC cluster—you should modify this file to suit your own setup!
- `benchmarking/` — supports probing/downstream evaluation.
- `prepare.py` — data prep: verify or download pretraining data + probe datasets + any pretrained weights.
- `probe.py` — downstream probes (KNN, few-shot, linear, segmentation, slide AUROC, survival, robustness).
- `submit/train_1gpu.sbatch` — SLURM launcher for single-GPU training.
- `labless/submit_to_labless.py` + `labless/labless.yaml` — package a run and post it to the live labless tracker.
- `download_TCGA.sh` — manual utility, run by hand if you want the full 12K TCGA open-access SVS slide set (~13 TB) for forking the tile-extraction recipe. Not invoked by `prepare.py` and not needed for any standard training workflow.
- `pyproject.toml` + `uv.lock` — Python dependencies used by `uv sync`.

## Data

`prepare.py` prepares the necessary data for pretraining and downstream probing. Flag `download=True` to fetch/prepare the configured datasets into the folders specified by the YAML; flag `download=False` to verify that all required paths are already populated.

On the MedARC cluster, the checked-in data paths are the intended defaults. On another VM, copy the config you plan to run and edit `data.dataset_dir` plus every `probe.dataset_roots.*` path to writable local storage before `prepare.py download=True`.

**What `download=True` does**
1. **TCGA tiles**: `huggingface_hub.snapshot_download` (filtered to `shard-*.parquet`) pulls the 200 parquet shards (~120 GB total, `{path: string, jpeg: binary}` rows with 64-row row groups) from [`medarc/nanopath`](https://huggingface.co/datasets/medarc/nanopath) into `data.dataset_dir`.
2. **Probe datasets**: for each empty configured root, fetches/unpacks and, where needed, pre-extracts the probe data. BRACS, BreaKHis, PCam, PanNuke, UCLA Lung, PathoROB, and MoNuSAC come from their official public sources. MHIST, CoNSeP, SurGen, and BoehmK survival use the [`medarc/nanopath`](https://huggingface.co/datasets/medarc/nanopath) probe mirror for portable noninteractive setup; before fetching MHIST, CoNSeP, or BoehmK survival, `prepare.py` prints that users must satisfy the official upstream form/access terms first. Slide-level probes cache 20x/512 tissue grids (`tiles.parquet`, `surgen-*.parquet`, or `patches.parquet`) so `probe.py` never opens raw WSIs; SurGen and BoehmK survival prepare the full grid but stream deterministic raster-spaced sub-bags for runtime.
3. **DINOv2 backbone weights**: `torch.hub.load_state_dict_from_url` fetches the Meta checkpoint for `model.type` from `dl.fbaipublicfiles.com` into `~/.cache/torch/hub/checkpoints/`.

**Prerequisites**
- ~120 GB free wherever `data.dataset_dir` lives for the parquet shards (cluster default: `/data/nanopath_parquet`).
- Probe data disk varies by suite. Expect that it might take a few hours for one-time downloading and preprocessing of all probe datasets.

### Regenerating the tile dataset from raw SVS

`prepare.py` itself never touches raw SVS files—it always pulls the ready-made parquet shards from HF. If you want, however, you can download the full ~13 TB original SVS files from TCGA and pre-extract different tiles to pretrain on. Two-step workflow (decode SVS → JPEG dir + manifest, then pack into parquet shards):

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

To publish a new variant of the dataset, you can push the resulting shards to a fresh HF dataset repo and update `HF_REPO_ID` in `prepare.py`.

## Running

Smoke (short training + full probe, ~20 minutes):

```bash
sbatch submit/train_1gpu.sbatch configs/smoke.yaml
# or directly on a GPU machine: `python train.py configs/smoke.yaml`
```

Full leading `nanopath` recipe (~1 hour):

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
- labless submission payload: `project.output_dir/labless_submission.json`.
- checkpoints: rolling `latest.pt` written every `train.save_every` steps under `project.output_dir`, plus one final save at end of run. `save_every: null` (smoke) disables both; probes always get their own short-lived checkpoint regardless.

## Experiment log

See the live [labless nanopath log](https://labless.dev/nano-projects/nanopath) for submitted runs, including negative results. Labless is the source of truth for experiment history so the record updates immediately when contributors submit runs.

## Acknowledgements

Inspired by [nanochat](https://github.com/karpathy/nanochat). The DINOv2 backbone weights are [Meta checkpoints](https://github.com/facebookresearch/dinov2) loaded by state-dict into our own clean ViT implementation. Tile-classification and segmentation probes follow the [THUNDER benchmark](https://mics-lab.github.io/thunder/); slide-level probes follow [PathoBench](https://huggingface.co/datasets/MahmoodLab/Patho-Bench).
