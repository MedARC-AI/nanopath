# nanopath

![nanopath logo](imgs/nanopath_logo.png)

> **Sept 2026 update**: Our ~20 min. fast evaluation suite now adds more downstream datasets, removes explicitly TCGA-derived tasks, reworks the lightweight probing code, and reweights final score calculation. "nanopath-evals" is now a much more faithful proxy for performance on official benchmarks. However, this required resetting the Labless live leaderboard plot (original is still preserved but should not be hillclimbed on).

`nanopath` is a super lean experimental harness for training tile-level computational pathology foundation models, inspired by [nanochat](https://github.com/karpathy/nanochat). In ~1 hour it trains on 1 million pathology tiles on a single GPU and evaluates a broad suite of downstream probes spanning tile classification, segmentation, slide-level mutation/progression/survival, and robustness. The goal is to easily explore and iterate on research directions to see what works best on small-scale, then scale up the best performing training recipes with more data and larger compute.

This repository is intentionally made to be compatible with [autoresearch](https://github.com/karpathy/autoresearch)-style pursuits, and we even have a live autoresearch-style plot in [Leaderboard](#leaderboard). Nanopath models train until the next full batch would exceed the 1,000,000 tile-presentation cap or until the run reaches the 1e18-FLOP cap.

**Want to get involved? Join us in the [MedARC Discord](https://discord.gg/tVR4TWnRM9) (find us in #path-fm)!**

## Quickstart

Install [uv](https://docs.astral.sh/uv/) first if you don't have it, then:

```bash
git clone https://github.com/MedARC-AI/nanopath.git && cd nanopath
uv sync && source .venv/bin/activate
wandb login  # or: export WANDB_MODE=offline before launching noninteractive SLURM jobs

# download pretraining & probe datasets & DINOv2 pretrained ckpt
python prepare.py download=True

# smoke test: very short training, then probe evals to ensure no errors
./submit/train_1gpu.sbatch configs/smoke.yaml
# or directly on a GPU machine: python train.py configs/smoke.yaml

# train and evaluate the current nanopath recipe
# auto-submits to Labless if config passes submission requirements and you provide run name/notes & GitHub login
RUN_DIR=$PWD/data/main/my-run
./submit/train_1gpu.sbatch configs/main.yaml output_dir=$RUN_DIR
# or directly on a GPU machine: python train.py configs/main.yaml output_dir=$RUN_DIR
```

`pyproject.toml` pins `torch` / `torchvision` against the CUDA 12.9 wheel index. If your GPU/driver needs a different CUDA build, edit the `torch` and `torchvision` lines in `pyproject.toml` before `uv sync`.

A successful model training prints periodic train lines, appends metrics to `metrics.jsonl`, and writes the final comparison artifact to `summary.json`. `configs/smoke.yaml` is simply meant to pretrain briefly and then run the fixed downstream probe suite to ensure everything works without errors.

W&B can run online or offline, but set that up before submitting a noninteractive job: either run `wandb login` once, or export `WANDB_MODE=offline`.

## Leaderboard

<a href="https://labless.dev/nano-projects/nanopath-v2">
  <img src="https://api.labless.dev/api/nano-projects/nanopath-v2/plot.svg" alt="nanopath progress plot" width="1290">
</a>

`final_score` weights classification, segmentation, progression, mutation, survival, and CRoMa robustness at 30%, 15%, 17.5%, 15%, 7.5%, and 15%. These columns summarize a 20-dataset suite derived from [THUNDER](https://mics-lab.github.io/thunder/), [PathoBench](https://github.com/mahmoodlab/patho-bench), [LEOPARD](https://leopard.grand-challenge.org/), [PathoROB](https://arxiv.org/abs/2507.17845), and [CRoMa](https://github.com/clemsgrs/croma), with modifications to keep single-GPU evaluation lightweight and use train/validation-only data. See [benchmarking/README.md](benchmarking/README.md) for the full protocol and provenance caveats.

Nanopath models should be submitted to [Labless](https://labless.dev). `main` corresponds to this repository's main recipe. For promotion, [@PaulScotti](https://github.com/PaulScotti) retrains an unvalidated candidate three times with different randomly selected RNG seeds; its median run becomes the validated `leader` if it beats the incumbent by at least 0.004. `robust-norm-s9876` is the approved exception.

![nanopath final score compared with held-out official evaluations](imgs/threepanel-v2.png)

![nanopath classification and segmentation scores compared with THUNDER](imgs/twopanel-v2.png)

As you can see in the above correlation plots, our fast ~20 minute evaluation suite "nanopath-evals" used to calculate `final_score` strongly correlates to the much slower, official pathology foundation model benchmarks of THUNDER, HEST, and PathoBench. This suggests nanopath hillclimbing should reflect useful, generalizable improvements.

### nanopath models

The `main` branch of `nanopath` codebase reflects the `lr-and-curation` nanopath recipe. Clicking the model name in table below will take you to a GitHub branch for the code used to train its Labless validated run.

| # | Description | final score | classification | segmentation | progression | mutation | survival | robustness | Contributors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | [robust-norm](https://github.com/MedARC-AI/nanopath/tree/robust-norm-v2) | **0.6476** | 0.7507 | 0.6024 | 0.6136 | 0.5885 | 0.6010 | 0.6088 | @anishdulal |
| 2 | jepa-fino | 0.6465 | 0.7384 | 0.6016 | 0.5903 | 0.6190 | 0.6210 | 0.6132 | @ml-and-ml |
| 3 | I-JEPA contig patch | 0.6409 | 0.7219 | 0.5993 | 0.5931 | 0.6148 | 0.6172 | 0.6143 | @NimaAsh |
| 4 | block-strided-cls | 0.6388 | 0.7477 | 0.6039 | 0.5390 | 0.6066 | 0.6335 | 0.6069 | @RyanKim17920 |
| 5 | [lr-and-curation](https://github.com/MedARC-AI/nanopath) | 0.6332 | 0.7048 | 0.5940 | 0.5948 | 0.6025 | 0.6199 | 0.6114 | @nevasini1 |

### Baselines

| # | Name | Description | final score | classification | segmentation | progression | mutation | survival | robustness |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | GenBio-PathFM | GenBio-PathFM ViT-G/16 | **0.6999** | 0.8213 | 0.6100 | 0.7046 | 0.6232 | 0.6312 | 0.6525 |
| 2 | UNI-2-h | MahmoodLab UNI-2-h ViT-H/14 | 0.6944 | 0.8161 | 0.6323 | 0.7156 | 0.6408 | 0.6105 | 0.5837 |
| 3 | Midnight-12K | Kaiko Midnight-12K ViT-G/14 | 0.6937 | 0.7761 | 0.6326 | 0.7080 | 0.6170 | 0.6100 | 0.6918 |
| 4 | H-optimus-0 | H-optimus-0 ViT-G/14-reg | 0.6833 | 0.8082 | 0.5916 | 0.6961 | 0.6485 | 0.6059 | 0.5838 |
| 5 | H0-mini | Bioptimus H0-mini ViT-B/14-reg | 0.6809 | 0.7956 | 0.6371 | 0.6586 | 0.6026 | 0.5945 | 0.6429 |
| 6 | Virchow | Paige/Microsoft Virchow ViT-H/14 | 0.6775 | 0.7728 | 0.6326 | 0.6541 | 0.6291 | 0.6173 | 0.6379 |
| 7 | GigaPath | Prov-GigaPath tile encoder ViT-G/16 | 0.6685 | 0.7948 | 0.6197 | 0.6813 | 0.6130 | 0.5802 | 0.5492 |
| 8 | GigaPath-Flash | Prov-GigaPath-Flash tile encoder ViT-S/16 | 0.6500 | 0.7742 | 0.5569 | 0.6620 | 0.5796 | 0.6122 | 0.5703 |
| 9 | Kaiko-S/16 | Kaiko pathology ViT-S/16 | 0.6409 | 0.7737 | 0.6060 | 0.5963 | 0.5539 | 0.5907 | 0.5741 |
| 10 | OpenMidnight | OpenMidnight ViT-G/14-reg | 0.6382 | 0.6640 | 0.6306 | 0.6748 | 0.5829 | 0.6058 | 0.6229 |
| 11 | DINOv2-G/14 | Meta DINOv2-G/14-reg | 0.6149 | 0.6804 | 0.5753 | 0.5928 | 0.6038 | 0.6288 | 0.5534 |
| 12 | DINOv2-L/14 | Meta DINOv2-L/14-reg | 0.6124 | 0.6632 | 0.5667 | 0.6298 | 0.6005 | 0.6009 | 0.5540 |
| 13 | DINOv2-B/14 | Meta DINOv2-B/14-reg | 0.5988 | 0.6500 | 0.5691 | 0.5753 | 0.6062 | 0.6015 | 0.5449 |
| 14 | DINOv2-S/14 | Meta DINOv2-S/14-reg | 0.5945 | 0.6480 | 0.5665 | 0.5364 | 0.6202 | 0.6220 | 0.5437 |

Baseline rows are frozen reference checkpoints evaluated with the same probe suite. They help calibrate the plot, but pathology-specific baselines are not valid initialization points for nanopath leaderboard submissions. The reference scripts live in `baselines/`.

### How to submit to the leaderboard

Labless is our public run ledger and live plot for `nanopath`. You do not need a Labless password or a pull request to make a leaderboard claim; the submitter connects your submission to your GitHub identity through GitHub's device sign-in. We encourage you to submit *all* completed full runs, including null results and incremental tweaks; a dense public ledger lets you (and AI agents, see our [public ledger API](https://labless.dev/docs/ledger-api)) mine through everyones runs to uncover new insights.

See [labless/README.md](labless/README.md) for Labless submission details and public API usage.

`configs/main.yaml` is the current nanopath training recipe. A normal SLURM submission is:

```bash
RUN_DIR=$PWD/data/main/my-run
./submit/train_1gpu.sbatch configs/main.yaml output_dir=$RUN_DIR
```

The pipeline is:

1. Run `./submit/train_1gpu.sbatch ...` or `python train.py ...` to start your training run. For full runs, the launcher asks for a short `run_name`, an optional experiment note naming the unique change and why, and GitHub device sign-in before scheduling the GPU job. Leaving the run name blank or failing to sign in will lead to skipping labless submission.
2. Let `train.py` finish the final probe. The run directory will contain `summary.json`, `metrics.jsonl`, and the source snapshot written at launch under `labless_source/`. The submitter writes `labless_submission.json`, checks the run caps and locked benchmark surface, posts to `api.labless.dev`, and shows the run as `unvalidated` until maintainer validation.

Manual submission is still available for direct `python train.py` runs or copied output directories:

```bash
./labless/submit_to_labless.py output_dir=$RUN_DIR run_name=kde-crops notes="vs main: larger local crops to retain tissue context"
```

Public full-run submissions must satisfy:

- `summary.max_train_samples == 1000000`
- `summary.tile_presentations <= 1000000`
- `summary.max_train_flops == 1e18`
- `final_score` is present
- no saved-source changes to `probe.py` or anything under `benchmarking/`
- no locked probe config changes except local `probe.dataset_roots`

The `run_name` is the short label shown next to your dot on the Labless plot; keep it under 20 characters and make it describe what changed. Short smoke-sized runs, failed runs, and runs missing the saved source snapshot stay local. Each verified GitHub login can submit at most 100 runs per 24 hours.

Public submissions have no wall-clock limit. Each maintainer reproduction must train within 2 hours. **You don't need an H100 or a PR to submit**; Labless handles the public record and maintainer validation.

Code-cleanup PRs are still welcome when they simplify the codebase without changing benchmark performance on the main recipe. Leaderboard claims should go through Labless instead of a pull request.

### What you must NOT change for a leaderboard submission

Anything not explicitly fixed below (e.g., model architecture, training objective, optimizer, lr scheduler, data augmentations, masking, dataset curation) is fair game for modification.

**Training ends at 1,000,000 tile-presentation samples OR 1e18 total FLOPs**

Every leaderboard run is bounded by two possible caps:

- **`train.max_train_samples` ≤ 1,000,000 tile presentations**. A training sample is one source TCGA tile emitted as one dataloader item; if the same underlying tile is seen again later, that is another tile presentation. Teacher/student views, global/local crops, masks, or other augmentations derived from that tile do not multiply the sample count, though their compute still counts toward FLOPs. `train.py` never starts a batch that would push `summary.tile_presentations` over the cap.
- **`train.max_train_flops` ≤ 1e18 training FLOPs**, measured directly via `torch.utils.flop_counter.FlopCounterMode` on the first step (forward + backward + optimizer.step) and reused thereafter since per-step shapes are fixed. This counts everything that touches the GPU during a step (student backbone, EMA teacher forward, projection heads, masking, etc.).

LR decay, weight decay, teacher-temperature, freeze, and KDE schedules are keyed to `train_flops / train.max_train_flops`; LR warmup is keyed to tile presentations so it finishes early in the 1,000,000-tile sample-capped run. With the current small model and augmentations, `configs/main.yaml` normally reaches the sample cap at about 19% of the 1e18-FLOP budget, so the FLOP-keyed schedules intentionally stop early unless you change the caps or schedule fractions.

Wall time is logged for diagnostics and standardized reruns, but it is not a public-submission eligibility cap. Maintainer validation is separate: the submitted recipe must complete training on the maintainer's single 80 GB H100 within 2 hours.
Intensive preprocessing before model training starts, such as tile extraction, data curation, metadata joins, indexing, or embedding generation, is allowed and is not counted as training time.

**TCGA as the only tile source**
- Every image tile used for training must be produced exclusively from the 12K TCGA WSIs. You can change tile extraction, filtering, sampling, curation, and preprocessing before the capped model-training run begins.
- Public non-tile information is fair game: metadata, clinical/genomic labels, text, ontologies, annotations, or other non-image-tile signals from any public source may be used however you want.

**Probe evaluation must be untouched**
- All of `probe.py` and `benchmarking/` (note this means you *can* modify model.py however you wish!)
- All probe config variables in `configs/main.yaml`.

**Pretraining must not use pathology-specific pretrained models**
Non-pathology pretrained models such as DINOv2 may be used for initialization, teachers, data curation, or preprocessing. Pathology-trained checkpoints such as H-optimus-0 or OpenMidnight may not initialize weights or guide training, but they may be used before and separately from training for TCGA-tile curation or preprocessing.

### Labless for live tracking

Full training runs auto-submit to the labless live tracker if certain criteria are met (see [How to submit to the leaderboard](#how-to-submit-to-the-leaderboard)).

The script reads `summary.json` and `metrics.jsonl`, reviews `output_dir/labless_source` rather than your current working tree, and posts the local payload in `labless_submission.json` after GitHub device sign-in succeeds. W&B can be online or offline; online runs add a public W&B link, while source review always comes from the local snapshot. `AGENTS.md` and `CLAUDE.md` are excluded from Labless source packaging. The labless website, run log, and plot update automatically.

## Repository layout

### Primary files meant to be hacked
- `train.py` — main pretraining loop
- `model.py` — model architecture and training objectives
- `dataloader.py` — TCGA tile loader and data augmentations
- `configs/{smoke,main}.yaml` — training recipes (e.g., hyperparameters)

### Helper files
- `AGENTS.md` — guidelines for design philosophy, coding rules, experiment discipline, cluster conventions, etc. Note this is Paul's personal `AGENTS.md` file and has instructions specific to our MedARC cluster—you should modify this file to suit your own setup!
- `benchmarking/` — dataset/protocol documentation.
- `prepare.py` — data prep: verify or download pretraining data + probe datasets + any pretrained weights.
- `probe.py` — downstream probes (KNN, few-shot, linear, segmentation, slide AUROC, survival, robustness).
- `submit/train_1gpu.sbatch` — SLURM launcher for single-GPU training.
- `labless/submit_to_labless.py` — package a run and post it to the live labless tracker.
- `download_TCGA.sh` — manual utility, run by hand if you want the full 12K TCGA open-access SVS slide set (~13 TB) for forking the tile-extraction recipe. Not invoked by `prepare.py` and not needed for any standard training workflow.
- `pyproject.toml` + `uv.lock` — Python dependencies used by `uv sync`.

## Data

`prepare.py` prepares the necessary data for pretraining and downstream probing. By default it reads `configs/main.yaml`; pass a YAML path before the flag to prepare a different config, e.g. `python prepare.py configs/smoke.yaml download=True`. Flag `download=True` to fetch/prepare the configured datasets into the folders specified by the YAML; flag `download=False` to verify that all required paths are already populated.

On the MedARC cluster, the checked-in `/data` paths are the intended shared defaults and existing populated roots are reused. On a machine without writable `/data` or `/block` mounts, `download=True` rewrites the checked-in main and smoke configs to ignored repo-local `data/` roots before downloading.

**What `download=True` does**
1. **TCGA tiles**: `huggingface_hub.snapshot_download` (filtered to `shard-*.parquet`) pulls the 200 parquet shards (~120 GB total, `{path: string, jpeg: binary}` rows with 64-row row groups) from [`medarc/nanopath`](https://huggingface.co/datasets/medarc/nanopath) into `data.dataset_dir`.
2. **Probe datasets**: downloads the exact evaluation snapshot from [`medarc/nanopath-evals`](https://huggingface.co/datasets/medarc/nanopath-evals) into each missing configured root, then verifies every required record.
3. **DINOv2 backbone weights**: `torch.hub.load_state_dict_from_url` fetches the Meta checkpoint for `model.type` from `dl.fbaipublicfiles.com` into `~/.cache/torch/hub/checkpoints/`.

**Prerequisites**
- About 355 GB free for a fresh complete setup: ~120 GB of pretraining shards, ~215 GB of extracted probe data, and temporary room while the largest image archive is extracted. Existing populated roots reduce the download and space requirement.
- Acceptance of each upstream benchmark dataset's original research-use terms. The MedARC mirror preserves the data needed by the protocol but does not relicense its components.

Our evaluation suite only downloads a small subset of non-test data derived from [THUNDER](https://mics-lab.github.io/thunder/), [PathoBench](https://github.com/mahmoodlab/patho-bench), [LEOPARD](https://leopard.grand-challenge.org/), and [PathoROB](https://arxiv.org/abs/2507.17845). It contains no official THUNDER, HEST, or CPTAC classification test records; HEST is absent entirely, CPTAC appears only in the existing CPTAC-PDA survival development probe, PanNuke Fold3 is absent, and the unused TCGA center is removed from downloadable Tolkach ESCA. See [benchmarking/README.md](benchmarking/README.md) for the precise split contract.

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
jpeg_dir = Path('/data/$USER/nanopath/nanopath_jpegs_tmp')
prepare_tiles(Path('/data/TCGA/sample_dataset_30.txt'), jpeg_dir, split_seed=42)
pack_from_jpeg_dir(jpeg_dir, jpeg_dir / 'manifest.txt', Path('/data/$USER/nanopath/nanopath_parquet'))
"
```

Point `data.dataset_dir` at the packed parquet directory before training. To publish a new variant of the training dataset, push the resulting shards to a fresh HF dataset repo and update `HF_TRAIN_REPO_ID` in `prepare.py`.

## Running

Smoke (short training + full probe):

```bash
./submit/train_1gpu.sbatch configs/smoke.yaml
# or directly on a GPU machine: `python train.py configs/smoke.yaml`
```

Full main `nanopath` recipe:

```bash
./submit/train_1gpu.sbatch configs/main.yaml
# or directly on a GPU machine: `python train.py configs/main.yaml`
```

`submit/train_1gpu.sbatch` is a prompt-aware launcher when run directly: it collects Labless run name, notes, and GitHub device login before submitting itself to SLURM, then auto-submits eligible completed full runs. Calling `sbatch submit/train_1gpu.sbatch ...` bypasses that prompt and trains without auto-submit. `configs/main.yaml` is sized for an 80 GB H100 at `train.batch_size: 128`. On smaller cards you can set `train.activation_checkpointing: true` and lower `train.batch_size` if you OOM.

The checked-in `#SBATCH` lines are specific to our MedARC cluster. On another SLURM cluster, edit those header lines once to match your queue, or run `python train.py ...` directly on an allocated GPU.

## Outputs

`prepare.py … download=True` reads `configs/main.yaml` by default and checks every path train.py will read, downloading data if specified config paths are missing.

- run outputs: `project.output_dir` (MedARC cluster default `/data/$USER/nanopath/main/...`; auto-localized default `nanopath/data/main/...`). Final probe results log to `metrics.jsonl`.
- wandb: `project.wandb_dir` (cluster default `/data/$USER/nanopath/wandb`; auto-localized default `nanopath/data/wandb`).
- parquet tile shards: `data.dataset_dir` (defaults to `/data/nanopath_parquet`).
- probe datasets: canonical shared `/data/thunder-data`, `/data/surgen`, `/data/leopard_bcr`, `/data/CPTAC-PDA`, `/data/pathorob`, and `/data/ucla-lung` roots declared in `probe.dataset_roots`.
- DINOv2 backbone weights: `~/.cache/torch/hub/checkpoints/` for the selected `model.type`.
- SLURM logs: `slurm/<jobid>.{out,err}` in the repo.
- labless source snapshot: `project.output_dir/labless_source`.
- labless submission payload: `project.output_dir/labless_submission.json`.
- labless auto-submit token: `${project.output_dir}.labless_autosubmit.json` while a prompt-armed SLURM job is running; the launcher removes it after the post-run submission attempt.
- checkpoints: rolling `latest.pt` written every `train.save_every` steps under `project.output_dir`, plus one final save at end of run. `save_every: null` (smoke) disables both; probes always get their own short-lived checkpoint regardless.

## Acknowledgements

Inspired by [nanochat](https://github.com/karpathy/nanochat). The DINOv2 backbone weights are [Meta checkpoints](https://github.com/facebookresearch/dinov2) loaded by state-dict into our own clean ViT implementation. Tile-classification and segmentation probes follow the [THUNDER benchmark](https://mics-lab.github.io/thunder/); slide-level probes follow [PathoBench](https://huggingface.co/datasets/MahmoodLab/Patho-Bench) and LEOPARD.
