# nanopath

![nanopath logo](imgs/nanopath_logo.png)

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

# train and evaluate the current main nanopath recipe
# auto-submits to Labless if config passes submission requirements and you provide run name/notes & GitHub login
RUN_DIR=$PWD/data/main/my-run
./submit/train_1gpu.sbatch configs/main.yaml output_dir=$RUN_DIR
# or directly on a GPU machine: python train.py configs/main.yaml output_dir=$RUN_DIR
```

`pyproject.toml` pins `torch` / `torchvision` against the CUDA 12.9 wheel index. If your GPU/driver needs a different CUDA build, edit the `torch` and `torchvision` lines in `pyproject.toml` before `uv sync`.

A successful model training prints periodic train lines, appends metrics to `metrics.jsonl`, and writes the final comparison artifact to `summary.json`. `configs/smoke.yaml` is simply meant to pretrain briefly and then run the fixed downstream probe suite to ensure everything works without errors.

W&B can run online or offline, but set that up before submitting a noninteractive job: either run `wandb login` once, or export `WANDB_MODE=offline`.

## Leaderboard

<a href="https://labless.dev/nano-projects/nanopath">
  <img src="https://api.labless.dev/api/nano-projects/nanopath/plot.svg" alt="Nanopath progress plot" width="1290">
</a>

`mean_probe_score`, aka `final_probe_score`, is the average of linear, knn, 16-shot, segmentation, progression, mutation, survival, and robustness. These columns summarize a 12-dataset suite derived from [THUNDER](https://mics-lab.github.io/thunder/), [PathoBench](https://github.com/mahmoodlab/patho-bench), and LEOPARD, with modifications to keep single-GPU evaluation lightweight. See [benchmarking/README.md](benchmarking/README.md) for more information.
On Labless, the run labeled `main` reflects the current GitHub `main` branch, and the run labeled `leader` reflects the GitHub `leader` branch.

### Nanopath models

| # | Description | final score | linear | knn | 16-shot | segmentation | progression | mutation | survival | robustness | Contributors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | [I-JEPA contig patch](https://github.com/MedARC-AI/nanopath/tree/leader) | 0.6444 | 0.7842 | 0.7061 | 0.6383 | 0.2891 | 0.6575 | 0.6162 | 0.5783 | 0.8855 | @NimaAsh |
| 2 | [lr-and-curation](https://labless.dev/runs/run_sub_6c6c051f71) | 0.6357 | 0.7701 | 0.7005 | 0.6120 | 0.3077 | 0.6494 | 0.6084 | 0.5758 | 0.8612 | @nevasini1 |
| 3 | [dinov2-s-kde](https://labless.dev/runs/run_sub_0d8aeb2511) | 0.6277 | 0.7555 | 0.6839 | 0.5890 | 0.3089 | 0.6418 | 0.5994 | 0.5898 | 0.8531 | @PaulScotti |

### Baselines

| # | Name | Description | final score | linear | knn | 16-shot | segmentation | progression | mutation | survival | robustness |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | GenBio-PathFM | GenBio-PathFM ViT-G/16 | **0.6917** | 0.8076 | 0.7626 | 0.6970 | 0.3234 | 0.7680 | 0.6375 | 0.5964 | 0.9412 |
| 2 | UNI-2-h | MahmoodLab UNI-2-h ViT-H/14 | 0.6782 | 0.7910 | 0.7547 | 0.6961 | 0.3270 | 0.7330 | 0.6463 | 0.6137 | 0.8637 |
| 3 | H-optimus-0 | H-optimus-0 ViT-G/14-reg | 0.6763 | 0.7995 | 0.7676 | 0.6931 | 0.3241 | 0.7004 | 0.6584 | 0.5748 | 0.8926 |
| 4 | EXAONE-Path-2.5 | LG AI Research EXAONE-Path-2.5 ViT-B/14 | 0.6606 | 0.8028 | 0.7582 | 0.6872 | 0.2820 | 0.6859 | 0.6494 | 0.5781 | 0.8409 |
| 5 | Virchow | Paige/Microsoft Virchow ViT-H/14 | 0.6591 | 0.7915 | 0.7220 | 0.6114 | 0.3206 | 0.6689 | 0.6350 | 0.6239 | 0.8994 |
| 6 | GigaPath | Prov-GigaPath tile encoder ViT-G/16 | 0.6456 | 0.7977 | 0.7149 | 0.6537 | 0.3304 | 0.7041 | 0.6262 | 0.5932 | 0.7448 |
| 7 | Midnight-12K | Kaiko Midnight-12K ViT-G/14 | 0.6204 | 0.7684 | 0.6807 | 0.5758 | 0.2722 | 0.6840 | 0.6087 | 0.5907 | 0.7823 |
| 8 | DINOv2-giant | Untouched Meta `dinov2_vitg14_reg` | 0.6196 | 0.7689 | 0.7208 | 0.5834 | 0.2826 | 0.6000 | 0.6174 | 0.5849 | 0.7985 |
| 9 | OpenMidnight | OpenMidnight ViT-G/14-reg | 0.6114 | 0.7926 | 0.7135 | 0.4335 | 0.3087 | 0.6993 | 0.6091 | 0.5907 | 0.7438 |
| 10 | DINOv2-small | Untouched Meta `dinov2_vits14_reg` | 0.5841 | 0.6968 | 0.6249 | 0.5834 | 0.2704 | 0.5827 | 0.6225 | 0.5374 | 0.7543 |
| 11 | DINOv2-small random | Randomized weights `dinov2_vits14_reg` | 0.4703 | 0.5255 | 0.5066 | 0.4139 | 0.2701 | 0.6922 | 0.5648 | 0.5984 | 0.1905 |

Baseline rows are frozen reference checkpoints evaluated with the same probe suite. They help calibrate the plot, but pathology-specific baselines are not valid initialization points for nanopath leaderboard submissions. The reference scripts live in `baselines/`.

### How to submit to the leaderboard

Labless is our public run ledger and live plot for `nanopath`. You do not need a Labless password or a pull request to make a leaderboard claim; the submitter connects your submission to your GitHub identity through GitHub's device sign-in. Submit every completed full run, including null results and incremental tweaks; the dense public ledger lets maintainers and AI agents mine cross-run patterns that isolated wins would hide.

`configs/main.yaml` is the current `nanopath` main-branch training recipe. A normal SLURM submission is:

```bash
RUN_DIR=$PWD/data/main/my-run
./submit/train_1gpu.sbatch configs/main.yaml output_dir=$RUN_DIR
```

The pipeline is:

1. Run `./submit/train_1gpu.sbatch ...` or `python train.py ...` to start your training run. For full runs, the launcher asks for a short `run_name`, notes (a description that will accompany your run on labless), and GitHub device sign-in before scheduling the GPU job. Leaving the prompts blank or failing to sign in will lead to skipping labless submission.
2. Let `train.py` finish the final probe. The run directory will contain `summary.json`, `metrics.jsonl`, and the source snapshot written at launch under `labless_source/`. The submitter writes `labless_submission.json`, checks the run caps and locked benchmark surface, posts to `api.labless.dev`, and shows the run as `pending` until maintainer validation.

Manual submission is still available for direct `python train.py` runs or copied output directories:

```bash
./labless/submit_to_labless.py output_dir=$RUN_DIR run_name=kde-crops notes="what changed and why"
```

Public full-run submissions must satisfy:

- `summary.max_train_samples == 1000000`
- `summary.tile_presentations <= 1000000`
- `summary.max_train_flops == 1e18`
- final `mean_probe_score` / `final_probe_score` is present
- no saved-source changes to `probe.py` or anything under `benchmarking/`
- no locked probe config changes except local `probe.dataset_roots`

The `run_name` is the short label shown next to your dot on the Labless plot; keep it under 20 characters and make it describe what changed. Short smoke-sized runs, failed runs, and runs missing the saved source snapshot stay local. Each verified GitHub login can submit at most 100 runs per 24 hours.

To top the leaderboard you must beat the highest validated Labless run on `mean_probe_score` by at least 0.006. Public submissions have no wall-clock limit, so train on whatever hardware you have access to. [@PaulScotti](https://github.com/PaulScotti) will inspect promising submissions, independently rerun candidates that pass this threshold on maintainer compute with a different rng seed, and validate them on Labless if training completes within 2 hours and the rerun still improves by at least 0.006. If the candidate code is pushed to nanopath `main`, Labless marks that run separately as `main`. **You don't need an H100 or a PR to submit**; labless handles the public record and maintainer validation.

Code-cleanup PRs are still welcome when they simplify the codebase without changing benchmark performance on the main recipe. Leaderboard claims should go through labless instead of a pull request.

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
- All of `probe.py` and `benchmarking/`
- All probe config variables in `configs/main.yaml`.

**Pretrained models are OK only if not pathology-specific**
You can use any pretrained model however you want, including for initialization, teachers, data curation, or preprocessing, as long as it was not originally trained on pathology-related data. DINOv2 is allowed; H-optimus-0, OpenMidnight, and other pathology-trained checkpoints are not.

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
- `benchmarking/` — supports probing/downstream evaluation.
- `prepare.py` — data prep: verify or download pretraining data + probe datasets + any pretrained weights.
- `probe.py` — downstream probes (KNN, few-shot, linear, segmentation, slide AUROC, survival, robustness).
- `submit/train_1gpu.sbatch` — SLURM launcher for single-GPU training.
- `labless/submit_to_labless.py` — package a run and post it to the live labless tracker.
- `download_TCGA.sh` — manual utility, run by hand if you want the full 12K TCGA open-access SVS slide set (~13 TB) for forking the tile-extraction recipe. Not invoked by `prepare.py` and not needed for any standard training workflow.
- `pyproject.toml` + `uv.lock` — Python dependencies used by `uv sync`.

## Data

`prepare.py` prepares the necessary data for pretraining and downstream probing. By default it reads `configs/main.yaml`; pass a YAML path before the flag to prepare a different config, e.g. `python prepare.py configs/smoke.yaml download=True`. Flag `download=True` to fetch/prepare the configured datasets into the folders specified by the YAML; flag `download=False` to verify that all required paths are already populated.

On the MedARC cluster, the checked-in `/data` and `/block` paths are the intended populated shared defaults. On a fresh clone, `prepare.py … download=True` rewrites any missing or empty checked-in data/probe roots to point into `nanopath/data/<name>`, preserving comments and formatting. It also moves `output_dir` and `wandb_dir` into `nanopath/data/` whenever a config's data roots are localized. The rewrite updates the selected config plus the checked-in `configs/main.yaml` and `configs/smoke.yaml`, so running prepare once still leaves both smoke and main directly runnable afterward. To force a different storage location, edit `data.dataset_dir`, `probe.dataset_roots.*`, `project.output_dir`, and `project.wandb_dir` to existing writable paths before downloading.

**What `download=True` does**
1. **TCGA tiles**: `huggingface_hub.snapshot_download` (filtered to `shard-*.parquet`) pulls the 200 parquet shards (~120 GB total, `{path: string, jpeg: binary}` rows with 64-row row groups) from [`medarc/nanopath`](https://huggingface.co/datasets/medarc/nanopath) into `data.dataset_dir`.
2. **Probe datasets**: for each empty configured root, fetches/unpacks and, where needed, pre-extracts the probe data from the [`medarc/nanopath`](https://huggingface.co/datasets/medarc/nanopath/tree/main/probes) probe mirror for portable noninteractive setup.
3. **DINOv2 backbone weights**: `torch.hub.load_state_dict_from_url` fetches the Meta checkpoint for `model.type` from `dl.fbaipublicfiles.com` into `~/.cache/torch/hub/checkpoints/`.

**Prerequisites**
- ~120 GB free wherever `data.dataset_dir` lives for the parquet shards (cluster default: `/data/nanopath_parquet`).
- Probe data disk varies by suite. Expect that it might take a few hours for one-time downloading all mirrored probe datasets.

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

Point `data.dataset_dir` at the packed parquet directory before training. To publish a new variant of the dataset, you can push the resulting shards to a fresh HF dataset repo and update `HF_REPO_ID` in `prepare.py`.

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

The checked-in `#SBATCH --partition=n` / `--qos=normal` lines are MedARC-specific. On another SLURM cluster, edit those header lines once to match your queue, or run `python train.py ...` directly on an allocated GPU.

## Outputs

The `/data`- and `/block`-rooted defaults below are the MedARC cluster layout; `prepare.py … download=True` rewrites missing or empty data/probe defaults in the selected config plus `configs/{main,smoke}.yaml` to live under `nanopath/data/` instead, and it localizes run outputs/W&B logs there too for those rewritten configs.

- run outputs: `project.output_dir` (MedARC cluster default `/data/$USER/nanopath/main/...`; auto-localized default `nanopath/data/main/...`). Final probe results log to `metrics.jsonl`.
- wandb: `project.wandb_dir` (cluster default `/data/$USER/nanopath/wandb`; auto-localized default `nanopath/data/wandb`).
- parquet tile shards: `data.dataset_dir` (defaults to `/data/nanopath_parquet`).
- probe datasets: `probe.dataset_roots` (defaults to shared `/block/...` and `/data/...` paths on the MedARC cluster; LEOPARD BCR defaults to `/data/leopard_bcr` with hpcroot group sharing).
- DINOv2 backbone weights: `~/.cache/torch/hub/checkpoints/` for the selected `model.type`.
- SLURM logs: `slurm/<jobid>.{out,err}` in the repo.
- labless source snapshot: `project.output_dir/labless_source`.
- labless submission payload: `project.output_dir/labless_submission.json`.
- labless auto-submit token: `${project.output_dir}.labless_autosubmit.json` while a prompt-armed SLURM job is running; the launcher removes it after the post-run submission attempt.
- checkpoints: rolling `latest.pt` written every `train.save_every` steps under `project.output_dir`, plus one final save at end of run. `save_every: null` (smoke) disables both; probes always get their own short-lived checkpoint regardless.

## Experiment log

See the live [labless nanopath log](https://labless.dev/nano-projects/nanopath) for submitted completed runs.

## Acknowledgements

Inspired by [nanochat](https://github.com/karpathy/nanochat). The DINOv2 backbone weights are [Meta checkpoints](https://github.com/facebookresearch/dinov2) loaded by state-dict into our own clean ViT implementation. Tile-classification and segmentation probes follow the [THUNDER benchmark](https://mics-lab.github.io/thunder/); slide-level probes follow [PathoBench](https://huggingface.co/datasets/MahmoodLab/Patho-Bench) and LEOPARD.
