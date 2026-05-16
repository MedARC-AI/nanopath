# labless integration

This folder contains the nanopath-to-labless bridge. The goal is simple: after
you train a model, one command publishes the run to the public nanopath tracker.

```bash
RUN_DIR=/data/$USER/nanopath/leader/my-run
./labless/submit_to_labless.py output_dir=$RUN_DIR contributor=@yourgithub run_name=kde-crops notes="what changed and why"
```

## What the submit script does

`submit_to_labless.py` should be run from the nanopath repo root after
`train.py` finishes. It:

1. Reads `summary.json` and `metrics.jsonl` from `output_dir`.
2. Extracts the final `mean_probe_score` and probe submetrics.
3. Downloads the existing `nanopath-source-<wandb_run_id>` W&B code artifact
   and diffs that source against the current main commit for `train.py`,
   `model.py`, `dataloader.py`, `prepare.py`, and the config YAML used by the
   run.
4. Records hardware, Python version, optional W&B URL, and artifact paths.
5. Writes the exact public payload to `output_dir/labless_submission.json`.
6. Posts it to `https://api.labless.dev/api/nano-projects/nanopath/submissions`.

The labless backend stores the submission as a run with artifact pointers. The
website fetches the API data and the SVG plot from `api.labless.dev`, so the run
appears in the project log, run table, and plot without opening a pull request.

## Submit a completed run

Run training first:

```bash
RUN_DIR=/data/$USER/nanopath/leader/my-run
sbatch submit/train_1gpu.sbatch configs/leader.yaml output_dir=$RUN_DIR
# or directly on a GPU machine:
python train.py configs/leader.yaml output_dir=$RUN_DIR
```

Then point the submit script at the same run directory:

```bash
./labless/submit_to_labless.py \
  output_dir=$RUN_DIR \
  contributor=@yourgithub \
  run_name=kde-crops \
  wandb_url=https://wandb.ai/... \
  notes="changed the crop schedule and kept all probe paths untouched"
```

Completed submissions require both `summary.json` and `metrics.jsonl`. The run
is shown as `pending` until the organizer validates it. A copied config such as
`configs/new_config.yaml` is accepted if the completed `summary.json` reports
the full `max_train_flops: 1e18` budget; short local configs are rejected even
if they are not named smoke.
Use the same config you prepared and trained with; off the MedARC cluster, copy
the config and point its data paths at writable local storage before training.
Smoke runs are local setup checks only and are not accepted by labless.

## Submit a baseline/reference run

Tracked reference baseline scripts write the same `summary.json` and
`metrics.jsonl` files as `train.py`, so they can be submitted the same way:

```bash
python baselines/dinov2_small_baseline.py configs/leader.yaml
./labless/submit_to_labless.py \
  output_dir=/data/$USER/nanopath/baselines/dinov2-small \
  contributor=@yourgithub \
  notes="reran the frozen DINOv2-small reference"
```

The submit script detects `summary.family == "baseline"` and marks the run as
`tier=baseline`. Labless currently tracks GenBio-PathFM plus DINOv2 giant and
small references; other nanopath baselines can stay in the repo README without
becoming Labless reference rows. The nanopath leaderboard still ranks validated
trained `configs/leader.yaml` descendants by score.

## Useful options

Arguments are `key=value`; there is no `argparse`.

| key | use |
|---|---|
| `output_dir` | Required run directory. |
| `contributor` | GitHub/Discord handle shown on labless. |
| `run_name` | Short plot label, 20 characters or fewer. |
| `notes` | Short explanation of what changed and why. |
| `wandb_url` | W&B run URL; optional for new runs whose `summary.json` already records `wandb.url`. |
| `tier` | `full` or `baseline`; inferred when omitted. |
| `hardware` | Override detected hardware string. |
| `dry_run=true` | Write `labless_submission.json` without posting. |
| `api_url` | Use a local labless backend for testing. |
| `main_run_id` / `main_commit` | Local testing override for the live main lookup; both are required when either is set. |

If labless later enables a private submission token, set
`LABLESS_SUBMIT_TOKEN` in the environment before running the script.

## Validation rules

The benchmark score is only meaningful when evaluation stays fixed. The script
marks submissions invalid if the saved W&B source artifact changed:

- `probe.py`
- anything under `benchmarking/`

The current checkout can change after training; labless uses the W&B source
artifact from the run, not the present working tree, when building the review
diff.

## What becomes public

The payload intentionally makes the run inspectable. It includes:

- contributor handle and notes
- final metric and probe submetrics
- run family, recipe id, and tier (`baseline` for frozen reference scripts)
- W&B source artifact, git remote, commit, changed review files, and a capped
  review-file snapshot for selectable diffs
- hardware, hostname, Python version, and optional GPU summary
- artifact paths or URLs for `summary.json`, `metrics.jsonl`, W&B, SLURM logs,
  and `labless_submission.json`

The patch is only collected for `train.py`, `model.py`, `dataloader.py`,
`prepare.py`, and the config YAML used by the run. If none of those files differ
from main, no main patch is sent. The capped review-file snapshot lets
labless diff a run against other logged non-baseline runs. Binary or
large-file suffixes are omitted from patches and listed in the payload. Local
artifact paths are provenance pointers; the script does not upload model
weights or raw data.

## Maintainer validation

New completed full runs appear on the plot as `pending`. A maintainer can
replicate a promising run, then mark it `validated` in labless. The public
leader label is the highest scoring validated run. Maintainers mark a separate
`main` state with the full git commit pushed to the project repo, so the submit
script can diff the saved W&B source artifact directly against current main.
Failed runs are not accepted as public submissions.
