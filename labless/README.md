# labless integration

This folder contains the nanopath-to-labless bridge. The goal is simple: after
you train a model, one command publishes the run to the public nanopath tracker.

```bash
RUN_DIR=/data/$USER/nanopath/leader/my-run
./labless/submit_to_labless.py output_dir=$RUN_DIR contributor=@yourgithub notes="what changed and why"
```

## What the submit script does

`submit_to_labless.py` is dependency-free and should be run from the nanopath
repo root after `train.py` finishes. It:

1. Reads `summary.json` and `metrics.jsonl` from `output_dir`.
2. Extracts the final `mean_probe_score` and probe submetrics.
3. Records git branch, commit, dirty files, diff summary, hardware, Python
   version, optional W&B URL, and artifact paths.
4. Writes the exact public payload to `output_dir/labless_submission.json`.
5. Posts it to `https://api.labless.dev/api/nano-projects/nanopath/submissions`.

The labless backend stores the submission as an idea, attempt, and run. The
website fetches the API data and the SVG plot from `api.labless.dev`, so the run
appears in the project log, run table, and plot without opening a pull request.

`labless.yaml` is the project manifest. It records the metric, validation
rules, run tiers, current trained leader, and frozen reference baselines used by
the public plot. You do not need to edit it before submitting a run; change it
only when the nanopath leaderboard policy itself changes.

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
  wandb_url=https://wandb.ai/... \
  notes="changed the crop schedule and kept all probe paths untouched"
```

Completed submissions require both `summary.json` and `metrics.jsonl`. The run
is shown as `pending` until the organizer validates it.
Use the same config you prepared and trained with; off the MedARC cluster, copy
the config and point its data paths at writable local storage before training.

## Submit a baseline/reference run

Baseline scripts under `baselines/` write the same `summary.json` and
`metrics.jsonl` files as `train.py`, so they can be submitted the same way:

```bash
python baselines/dinov2_small_baseline.py configs/leader.yaml
./labless/submit_to_labless.py \
  output_dir=/data/$USER/nanopath/baselines/dinov2-small \
  contributor=@yourgithub \
  notes="reran the frozen DINOv2-small reference"
```

The submit script detects `summary.family == "baseline"` and marks the run as
`tier=baseline`. Baseline points are reference context for the tracker. The
nanopath leaderboard still promotes trained `configs/leader.yaml` descendants
through maintainer validation.

## Submit a failed run

Failed attempts are useful because they tell the next contributor what not to
repeat. Submit them too:

```bash
./labless/submit_to_labless.py \
  output_dir=/data/$USER/nanopath/leader/my-failed-run \
  contributor=@yourgithub \
  status=failed \
  failure_reason="OOM after increasing batch size" \
  notes="activation checkpointing was not enough on a 24GB card"
```

Failed runs do not need a final score, but the script still includes any files
and metrics that exist.

## Useful options

Arguments are `key=value`; there is no `argparse`.

| key | use |
|---|---|
| `output_dir` | Required run directory. |
| `contributor` | GitHub/Discord handle shown on labless. |
| `notes` | Short explanation of what changed and why. |
| `wandb_url` | Optional public W&B run URL. |
| `status` | `completed` or `failed`; default is `completed`. |
| `failure_reason` | Human-readable reason for failed runs. |
| `title` | Optional display title. |
| `tier` | `smoke`, `pilot`, `full`, `replicate`, or `baseline`; inferred when omitted. |
| `hardware` | Override detected hardware string. |
| `dry_run=true` | Write `labless_submission.json` without posting. |
| `api_url` | Use a local labless backend for testing. |

If labless later enables a private submission token, set
`LABLESS_SUBMIT_TOKEN` in the environment before running the script.

## Validation rules

The benchmark score is only meaningful when evaluation stays fixed. The script
records dirty git state and marks submissions invalid if changed files include:

- `probe.py`
- anything under `benchmarking/`

Commit or stash unrelated work before submitting. Dirty training-code changes
are allowed and recorded so maintainers can inspect what produced the point.

## What becomes public

The payload intentionally makes the run inspectable. It includes:

- contributor handle and notes
- final metric and probe submetrics
- run family, recipe id, and tier (`baseline` for frozen reference scripts)
- git remote, branch, commit, dirty flag, changed files, and diff summary
- hardware, hostname, Python version, and optional GPU summary
- artifact paths or URLs for `summary.json`, `metrics.jsonl`, W&B, SLURM logs,
  and `labless_submission.json`

Local artifact paths are provenance pointers; the script does not upload model
weights or raw data.

## Maintainer validation

New completed runs appear on the plot as `pending`. A maintainer can replicate a
promising run, then mark it `validated` or `leader` in labless. Failed and
rejected runs remain visible because they are useful research context.
