# labless integration

This folder contains the nanopath-to-labless bridge. The goal is simple: full
SLURM runs launched through `submit/train_1gpu.sbatch` can publish automatically
after training, while direct runs can still be submitted with one command.

```bash
RUN_DIR=$PWD/data/main/my-run
./labless/submit_to_labless.py output_dir=$RUN_DIR \
    run_name=kde-crops \
    notes="vs main: larger local crops to retain tissue context"
```

## What the submit script does

`submit_to_labless.py` should be run from the nanopath repo root after
`train.py` finishes. It:

1. Reads `summary.json` and `metrics.jsonl` from `output_dir`.
2. Extracts `final_score` plus diagnostic probe metrics.
3. Uses the local `output_dir/labless_source` snapshot written by `train.py` and
   diffs that source against the current main commit for `train.py`, `model.py`,
   `dataloader.py`, `prepare.py`, and the config YAML used by the run.
4. Records hardware, Python version, optional W&B run link, and the full changed
   path list from the saved source snapshot.
5. Writes the submission payload to `output_dir/labless_submission.json`.
6. Opens GitHub's device sign-in flow, or uses the preauthorized token file
   written by `submit/train_1gpu.sbatch`, and posts it to the Labless nanopath
   project.

The labless backend stores the submission as a run with saved source context and
an optional W&B run link. It derives the public contributor from the verified
GitHub login, rejects scoped OAuth tokens, and accepts at most 100 submissions per
login per 24 hours. The website fetches the API data and the SVG plot from
`api.labless.dev`, so the run appears in the project log, run table, and plot
without opening a pull request.

## Submit a completed run

For SLURM runs, use the prompt-aware launcher:

```bash
RUN_DIR=$PWD/data/main/my-run
./submit/train_1gpu.sbatch configs/main.yaml output_dir=$RUN_DIR
```

For configs with `max_train_samples=1000000`, `max_train_flops=1e18`, and
probes enabled, the launcher asks for a Labless run name, optional experiment
note, and GitHub device sign-in before scheduling the GPU job. If the run name is skipped or login
does not complete, the job still trains but does not auto-submit. Plain
`sbatch submit/train_1gpu.sbatch ...` also trains without auto-submit because
there is no interactive prompt before scheduling.

For direct GPU runs, train first:

```bash
RUN_DIR=$PWD/data/main/my-run
python train.py configs/main.yaml output_dir=$RUN_DIR
```

Then point the submit script at the same run directory:

```bash
./labless/submit_to_labless.py \
    output_dir=$RUN_DIR \
    run_name=kde-crops \
    notes="vs main: larger local crops to retain tissue context"
```

Completed submissions require both `summary.json` and `metrics.jsonl`. The run
is shown as `unvalidated` until the organizer validates it. A copied config such as
`configs/new_config.yaml` is accepted if the completed `summary.json` reports
`max_train_samples: 1000000`, `tile_presentations <= 1000000`, and
`max_train_flops: 1e18`; short local configs are rejected even if they are not named smoke.
Use the same config you prepared and trained with. The 12-task THUNDER
classification panel, PanNuke plus two-task SegPath panel, and canonical
train/validation roots are locked together.
Smoke runs are local setup checks only and are not accepted by labless.

## Submit a baseline/reference run

Tracked reference baseline scripts write the same `summary.json` and
`metrics.jsonl` files as `train.py`, so they can be submitted the same way:

```bash
python baselines/dinov2_small_baseline.py configs/main.yaml
./labless/submit_to_labless.py \
  output_dir=$PWD/data/baselines/dinov2-small \
  notes="reran the frozen DINOv2-small reference"
```

The submit script detects `summary.family == "baseline"` and marks the run as
`tier=baseline`. Labless uses UNI-2-h and Virchow as pathology references;
DINOv2 giant, large, and small provide natural-image references. Other baselines,
including GigaPath, can stay in the repo README without becoming Labless
reference rows. The nanopath leaderboard still ranks validated completed full
runs by score.

## Useful options

Arguments are `key=value`; there is no `argparse`.

| key | use |
|---|---|
| `output_dir` | Required run directory. |
| `run_name` | Short plot label, 20 characters or fewer. |
| `notes` | Starting recipe, main change, and why it might affect probes. |
| `wandb_url` | Optional W&B run URL for linking the external dashboard; private or unlisted W&B URLs are accepted because labless only validates URL shape. |
| `tier` | `full` or `baseline`; inferred when omitted. |
| `hardware` | Override detected hardware string. |
| `source_dir` / `source_commit` | Dry-run repair knobs only; real submissions use `output_dir/labless_source` and summary git metadata. |
| `review_config` | Repo-relative `configs/*.yaml` to review when `summary.config_path` points at an external launched copy. |
| `dry_run=true` | Write `labless_submission.json` without posting. |
| `login_only=true` / `token_output` | Internal launcher path: perform GitHub device sign-in before SLURM scheduling and write a mode-600 token file. |
| `github_token_file` | Internal launcher path: submit after training with the preauthorized token file instead of prompting inside the compute job. |
| `api_url` | Use a local labless backend for testing. |
| `main_run_id` / `main_commit` | Dry-run testing override only; real submissions compare against the current official GitHub `main`. |

For real submissions, the script prompts you to open
`https://github.com/login/device` and enter a short code. Dry runs write the
payload locally without signing in or posting.

## Validation rules

The benchmark score is only meaningful when evaluation stays fixed. The script
marks submissions invalid if the saved source snapshot changed:

- `probe.py`
- anything under `benchmarking/`

The config YAML may only change train/model/data tunables and local
`probe.dataset_roots`. Labless rejects changes to the locked probe suite keys
such as dataset lists, probe count, and model weights. Helper code changes must
stay inside the flat reviewed surface (`train.py`, `model.py`, `dataloader.py`,
`prepare.py`, or the config YAML used by the run); hidden helper modules such as `losses.py`
are rejected.

The current checkout can change after training; labless uses the source snapshot
from the run, not the present working tree, when building the review diff. W&B
may be online or offline because source review never depends on the W&B API.
`AGENTS.md` and `CLAUDE.md` are ignored when source changes are packaged.

## What becomes public

The payload intentionally makes the run inspectable. It includes:

- verified GitHub login and notes
- final metric and canonical family metrics:
  `classification_mean_f1`, `seg_mean_f1`, `slide_mean_auc`, `auc_mean`,
  `survival_mean_cindex`, and `robustness_mean`, plus CRoMa diagnostic cells
- run family, recipe id, and tier (`baseline` for frozen reference scripts)
- source snapshot id, optional git remote, commit, full changed source path list,
  changed review files, and a capped review-file snapshot for server-built diffs
- hardware, Python version, and optional GPU summary
- W&B run URL

The public API redacts local machine paths, hostnames, users, repo roots, and
local artifact paths from submitted rows.

The final score weights classification, segmentation, progression, mutation,
survival, and CRoMa robustness at 30%, 15%, 17.5%, 15%, 7.5%, and 15%.
Labless rejects scores that do not match those submitted components.

Agents can crawl the public experiment ledger directly with the JSON API:

```bash
curl -fsS "https://api.labless.dev/api/nano-projects/nanopath-v2/experiment-log?limit=100" \
  | jq '.runs[] | {run_id, title, validation, metric_value, summary}'
```

Fetch the first page, inspect `runs[]`, then follow a row's `api_url` for full
run-detail JSON or `review_patch_url` for a source patch when one is available.
Use `next_after_updated_at` plus `next_after_run_id` to request the next page.
When there is no next page, save `watermark_updated_at` plus `watermark_run_id`
and send them later to poll for newly submitted, renamed, re-noted, or
revalidated runs. Labless does not currently expose raw console logs or full
per-step `metrics.jsonl` histories.

The review snapshot is only collected for `train.py`, `model.py`,
`dataloader.py`, `prepare.py`, and the config YAML used by the run. Labless
builds capped patches server-side when it compares two logged snapshots. Binary
or large-file suffixes are omitted from patches and listed in the payload. Local
hostnames, users, working directories, repo roots, artifact paths, model weights,
and raw data are not posted.

## Maintainer validation

New completed full runs appear as `unvalidated`. A maintainer reruns promising
candidates three times with different randomly selected training seeds. The
median run becomes validated leader if it beats the incumbent by at least
0.004; `robust-norm-s9876` is the approved exception. Maintainers separately
mark the run whose commit is on `main`. Failed runs are not accepted.

Historical v2 runs without retained weights are marked as estimates. Their
robustness is the original quality-adjusted value minus 0.3063356348369218, the
mean difference measured on the six plotted nanopath checkpoints (SD 0.01637).
The five predictive components are unchanged and the new six-family weights
are applied. Estimated points are labeled with ≈ and cannot establish a leader.
New submissions must provide measured `croma_mean` and satisfy
`robustness_mean == (1 + croma_mean) / 2`; historical estimates are a maintainer
migration exception. Original scores remain available in each run's metrics.
