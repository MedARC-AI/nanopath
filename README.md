# NanoPath

Lean single-stage computational pathology foundation model training code.

The default recipe is:

- `lazyslide` for tiling and WSI reads
- the OpenMidnight `sample_dataset_30.txt` patch list as the TCGA source of truth
- one shared RoPE-only ViT encoder
- `LeJEPA` loss proxy for geometry
- same-step detached-target latent masked prediction
- nano-to-small scaling-law fitting before the large run
- offloaded Thunder linear probes: `bach`, `break_his`, `mhist`

## Setup

```bash
cd /admin/home/paul/nanopath
uv sync
source /admin/home/paul/nanopath/.venv/bin/activate
mkdir -p /data/nanopath
```

## Main entrypoints

```bash
python train.py configs/nano.yaml
python scale.py configs/scale.yaml
```

Large outputs, checkpoints, `wandb` logs, and scaling fits live under `/data/nanopath`, not in the repo.

Fresh non-resume launches delete and recreate `project.output_dir` before training starts.
So rerunning a config such as `configs/nano.yaml` rewrites `/data/nanopath/nano` rather than aborting on stale files.

## Recommended workflow

The intended loop is:

1. Use `smoke` only for code-path sanity.
2. Use `nano` as the main battleground for training changes.
3. Promote only the promising `nano` changes to `micro`.
4. Once `micro` looks stable, run `mini` and `small` to populate the scaling fit.
5. Fit the scaling law from the completed summaries.
6. Run the largest model once the recipe is already stable, rather than sweeping there.

In practice that means:

- `nano` is where most experimentation should happen
- `micro` is the first confirmation gate
- `mini` and `small` are for scaling confirmation, not broad idea search
- the largest run should be a promotion, not an exploration run

## Data policy

The TCGA patch source is the same precomputed sample list used in OpenMidnight:

- training reads `/block/TCGA/sample_dataset_30.txt`
- each line is `slide_path x y level`
- train/val splits are assigned at patient level by hashing the TCGA barcode parsed from `slide_path`
- each sample is one listed TCGA patch, not a fresh random thumbnail-filtered crop
- validation reuses a deterministic patch and augmentation stream per index so `val/*` is comparable across evals
- the effective patch mpp comes from the listed WSI pyramid level
- line-offset caches for the train/val split are generated under `/data/nanopath/cache` on first use

## Experiment strategy

Use `nano` for:

- objective changes
- augmentation changes
- masking changes
- predictor changes
- optimizer and batch-size changes
- loader and throughput work
- fast stability checks

Promote a change to `micro` only if:

- the `nano` run is clearly healthier or better
- the probes do not regress
- the change is simple enough to justify carrying forward

Do not trust a change just because it helped on `nano`. Keep it only if `micro` agrees.

## Sweeping policy

The intended sweep policy is narrow and staged.

- Sweep on `nano`, not on `micro`, `mini`, or `small`
- Change one small cluster of knobs at a time
- Keep the sweep local to the actual uncertainty
- Use `micro` to confirm winners, not to repeat the full sweep
- Do not run broad hyperparameter searches on the large models

Good `nano` sweep targets:

- `lambda_sig`
- `lambda_lat`
- `latent_mask_ratio`
- `latent_mask_mode`
- crop ranges and augmentation strengths
- predictor width or depth
- batch size and the implied learning rate

Bad large-scale sweep targets:

- repeating the whole augmentation search on `micro`
- doing optimizer sweeps on `small`
- searching many masking schemes on the largest run

## Scale-up strategy

The intended promotion ladder is:

- `smoke` on CPU for correctness
- `nano` on `1xH100` for cheap iteration
- `micro` on `1xH100` for confirmation
- `mini` on `1xH100` or a few GPUs if needed
- `small` on the full `8xH100` node

The current configs reflect that:

- [nano.yaml](/admin/home/paul/nanopath/configs/nano.yaml) is the cheap default battleground
- [micro.yaml](/admin/home/paul/nanopath/configs/micro.yaml) is the first serious confirmation run
- [mini.yaml](/admin/home/paul/nanopath/configs/mini.yaml) is the mid-scale fit rung
- [small.yaml](/admin/home/paul/nanopath/configs/small.yaml) is the node-scale run

The checked-in `micro`/`mini`/`small` configs keep tile presentations per parameter approximately flat, so the default family is at least internally consistent before any explicit scaling fit.

The scaling fit currently reads summaries from `micro`, `mini`, and `small` in [scale.yaml](/admin/home/paul/nanopath/configs/scale.yaml). `nano` is intentionally treated as the fast experimentation tier rather than the main fit tier.

The checked-in `small` config also leaves `cbs_doubling_steps` empty on purpose. CBS promotion should come from branched measurements, not from a guessed default on the node-scale run.

### How the scaling fit works

`scale.py` does not read raw training logs. It reads the final `summary.json` files written by completed runs under `/data/nanopath/*/summary.json`.

Each completed run contributes:

- `model_params`
- `best_val_lejepa_proxy`
- `tile_presentations`
- `visible_patch_presentations`
- `masked_target_presentations`
- `train_flops`

The current implementation requires at least `6` finite completed summary points across at least `3` model sizes before it will fit at all.

If that requirement is not met, `scale.py` writes a preflight `fit.json` explaining how many eligible points and model sizes it found, then exits loudly instead of fitting junk.

The current fit config in [scale.yaml](/admin/home/paul/nanopath/configs/scale.yaml) uses:

- `best_val_lejepa_proxy` as the loss to model
- `micro`, `mini`, and `small` as the included families
- the current base-rung target size `501,669,120` params
- candidate data-budget axes:
  `tile_presentations`, `visible_patch_presentations`, `masked_target_presentations`, and `train_flops`

`scale.py` also filters out summaries with non-finite loss or invalid parameter counts before fitting, so one bad run does not poison the whole fit.

The fitted law is:

`L(N, D) = L_inf + A * N^(-alpha) + B * D^(-beta)`

where:

- `N` is model size in parameters
- `D` is one data-budget axis
- `L` is the chosen loss

Internally the code rescales `N` and `D` by their median values before fitting so the optimization is numerically better behaved.

`scale.py` then tries each candidate data axis and scores it by leave-one-out error:

- drop one completed run
- fit the law on the remaining runs
- predict the held-out loss
- measure relative error
- average that over all held-out runs

The best axis is the one with the lowest leave-one-out relative MAE. That chosen axis is then used for the final fit.

### How to apply the fit

The purpose of the fit is not to choose the recipe. The recipe should already be stable by the time you do this.

Its purpose is to answer:

- how much data budget should the next larger model get
- whether the current small-to-medium runs are internally consistent
- what frontier of model size versus recommended budget the current recipe implies

After fitting, [scale.py](/admin/home/paul/nanopath/scale.py) writes:

- `/data/nanopath/scaling/fit.json`
- `/data/nanopath/scaling/frontier.csv`

`fit.json` contains:

- the chosen axis
- the leave-one-out scores for all candidate axes
- the fitted parameters `L_inf`, `A`, `alpha`, `B`, `beta`
- `target_model_params`
- `target_data_budget`
- `target_predicted_loss`

`frontier.csv` contains the recommended budget and predicted loss for each observed model size plus the configured target model size.

The recommended budget for the target model is computed by balancing the model-size term and the data-budget term from the fitted law. In code this is the `d_opt` value in [scale.py](/admin/home/paul/nanopath/scale.py).

`target_data_budget` is expressed in the winning axis, not always in steps. For example:

- if `best_axis` is `train_flops`, then `target_data_budget` is a FLOPs budget
- if `best_axis` is `visible_patch_presentations`, then it is a visible-patch budget
- if `best_axis` is `tile_presentations`, then it is a tile budget

The checked-in training configs no longer convert that budget into a fixed `max_steps`.
They now run for a fixed `train.max_wall_seconds` budget of one in-loop hour, so the default summaries are directly comparable on equal wall-clock training time even when throughput changes across model sizes or recipe variants.

### What to do in practice

The intended usage is:

1. Run several completed `micro`, `mini`, and `small` trainings with the same `project.recipe_id`.
2. Let each run finish and write its `summary.json`.
3. Make sure you have at least `6` finite summaries spanning at least `3` distinct model sizes.
4. Run:

```bash
python scale.py configs/scale.yaml
```

5. Read `fit.json` and `frontier.csv`.
6. Compare the recommended `target_data_budget` against the achieved one-hour summaries for the next larger model.
7. Promote the current recipe to the large run without changing the checked-in one-hour stop rule unless you are intentionally running a separate longer-budget experiment.

The important constraint is that the summaries must come from comparable runs. If you mix fundamentally different recipes into the same fit, the scaling law becomes much less meaningful.

The checked-in `configs/scale.yaml` now enforces that by filtering on both:

- `family_filter`
- `recipe_filter` against `summary["recipe_id"]`

So when you branch the recipe in a materially different way, give that branch a new `project.recipe_id` before collecting scaling data.

So:

- use `nano` for idea search
- use `micro`, `mini`, and `small` for the actual scaling fit
- only fit after the recipe is mostly settled
- use the fit to allocate budget, not to justify unstable recipe changes

## Reading wandb

The most important online signals are:

- `val_thunder/mean_probe_f1`
- `val_thunder/mean_probe_acc`
- `val_thunder/probe_bach_val_f1`
- `val_thunder/probe_break_his_val_f1`
- `val_thunder/probe_mhist_val_f1`

`val_thunder/mean_probe_f1` is the simple average of the Thunder macro-F1 scores for `bach`, `break_his`, and `mhist`.
`val_thunder/mean_probe_acc` is the simple average of the per-dataset balanced accuracies exposed as `val_thunder/probe_*_val_acc`.
These are the most interpretable representation-health metrics during pretraining.
Because the probe job runs asynchronously on a separate SLURM allocation, these metrics are logged against `val_thunder/step` rather than the main training step stream.

The other validation metrics are internal training diagnostics:

- `val/pred`: JEPA view-consistency term, lower is better
- `val/sig`: anti-collapse regularizer, lower is better
- `val/latent`: latent masked-prediction loss, lower is better
- `val/lejepa_proxy`: JEPA-side recipe-selection proxy, lower is better

`val/lejepa_proxy` is useful for comparing nearby runs, but it is not a downstream benchmark metric.

Probe cadence matters:

- probe submissions only happen on eval steps
- the submission cadence is controlled by both `train.eval_every` and `probe.every`
- only the latest queued checkpoint is kept if Thunder falls behind the training loop
- for predictable submit points, keep `probe.every` equal to or a multiple of `eval_every`

Single GPU:

```bash
ssh n-1
cd /admin/home/paul/nanopath
uv sync
source /admin/home/paul/nanopath/.venv/bin/activate
mkdir -p /data/nanopath
CUDA_VISIBLE_DEVICES=0 python train.py configs/nano.yaml
```

Full node:

```bash
ssh n-1
cd /admin/home/paul/nanopath
uv sync
source /admin/home/paul/nanopath/.venv/bin/activate
mkdir -p /data/nanopath
torchrun --standalone --nproc_per_node=8 train.py configs/small.yaml
```

SLURM launcher:

```bash
sbatch /admin/home/paul/nanopath/scripts/train.sbatch
```

Edit [train.sbatch](/admin/home/paul/nanopath/scripts/train.sbatch) before submit:

- set `CONFIG_FILE` to the config you want
- set `NPROC_PER_NODE` to the matching GPU count
- keep `1` GPU for `nano` and `micro` unless there is a specific reason not to
- use the full node only when promoting to `small`

## Continuous probes

The committed training configs now offload downstream probing to Thunder on a separate `1xH100` SLURM job.
The main training loop keeps running, while the probe job:

- loads the saved NanoPath checkpoint through a Thunder custom-model adapter
- recomputes Thunder embeddings for `bach`, `break_his`, and `mhist`
- trains and evaluates Thunder linear probes in one parallel batch across those `3` datasets
- writes the result back under the run output so the main process can log it into the same `wandb` run

The Thunder job reads datasets directly from `/block/eva-data` via symlinked dataset roots:

- `bach`: `/block/eva-data/bach`
- `break_his`: `/block/eva-data/breakhis`
- `mhist`: `/block/eva-data/mhist`

## Required sample list

`train.py` expects the OpenMidnight-style TCGA sample list at:

- `/block/TCGA/sample_dataset_30.txt`

Each row must be:

- `slide_path x y level`

where `x` and `y` are level-0 coordinates and `level` is the WSI pyramid level used when reading the `224x224` base patch.

## Repo shape

- `model.py`: model and loss stack in the `nanoGPT`-style split
- `dataloader.py`: TCGA sample-list loading and JEPA view construction
- `probe.py`: asynchronous Thunder probe submission, result collection, and logging
- `thunder_adapter.py`: Thunder custom-model adapter for NanoPath checkpoints
- `train.py`: single-stage training loop
- `scale.py`: scaling-law fitting and budget recommendation
- `configs/*.yaml`: all meaningful knobs
- `scripts/*.sh`: minimal launch helpers
