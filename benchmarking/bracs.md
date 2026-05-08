# BRACS

## Role In Nanopath

`bracs` is a breast ROI classification probe. It contributes one scalar to `mean_probe_score`: the mean of linear, KNN, and SimpleShot validation macro F1.

## Source

- Dataset: BRACS ROI
- Benchmark family: Thunder tile-classification tasks (`linear_probing`, `knn`, `simple_shot`)
- Download used by `prepare.py`: `ftp://histoimage.na.icar.cnr.it/BRACS_RoI/`

## Split And Labels

Nanopath uses the checked-in split metadata in `bracs.json`.

| split | images |
|---|---:|
| train | 3657 |
| val | 312 |
| test | 570 |

Only train and val are read by `probe.py`. The seven labels follow the BRACS ROI folders: `N`, `PB`, `UDH`, `FEA`, `ADH`, `DCIS`, and `IC`.

## Implementation

The image adapter reads relative PNG paths from `benchmarking/bracs.json`. Frozen embeddings are reused for:

- AdamW linear probe: LR ∈ {1e-3, 1e-4, 1e-5}, weight decay 1e-4, batch size 64, 200 epochs; report the best val macro F1 across all LR × epoch checkpoints
- cosine KNN: k ∈ {1, 3, 5, 10, 20, 30, 40, 50}, k selected by val F1
- SimpleShot few-shot: shots ∈ {1, 2, 4, 8, 16}, 500 random support-set trials per shot with per-example majority vote; the few-shot scalar is the mean val F1 across shots

The dataset score is the mean of the three validation macro F1 scores.

## Difference From Original Usage

BRACS has its own train/validation/test organization. Nanopath uses train and validation only, because the leaderboard is an iterative validation benchmark and should not consume official test labels.

## Runtime

BRACS is one of the main bottlenecks despite its modest image count. Runtime is dominated by the three-head probe, especially the 200-epoch linear LR sweep and 500-trial few-shot computation.

| model | wall |
|---|---:|
| DINOv2-S | 209.8s |
| OpenMidnight | 185.9s |
| H-optimus-0 | 184.9s |
