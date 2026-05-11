# PanNuke

## Role In Nanopath

`pannuke` is a multi-organ nucleus segmentation probe. It contributes one scalar to `mean_probe_score`: validation macro Jaccard.

## Source

- Dataset: PanNuke
- Benchmark family: Thunder segmentation task
- Download used by `prepare.py`: `https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_<N>.zip`

## Split

Nanopath uses fixed PanNuke folds:

| split | source fold | images |
|---|---|---:|
| train | Fold1 | 2656 |
| val | Fold2 | 2523 |

Fold3 may be downloaded by `prepare.py`, but it is not part of `mean_probe_score`.

## Implementation

`probe.py` reads `images.npy` and `masks.npy` by memory map, derives integer class labels from the mask channels, extracts frozen patch tokens, trains the shared MaskTransformer decoder for 30 epochs, selects by validation dice loss, and reports validation macro Jaccard with the Thunder-compatible background-only weighting.

## Difference From Original Usage

PanNuke is commonly evaluated with fold-based protocols across all folds. Nanopath fixes Fold1/Fold2 so every training run uses the same fast validation probe and leaves Fold3 out of the leaderboard score.

## Runtime

PanNuke is one of the most expensive tasks, but it runs in a background thread and partially overlaps with classification and slide probes.

| model | wall |
|---|---:|
| DINOv2-S | 165.3s |
| OpenMidnight | 319.4s |
| H-optimus-0 | 309.5s |
| GenBio-PathFM | 279.8s |
