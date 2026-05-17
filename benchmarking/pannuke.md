# PanNuke

## Role In Nanopath

`pannuke` is a multi-organ nucleus segmentation probe. It contributes one scalar to `mean_probe_score`: validation macro Jaccard.

## Source

- Dataset: PanNuke
- Benchmark family: Thunder segmentation task
- Upstream source: `https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_<N>.zip`
- Download used by `prepare.py`: `medarc/nanopath`, under `probes/pannuke/`

## Split

Nanopath uses fixed PanNuke folds:

| split | source fold | images |
|---|---|---:|
| train | Fold1 | 2656 |
| val | Fold2 | 2523 |

Fold3 is not downloaded by `prepare.py` and is not part of `mean_probe_score`.

## Implementation

`probe.py` reads `images.npy` and `masks.npy` by memory map, derives integer class labels from the mask channels, extracts frozen patch tokens, trains the shared MaskTransformer decoder for 30 epochs, selects by validation dice loss, and reports validation macro Jaccard with the Thunder-compatible background-only weighting.

## Difference From Original Usage

PanNuke is commonly evaluated with fold-based protocols across all folds. Nanopath fixes Fold1/Fold2 so every training run uses the same fast validation probe and leaves Fold3 out of the leaderboard score.
