# MoNuSAC

## Role In Nanopath

`monusac` is a nucleus segmentation probe. It contributes one scalar to `mean_probe_score`: validation macro Jaccard.

## Source

- Dataset: MoNuSAC train images and annotations
- Download used by `prepare.py`: Google Drive file id `1lxMZaAPSpEHLSxGA9KKMt_r-4S8dwLhq`

## Split

Nanopath uses only the official train package. It creates deterministic 3-fold slide-disjoint validation splits with `SEG_SPLIT_SEED = 1337`.

| split | slides | images |
|---|---:|---:|
| train pool | 46 | 209 |
| per-fold train | 30-31 | ~139 |
| per-fold val | 15-16 | ~70 |

## Implementation

`prepare.py` rasterizes XML polygon annotations into `.npy` label maps. `probe.py` resizes images and masks to 256x256, extracts frozen patch tokens from the center 224x224 crop once, trains a small MaskTransformer decoder for 30 epochs on each fold, selects the head by validation dice loss, and reports mean validation macro Jaccard.

## Difference From Original Usage

MoNuSAC is a challenge dataset with held-out evaluation data. Nanopath does not use challenge test data; it builds an internal train/validation split from the train package. The MaskTransformer head and per-image macro Jaccard come from Thunder; MoNuSAC is not in Thunder's standard suite, so this is the Thunder seg-head applied to a non-Thunder dataset.

## Runtime

| model | wall |
|---|---:|
| DINOv2-S | 24.9s |
| OpenMidnight | 92.6s |
| H-optimus-0 | 91.5s |
| GenBio-PathFM | 34.5s |
