# MHIST

## Role In Nanopath

`mhist` is a colorectal polyp tile-classification probe. It contributes one scalar to `mean_probe_score`: the mean of linear, KNN, and SimpleShot validation macro F1.

## Source

- Dataset: MHIST
- Benchmark family: Thunder tile-classification tasks (`linear_probing`, `knn`, `simple_shot`)
- Upstream access page: `https://bmirds.github.io/MHIST/`
- Portable setup mirror used by `prepare.py`: `medarc/nanopath` under `probes/mhist/`

`prepare.py download=True` prints that users must complete MHIST's Dataset Research Use Agreement before using the mirrored files.

## Split And Labels

Nanopath uses the checked-in split metadata in `mhist.json`.

| split | images |
|---|---:|
| train | 1743 |
| val | 432 |
| test | 977 |

Only train and val are read by `probe.py`. Train and val are a deterministic split of MHIST's official training partition; the official test partition is kept as provenance metadata and is not scored.

## Implementation

The probe embeds MHIST RGB images with the frozen backbone, then runs the same linear / KNN / SimpleShot heads as the other tile classifiers. The dataset score is the mean validation macro F1 across those three heads.

## Difference From Original Usage

MHIST ships with its own agreement-gated access path and task framing. Nanopath uses a checked-in split of the official training partition for fast frozen-backbone validation and keeps test metadata out of `mean_probe_score`.

## Runtime

| model | wall |
|---|---:|
| DINOv2-S | 18.4s |
| OpenMidnight | 29.1s |
| H-optimus-0 | 28.3s |
