# CoNSeP

## Role In Nanopath

`consep` is a colorectal nucleus segmentation probe. It contributes one scalar to `mean_probe_score`: validation macro Jaccard.

## Source

- Dataset: CoNSeP
- Upstream page: `https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet/`
- Portable setup mirror used by `prepare.py`: `medarc/nanopath` under `probes/consep/`

`prepare.py download=True` prints that users must satisfy the official Warwick access terms before using the mirrored files.

## Split

Nanopath uses the official `Train` folder with deterministic 3-fold validation (`SEG_SPLIT_SEED = 1337`).

| split | ROIs |
|---|---:|
| train pool | 27 |
| per-fold train | 18 |
| per-fold val | 9 |

The archive may include `Test`, but `probe.py` does not read it.

## Implementation

`probe.py` loads RGB PNGs and MATLAB `type_map` labels, remaps CoNSeP classes through `CONSEP_REMAP`, resizes to 256x256, extracts frozen patch tokens once, trains the shared MaskTransformer segmentation head on each fold, and reports the mean validation macro Jaccard.

## Difference From Original Usage

CoNSeP is often used with its official Train/Test split. Nanopath uses only repeated folds of Train, preserving Test for non-iterative evaluation outside `mean_probe_score`. The MaskTransformer head and per-image macro Jaccard come from Thunder; CoNSeP is not in Thunder's standard suite, so this is the Thunder seg-head applied to a non-Thunder dataset.

## Runtime

| model | wall |
|---|---:|
| DINOv2-S | 5.1s |
| OpenMidnight | 11.7s |
| H-optimus-0 | 18.5s |
| GenBio-PathFM | 5.9s |
