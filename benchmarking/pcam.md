# PCam

## Role In Nanopath

`pcam` is a lymph-node metastasis tile-classification probe derived from PatchCamelyon. It contributes one scalar to `mean_probe_score`: the mean of linear, KNN, and SimpleShot validation macro F1.

## Source

- Dataset: PatchCamelyon
- Benchmark family: Thunder tile-classification tasks (`linear_probing`, `knn`, `simple_shot`)
- Download used by `prepare.py`: `https://zenodo.org/api/records/2546921/files`

## Split And Labels

PCam does not use a checked-in JSON split. `probe.py` reads the official H5 files and takes deterministic subsets:

| split | source file split | images used |
|---|---|---:|
| train | `train` | 3072 |
| val | `valid` | 768 |

The official test H5 files may be downloaded for completeness, but `probe.py` never reads them.

## Implementation

`ClassificationDataset(..., dataset="pcam")` samples fixed train and validation subsets with `PCAM_SUBSET_SEED = 1337`, embeds those images, and runs the same linear / KNN / SimpleShot probe heads as the other tile classifiers.

## Difference From Original Usage

Thunder lists the full official train/valid/test sets for PCam. Nanopath deliberately uses a small deterministic train/valid subset from those official H5 files so the full 12-dataset probe remains inside the final H100 window. This is a runtime adaptation, not an exact full-sample Thunder PCam run.

## Runtime

| model | wall |
|---|---:|
| DINOv2-S | 42.8s |
| OpenMidnight | 60.4s |
| H-optimus-0 | 63.9s |
