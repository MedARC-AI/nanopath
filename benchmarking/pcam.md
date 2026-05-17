# PCam

## Role In Nanopath

`pcam` is a lymph-node metastasis tile-classification probe derived from PatchCamelyon. It contributes one scalar to `mean_probe_score`: the mean of linear, KNN, and 16-shot SimpleShot validation macro F1.

## Source

- Dataset: PatchCamelyon
- Benchmark family: Thunder tile-classification tasks (`linear_probing`, `knn`, `simple_shot`)
- Upstream source: `https://zenodo.org/api/records/2546921/files`
- Download used by `prepare.py`: `medarc/nanopath`, under `probes/pcam/`

## Split And Labels

PCam does not use a checked-in JSON split. `probe.py` reads the official H5 files and takes deterministic subsets:

| split | source file split | images used |
|---|---|---:|
| train | `train` | 3072 |
| val | `valid` | 768 |

The HF mirror stores only those deterministic train/valid subset H5s. `probe.py` never reads the official test H5 files.

## Implementation

`ClassificationDataset(..., dataset="pcam")` samples fixed train and validation subsets with `PCAM_SUBSET_SEED = 1337`, embeds those images with Nanopath's default transform or the baseline script's explicit `probe.transform_policy`, and fits three heads on cached embeddings:

- AdamW linear probe: LR ∈ {1e-3, 1e-4, 1e-5}, weight decay 1e-4, batch size 64, 200 epochs; report the best val macro F1 across all LR × epoch checkpoints
- cosine KNN: k ∈ {1, 3, 5, 10, 20, 30, 40, 50}, k selected by val F1
- SimpleShot few-shot: 1000 deterministic 16-shot support sets per class, support/query embeddings centered by each support-set mean, class prototypes from class-specific centered support means, cosine nearest-centroid prediction, then per-query majority vote

The dataset score is `mean(linear_val_f1, knn_val_f1, fewshot_val_f1)`.

## Difference From Original Usage

Thunder lists the full official train/valid/test sets for PCam. Nanopath deliberately uses a small deterministic train/valid subset from those official H5 files so the full 11-dataset probe remains inside the final H100 window. This is a runtime adaptation, not an exact full-sample Thunder PCam run.
