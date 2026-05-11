# BreaKHis

## Role In Nanopath

`break_his` is a breast histology tile-classification probe. It contributes one scalar to `mean_probe_score`: the mean of linear, KNN, and 16-shot SimpleShot validation macro F1.

## Source

- Dataset: BreaKHis
- Download used by `prepare.py`: `http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz`
- Split provenance: EVA / Thunder 40X four-subtype protocol

## Split And Labels

Nanopath uses the checked-in split metadata in `break_his.json`.

| split | images |
|---|---:|
| train | 936 |
| val | 196 |
| test | 339 |

Only train and val are read by `probe.py`. The test split remains provenance metadata.

This is not the full 8-subtype, all-magnification BreaKHis task. Following the EVA/Thunder protocol, Nanopath uses only 40X images from the four subtypes with enough patients for a patient-disjoint split: fibroadenoma, tubular adenoma, ductal carcinoma, and mucinous carcinoma. Train, val, and test contain disjoint patient ids.

## Implementation

`probe.py` loads relative image paths from `benchmarking/break_his.json`, embeds each RGB image with model-native preprocessing from `model.py::probe_transforms`, and fits three heads on cached embeddings:

- AdamW linear probe: LR ∈ {1e-3, 1e-4, 1e-5}, weight decay 1e-4, batch size 64, 200 epochs; report the best val macro F1 across all LR × epoch checkpoints
- cosine KNN: k ∈ {1, 3, 5, 10, 20, 30, 40, 50}, k selected by val F1
- SimpleShot few-shot: 1000 deterministic 16-shot support sets per class, support/query embeddings centered by each support-set mean, class prototypes from class-specific centered support means, cosine nearest-centroid prediction, then per-query majority vote

The dataset score is `mean(linear_val_f1, knn_val_f1, fewshot_val_f1)`.

## Difference From Original Usage

BreaKHis is commonly evaluated with magnification-aware and patient-level protocols. Nanopath instead uses the fixed EVA/Thunder 40X four-subtype validation split as a lightweight representation probe and does not report an official test-set score.

## Runtime

Recent H100 timings:

| model | wall |
|---|---:|
| DINOv2-S | 15.1s |
| OpenMidnight | 20.7s |
| H-optimus-0 | 20.4s |
| GenBio-PathFM | 14.6s |
