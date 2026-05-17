# BRACS

## Role In Nanopath

`bracs` is a breast ROI classification probe. It contributes one scalar to `mean_probe_score`: the mean of linear, KNN, and 16-shot SimpleShot validation macro F1.

## Source

- Dataset: BRACS ROI
- Benchmark family: Thunder tile-classification tasks (`linear_probing`, `knn`, `simple_shot`)
- Upstream source: `ftp://histoimage.na.icar.cnr.it/BRACS_RoI/`
- Download used by `prepare.py`: `medarc/nanopath`, under `probes/bracs/`

## Split And Labels

Nanopath uses the checked-in split metadata in `bracs.json`, which mirrors the official BRACS `latest_version/{train,val,test}` folders.

| split | images |
|---|---:|
| train | 3657 |
| val | 312 |
| test | 570 |

Only train and val are read by `probe.py`. The seven labels follow the BRACS ROI folders: `N`, `PB`, `UDH`, `FEA`, `ADH`, `DCIS`, and `IC`.

## Implementation

The image adapter reads relative PNG paths from `benchmarking/bracs.json`. Trained Nanopath checkpoints use the square `Resize((224, 224))` default from `model.py::probe_transforms`; frozen baseline scripts set `probe.transform_policy`, with OpenMidnight also using THUNDER's square resize. Frozen embeddings are reused for:

- AdamW linear probe: LR ∈ {1e-3, 1e-4, 1e-5}, weight decay 1e-4, batch size 64, 200 epochs; report the best val macro F1 across all LR × epoch checkpoints
- cosine KNN: k ∈ {1, 3, 5, 10, 20, 30, 40, 50}, k selected by val F1
- SimpleShot few-shot: 1000 deterministic 16-shot support sets per class, support/query embeddings centered by each support-set mean, class prototypes from class-specific centered support means, cosine nearest-centroid prediction, then per-query majority vote

The dataset score is `mean(linear_val_f1, knn_val_f1, fewshot_val_f1)`.

## Difference From Original Usage

BRACS has its own train/validation/test organization. Nanopath uses the same train and validation folders as the official release/THUNDER split metadata, but does not score the official test folder because the leaderboard is an iterative validation benchmark and should not consume official test labels.
