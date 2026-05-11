# UCLA Lung

## Role In Nanopath

`ucla_lung` is a lung slide-level progression/regression classification probe. It contributes one scalar to `mean_probe_score`: validation AUROC from a balanced logistic linear probe on mean-pooled slide embeddings.

## Source

- Task metadata: PathoBench `ucla_lung/progression_regression`
- Raw images: IDR idr0082 Pennycuick lesions
- Download base used by `prepare.py`: `https://ftp.ebi.ac.uk/pub/databases/IDR/idr0082-pennycuick-lesions/20200517-ftp`

## Split And Tiles

Nanopath uses `ucla_lung.json`, derived from PathoBench fold 0. The 90 fold-0 train slides are evaluated with deterministic stratified 3-fold validation; the 22 fold-0 test slides remain provenance metadata and are not read by `probe.py`. Case ids are unique across train, val, and test.

| split | slides | cached tiles |
|---|---:|---:|
| train pool | 90 | full 20x/512 tissue grid |
| per-fold train | 60 | reused |
| per-fold val | 30 | reused |
| test | 22 | not cached |

Only train and val are read by `probe.py`.

## Implementation

`prepare.py` downloads fold-0 train NDPIs and extracts a deterministic 20x, 512 px, 0-overlap tissue grid into per-slide parquet caches, then concatenates those rows into `tiles.parquet`. A `pathobench_20x_512_v1` marker makes older capped or differently tiled caches fail verification and regenerate. `probe.py` embeds every cached tile once with a no-crop square resize, mean-pools tile embeddings per slide, then for each fold fits a balanced logistic linear probe (`sklearn.linear_model.LogisticRegression`, `class_weight="balanced"`, `max_iter=5000`) over `C ∈ {0.001, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0}`, averages val AUROC across the three folds at each `C`, and reports the best mean.

## Difference From Original Usage

PathoBench reports macro one-vs-rest AUROC for this task; with fold 0's two-class labels this equals binary AUROC. PathoBench runs linear probing on mean-pooled Trident features; Nanopath's balanced logistic linear probe matches that head class on custom-backbone features. Nanopath uses repeated train-derived validation from fold 0 and follows the uncapped Trident-style patch-grid contract for slide pooling; it does not report the PathoBench test-fold score. The tissue mask is a lightweight deterministic thumbnail mask rather than Trident HEST segmentation.

## Runtime

| model | wall |
|---|---:|
| DINOv2-S | 31.7s |
| OpenMidnight | 67.4s |
| H-optimus-0 | 63.6s |
| GenBio-PathFM | 140.3s |
