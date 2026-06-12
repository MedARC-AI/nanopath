# Benchmarking
This folder contains code specific for probing/downstream evaluation. The normal nanopath loop is to train within the 1,000,000-sample and 1e18-FLOP caps, freeze the model backbone, run the fixed downstream probe suite, and use that result to decide whether a training idea is worth scaling. The benchmark definition stays fixed for fair model comparisons. For the most part we borrow the same approach used by THUNDER / PathoBench / LEOPARD for downstream evaluations. Official test splits from these downstream datasets not used at all for our benchmarking, making models trained using nanopath still valid submissions for official benchmarking with reduced risk of overfitting to the test sets.

## Metric

`mean_probe_score` is the unweighted mean of the eight different probe task types.

```text
mean_probe_score = mean(
  linear, knn, 16-shot,
  segmentation,
  progression,
  mutation,
  survival,
  robustness
)
```

## Probe Families

| family | datasets | dataset scalar |
|---|---|---|
| Tile classification | `break_his`, `bracs`, `mhist`, `pcam` | separate column means for linear, KNN, and 16-shot SimpleShot macro F1 |
| Segmentation | `monusac`, `consep`, `pannuke` | macro Jaccard from a small MaskTransformer head on frozen patch tokens |
| Progression | `ucla_lung` | AUROC from a balanced logistic linear probe on mean-pooled slide embeddings |
| Mutation | `surgen` | AUROC for PathoBench SR386 RAS mutation status |
| Survival | `leopard_bcr`, `cptac_pda_os` | mean Harrell's c-index from train-fold z-scored fixed-ridge CoxPH survival probes |
| Robustness | `pathorob` | mean PathoROB-style robustness index across camelyon and tolkach_esca |

All probes keep the backbone frozen. Probe heads are intentionally small: they measure representation quality, not downstream fine-tuning capacity.

Probe heads consume each model's native frozen feature dimension rather than projecting every backbone to a common width. This intentionally evaluates each checkpoint as a deployed feature extractor, but it also means dimensionality is part of the baseline comparison: DINOv2-small emits 384-d features, DINOv2-G/OpenMidnight/H-optimus-0/UNI-2-h/GigaPath emit 1536-d features, Virchow emits 2560-d CLS plus mean-patch features, Midnight-12K emits 3072-d CLS plus mean-patch features, and GenBio-PathFM emits 4608-d features.

Linear, KNN, segmentation-head, and logistic hyperparameters are selected on the same internal validation splits that define their columns. Thunder-derived tile classifiers keep Thunder-style linear/KNN/16-shot SimpleShot heads; SimpleShot precomputes 1000 deterministic support sets and majority-votes query predictions. PathoBench-derived slide classifiers use balanced logistic linear probing; SurGen uses sklearn's `liblinear` solver. Survival mean-pools tiles to slides and slides to cases, applies train-fold z-scoring, then fits fixed `CoxPHSurvivalAnalysis(alpha=2.0)` on the full pooled feature matrix.

## Runtime Strategy

The full suite is designed to stay lightweight for the standard small Nanopath model by keeping the benchmark small where it can be small, and precomputing expensive slide tiling.

- Whole-slide tasks use cached tile grids, so the final probe embeds JPEG/parquet tiles rather than opening full WSIs. PathoBench-derived slide tasks use a 20x, 512 px, 0-overlap tissue grid following the Trident/PathoBench tutorial contract. UCLA Lung and CPTAC-PDA OS embed the full cached grid; SurGen and LEOPARD BCR stream deterministic up-to-768-tile raster-spaced sub-bags per slide so larger slide tasks fit the final-probe window. Probe caches are downloaded from the `medarc/nanopath` HF mirror during normal setup; maintainer-only source rebuild helpers exist in `prepare.py` for caches that were generated from official WSIs. The remaining preprocessing simplification is a deterministic thumbnail tissue mask instead of invoking Trident's HEST segmentation model during cache construction.
- PCam is a fixed subset of the official train/valid H5 files, mirrored as small H5s with the same filenames/schema.
- Tile classifiers use `Resize((224, 224))` from `model.py::probe_transforms` for trained Nanopath checkpoints. Frozen baseline scripts set their own `probe.transform_policy`. Patch-cache probes keep square resize because their inputs are already extracted tissue tiles.
- Segmentation runs after the worker-backed embedding probes, avoiding DataLoader forks while a CUDA-using segmentation thread is live.
- The same loaded frozen backbone serves every probe in one subprocess, avoiding repeated model load overhead.
- Official held-out THUNDER and PathoBench test splits are never used. PathoBench-derived probes use fold-0 train pools with deterministic train-derived validation folds.

Runtime depends on backbone size, feature dimensionality, cache warmth, and CPU decode/head-training throughput. The dataset summary below gives small-model reference times as a rough guide; high-dimensional survival heads can be CPU-bound.

## Dataset Summary

| dataset | task | tissue / organ | train units | val units | train tiles/images | val tiles/images | reference time | source | Nanopath adaptation |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `break_his` | tile classification (~700×460 microscope captures) | breast | 936 images | 196 images | 936 | 196 | 20.4s | BreaKHis 40X / EVA-Thunder | Patient-disjoint 4-subtype 40X split; linear/KNN/16-shot SimpleShot; no test scoring |
| `bracs` | ROI classification (variable-size WSI crops) | breast | 3657 ROIs | 312 ROIs | 3657 | 312 | 179.9s | BRACS ROI FTP | Frozen-embedding linear/KNN/16-shot SimpleShot; no official test scoring |
| `mhist` | tile classification (224 px patches) | colorectal polyps | 1743 images | 432 images | 1743 | 432 | 26.3s | MHIST | Official train partition split internally; linear/KNN/16-shot SimpleShot; official test not read |
| `pcam` | tile classification (96 px patches) | lymph node metastasis | 3072 images | 768 images | 3072 | 768 | 50.0s | PCam Zenodo | Fixed train/valid subset; linear/KNN/16-shot SimpleShot; official test not read |
| `monusac` | segmentation | multi-organ nuclei | ~31 slides/fold | ~15 slides/fold | 209 total images | 3 folds | 91.5s | MoNuSAC train set | Deterministic 3-fold slide split of train package; no test data |
| `consep` | segmentation | colorectal nuclei | 18 ROIs/fold | 9 ROIs/fold | 27 total images | 3 folds | 18.5s | CoNSeP | Deterministic 3-fold split of official Train folder; Test folder not read |
| `pannuke` | segmentation | multi-organ nuclei | Fold1 | Fold2 | 2656 images | 2523 images | 309.5s | PanNuke folds | Fixed Fold1/Fold2 protocol; Fold3 not scored |
| `ucla_lung` | slide progression classification | lung | 60 slides/fold | 30 slides/fold | full 20x/512 grid | 3 folds | 63.6s | PathoBench `ucla_lung/progression_regression` / IDR idr0082 | 3-fold balanced logistic AUROC over fold-0 train using the full tissue grid; 22-slide test fold held out |
| `surgen` | mutation classification | colorectal | ~207 slides/fold | ~104 slides/fold | 1,167,089 cached tiles; up to 768 embedded/slide | 3 folds | 386.8s | PathoBench SR386 / SurGen, mirrored as pre-extracted HF parquet | 3-fold validation over PathoBench fold-0 train; fold-0 test sealed |
| `leopard_bcr` | survival | prostate | 116 cases/fold | 58 cases/fold | 133,632 cached tiles; 768 embedded/slide | 162.0s | LEOPARD Grand Challenge public training set / official S3 | all 87 recurrence events plus 87 longest-follow-up censored controls; balanced subset, not official full-cohort leaderboard; 3-fold train-z-scored fixed-ridge CoxPH validation |
| `cptac_pda_os` | survival | pancreatic ductal adenocarcinoma | 51-52 cases/fold | 25-26 cases/fold | 131,136 cached tiles; full grid embedded | 3 folds | 139.4s | PathoBench CPTAC-PDA OS / TCIA PathDB | PathoBench fold-0 train only; fold-0 test held out; case-level pooling; train-z-scored fixed-ridge CoxPH |
| `pathorob` | robustness | breast lymph node + esophagus | n/a | n/a | 22,402 + 13,800 used patches | n/a | 74.5s | PathoROB HF datasets | Robustness index over camelyon/tolkach_esca; TCGA subset excluded |

## Files

Split metadata used directly by `probe.py`:

- `bracs.json`
- `break_his.json`
- `mhist.json`
- `ucla_lung.json`
- `surgen.json`
- `leopard_bcr.json`
- `cptac_pda_os.json`

Datasets without JSON here are split directly by code: PCam uses fixed subset H5s, PanNuke uses Fold1/Fold2, MoNuSAC and CoNSeP use deterministic train-folder splits, and PathoROB reads its public parquet subsets directly.

Dataset-specific notes:

- [break_his.md](break_his.md)
- [bracs.md](bracs.md)
- [mhist.md](mhist.md)
- [pcam.md](pcam.md)
- [monusac.md](monusac.md)
- [consep.md](consep.md)
- [pannuke.md](pannuke.md)
- [ucla_lung.md](ucla_lung.md)
- [surgen.md](surgen.md)
- [leopard_bcr.md](leopard_bcr.md)
- [cptac_pda_os.md](cptac_pda_os.md)
- [pathorob.md](pathorob.md)
