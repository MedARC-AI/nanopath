# Benchmarking
This folder contains code specific for probing/downstream evaluation. The normal nanopath loop is to train within 45 minutes on one H100 gpu, freeze the model backbone, run the fixed downstream probe suite, and use that result to decide whether a training idea is worth scaling. The benchmark definition stays fixed for fair model comparisons. For the most part we borrow the same approach used by THUNDER / PathoBench for downstream evaluations, with a notable exception that we entirely hold-out all test split data from this codebase (this means we can still evaluate our finished models on THUNDER & PathoBench official benchmarking without as much risk of overfitting).

## Metric

`mean_probe_score` is the unweighted mean of one scalar per dataset and is the single score we use to assess relative performance of trained nanopath models.

```text
mean_probe_score = mean(
  break_his, bracs, mhist, pcam,
  monusac, consep, pannuke,
  ucla_lung,
  surgen,
  boehmk_pfs,
  pathorob
)
```

## Probe Families

| family | datasets | dataset scalar |
|---|---|---|
| Tile classification | `break_his`, `bracs`, `mhist`, `pcam` | mean of linear, KNN, and 16-shot SimpleShot macro F1|
| Segmentation | `monusac`, `consep`, `pannuke` | macro Jaccard from a small MaskTransformer head on frozen patch tokens |
| Progression | `ucla_lung` | AUROC from a balanced logistic linear probe on mean-pooled slide embeddings |
| Mutation | `surgen` | AUROC for PathoBench SR386 RAS mutation status |
| Survival | `boehmk_pfs` | Harrell's c-index from a Cox time-to-event head with validation-selected `l1_ratio` and `alpha` |
| Robustness | `pathorob` | mean PathoROB-style robustness index across camelyon and tolkach_esca |

All probes keep the backbone frozen. Probe heads are intentionally small: they measure representation quality, not downstream fine-tuning capacity.

Probe heads consume each model's native frozen feature dimension rather than projecting every backbone to a common width. This intentionally evaluates each checkpoint as a deployed feature extractor, but it also means dimensionality is part of the baseline comparison: DINOv2-S emits 384-d features, DINOv2-G/OpenMidnight/H-optimus-0/UNI-2-h emit 1536-d features, Midnight-12K emits 3072-d CLS plus mean-patch features, and GenBio-PathFM emits 4608-d features.

Linear, KNN, segmentation-head, logistic, and Coxnet hyperparameters are selected on the same internal validation splits that define `mean_probe_score`. Thunder-derived tile classifiers keep Thunder-style linear/KNN/16-shot SimpleShot heads; SimpleShot precomputes 1000 deterministic support sets and majority-votes query predictions. PathoBench-derived slide classifiers use balanced logistic linear probing; SurGen uses sklearn's `liblinear` solver. Tiny train-derived probes use deterministic 3-fold validation over their official-train pool (`monusac`, `consep`, `ucla_lung`, `surgen`, `boehmk_pfs`) while reusing frozen embeddings/features. `probe.py` logs fold variance/std for those repeated probes so noisy improvements are easier to spot.

## Runtime Strategy

The full suite is designed for the final H100 probe window for the standard small Nanopath model by keeping the benchmark small where it can be small, and precomputing expensive slide tiling.

- Whole-slide tasks use pre-extracted tile grids, so the final probe embeds JPEG/parquet tiles rather than opening full WSIs. PathoBench-derived slide tasks use a 20x, 512 px, 0-overlap tissue grid following the Trident/PathoBench tutorial contract. UCLA Lung embeds the full cached grid; SurGen and BoehmK survival prepare the full grid but stream deterministic up-to-768-tile raster-spaced sub-bags per slide so large-slide tasks fit the final-probe window. UCLA Lung, SurGen, and BoehmK survival use pre-extracted `medarc/nanopath` parquet mirrors by default; `prepare.py` keeps official-source regeneration helpers for rebuilding the SurGen and BoehmK mirrors. The remaining preprocessing simplification is a deterministic thumbnail tissue mask instead of invoking Trident's HEST segmentation model during `prepare.py`.
- PCam is a fixed subset of the official train/valid H5 files, mirrored as small H5s with the same filenames/schema.
- Tile classifiers use `Resize((224, 224))` from `model.py::probe_transforms` for trained Nanopath checkpoints. Frozen baseline scripts set their own `probe.transform_policy`. Patch-cache probes keep square resize because their inputs are already extracted tissue tiles.
- Segmentation runs in a background thread while classification, slide, survival, and robustness probes run in the main worker for DINOv2-style backbones. CUDA kernels still serialize, but CPU-heavy decode/head work overlaps with segmentation head training.
- The same loaded frozen backbone serves every probe in one subprocess, avoiding repeated model load overhead.
- Test splits are not read by `probe.py`, which keeps official labels sealed during model development.

Recent H100 timings from the untouched baselines:

| dataset | DINOv2-random | DINOv2-S | DINOv2-G | UNI-2-h | Midnight-12K | OpenMidnight | H-optimus-0 | GenBio-PathFM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bracs` | 160.3s | 182.8s | 168.5s | 168.5s | 193.2s | 163.8s | 179.9s | 160.6s |
| `break_his` | 26.4s | 15.1s | 18.7s | 18.3s | 20.3s | 19.0s | 20.4s | 14.6s |
| `mhist` | 10.9s | 12.3s | 24.7s | 23.9s | 27.0s | 27.6s | 26.3s | 21.2s |
| `pcam` | 23.9s | 27.8s | 45.7s | 42.5s | 57.2s | 58.5s | 50.0s | 43.1s |
| `pannuke` | 159.2s | 165.3s | 303.0s | 289.9s | 481.1s | 732.9s | 309.5s | 279.8s |
| `monusac` | 22.7s | 24.9s | 139.5s | 126.3s | 150.9s | 64.0s | 91.5s | 34.5s |
| `consep` | 4.4s | 5.1s | 41.6s | 33.7s | 32.4s | 5.0s | 18.5s | 5.9s |
| `ucla_lung` | 27.2s | 31.7s | 63.4s | 44.4s | 65.4s | 68.7s | 63.6s | 140.3s |
| `surgen` | 205.0s | 234.5s | 414.5s | 252.0s | 416.5s | 467.3s | 386.8s | 1137.0s |
| `boehmk_pfs` | 83.7s | 84.0s | 148.9s | 92.7s | 148.5s | 148.9s | 148.5s | 435.8s |
| `pathorob` | 34.5s | 28.3s | 72.3s | 43.0s | 67.1s | 75.6s | 74.5s | 198.4s |

GenBio-PathFM is a slow outlier because each RGB tile is encoded as three single-channel ViT-G passes and the heads consume native 4608-d features. GenBio-PathFM baseline also runs segmentation sequentially because its three channel-wise ViT-G passes are already GPU-bound, and background PanNuke contention made the baseline much slower without changing the metric.

## Dataset Summary

| dataset | task | tissue / organ | train units | val units | train tiles/images | val tiles/images | reference H100 time | source | Nanopath adaptation |
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
| `boehmk_pfs` | survival | ovarian | ~97 slides/fold | ~49 slides/fold | 271,467 cached tiles; up to 768 embedded/slide | 3 folds | 84.0s | PathoBench BOEHMK PFS / Synapse, mirrored as pre-extracted HF parquet | 3-fold Coxnet `l1_ratio={0.5,1.0}`, `alpha={0.01,0.02,0.07}` validation over fold_0 train; PathoBench test held out |
| `pathorob` | robustness | breast lymph node + esophagus | n/a | n/a | 22402 + 16300 patches | n/a | 74.5s | PathoROB HF datasets | Robustness index over camelyon/tolkach_esca; TCGA subset excluded |

## Files

Split metadata used directly by `probe.py`:

- `bracs.json`
- `break_his.json`
- `mhist.json`
- `ucla_lung.json`
- `surgen.json`
- `boehmk_pfs.json`

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
- [boehmk_pfs.md](boehmk_pfs.md)
- [pathorob.md](pathorob.md)
