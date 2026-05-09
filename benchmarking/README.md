# Benchmarking
This folder contains code specific for probing/downstream evaluation. The normal nanopath loop is to train within 45 minutes on one H100 gpu, freeze the model backbone, run a broad downstream probe suite, and use that result to decide whether a training idea is worth scaling. It is a validation benchmark for rapid iteration. For the most part we borrow the same approach used by THUNDER / PathoBench for downstream evaluations, with a notable exception that we entirely hold-out all test split data from this codebase (this means we can still evaluate our finished models on THUNDER & PathoBench official benchmarking without as much risk of overfitting).

## Metric

`mean_probe_score` is the unweighted mean of one scalar per dataset and is the single score we use to assess relative performance of trained nanopath models.

```text
mean_probe_score = mean(
  break_his, bracs, mhist, pcam,
  monusac, consep, pannuke,
  her2, ucla_lung,
  surgen,
  crc_survival,
  pathorob
)
```

## Probe Families

| family | datasets | dataset scalar |
|---|---|---|
| Tile classification | `break_his`, `bracs`, `mhist`, `pcam` | mean of linear, KNN, and SimpleShot macro F1 |
| Slide classification | `her2`, `ucla_lung` | AUROC from a balanced logistic linear probe on mean-pooled slide embeddings |
| Segmentation | `monusac`, `consep`, `pannuke` | macro Jaccard from a small MaskTransformer head on frozen patch tokens |
| Mutation prediction | `surgen` | AUROC for PathoBench SR386 RAS mutation status |
| Survival | `crc_survival` | Harrell's c-index from a Cox survival head with validation-selected `l1_ratio` and `alpha` |
| Robustness | `pathorob` | mean PathoROB-style robustness index across camelyon and tolkach_esca |

All probes keep the backbone frozen. Probe heads are intentionally small: they measure representation quality, not downstream fine-tuning capacity.

Linear, KNN, SimpleShot, segmentation-head, logistic, and Coxnet hyperparameters are selected on the same internal validation splits that define `mean_probe_score`. Thunder-derived tile classifiers keep Thunder-style linear/KNN/SimpleShot heads; PathoBench-derived slide classifiers use logistic linear probing rather than KNN or SimpleShot. Tiny train-derived probes use deterministic 3-fold validation over their official-train pool (`monusac`, `consep`, `her2`, `ucla_lung`, `surgen`, `crc_survival`) while reusing frozen embeddings/features. `probe.py` logs fold variance/std for those repeated probes so noisy improvements are easier to spot. This is deliberate: the suite is a fast validation probe for model development, while official test labels stay sealed.

## Runtime Strategy

The full suite is designed for the final H100 probe window for the standard small Nanopath model by keeping the benchmark small where it can be small, and precomputing expensive slide tiling during `prepare.py download=True`. Giant frozen baselines are timing stress tests and can run beyond the small-model probe budget.

- Whole-slide tasks use pre-extracted tile grids or cached tile directories, so the final probe embeds JPEG/parquet tiles rather than opening full WSIs. PathoBench-derived slide tasks use an uncapped 20x, 512 px, 0-overlap tissue grid following the Trident/PathoBench tutorial contract; Nanopath does not impose a fixed tiles-per-slide cap. SurGen uses the pre-extracted `medarc/nanopath` parquet mirror by default because official CZI download + tiling is multi-hour; `prepare.py` keeps the official-source regeneration helper for rebuilding that mirror. The remaining preprocessing simplification is a deterministic thumbnail tissue mask instead of invoking Trident's HEST segmentation model during `prepare.py`.
- PCam is a fixed subset of the official train/valid H5 files.
- Segmentation runs in a background thread while classification, slide, survival, and robustness probes run in the main worker. CUDA kernels still serialize, but CPU-heavy decode/head work overlaps with segmentation head training.
- The same loaded frozen backbone serves every probe in one subprocess, avoiding repeated model load overhead.
- Test splits are not read by `probe.py`, which keeps the benchmark iterative and avoids consuming official test labels during model development.

Recent H100 timings from the untouched baselines after the uncapped PathoBench-style retile and no-crop patch transform. Wall time varies with concurrent jobs and OS page cache.

| dataset | DINOv2-S | OpenMidnight | H-optimus-0 |
|---|---:|---:|---:|
| `bracs` | 165.9s | 160.1s | 169.5s |
| `break_his` | 18.8s | 19.7s | 19.4s |
| `mhist` | 13.1s | 28.4s | 26.9s |
| `pcam` | 25.8s | 52.3s | 51.6s |
| `pannuke` | 160.1s | 296.8s | 284.2s |
| `monusac` | 23.8s | 86.9s | 87.2s |
| `consep` | 5.0s | 16.9s | 17.2s |
| `her2` | 64.2s | 81.6s | 80.4s |
| `ucla_lung` | 26.2s | 64.2s | 60.2s |
| `surgen` | 1267.6s | 2239.0s | 2201.9s |
| `crc_survival` | 191.6s | 274.9s | 275.5s |
| `pathorob` | 69.9s | 101.4s | 99.4s |
| **probe wall** | **1844.1s** | **3033.4s** | **2996.0s** |

The dominant bottleneck is SurGen slide embedding from the uncapped 1.17M-tile cache. PanNuke segmentation, BRACS fitting, CRC survival, PathoROB, HER2, and UCLA Lung are secondary; several of these overlap, but SurGen, CRC survival, and PathoROB sit on the main sequential path.

## Dataset Summary

| dataset | task | tissue / organ | train units | val units | train tiles/images | val tiles/images | H-optimus time | source | Nanopath adaptation |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `break_his` | tile classification (~700×460 microscope captures) | breast | 936 images | 196 images | 936 | 196 | 19.4s | BreaKHis 40X / EVA-Thunder | Patient-disjoint 4-subtype 40X split; no test scoring |
| `bracs` | ROI classification (variable-size WSI crops) | breast | 3657 ROIs | 312 ROIs | 3657 | 312 | 169.5s | BRACS ROI FTP | Frozen-embedding linear/KNN/few-shot; no official test scoring |
| `mhist` | tile classification (224 px patches) | colorectal polyps | 1743 images | 432 images | 1743 | 432 | 26.9s | MHIST | Official train partition split internally; official test not read |
| `pcam` | tile classification (96 px patches) | lymph node metastasis | 3072 images | 768 images | 3072 | 768 | 51.6s | PCam Zenodo | Fixed train/valid subset; official test not read |
| `monusac` | segmentation | multi-organ nuclei | ~31 slides/fold | ~15 slides/fold | 209 total images | 3 folds | 87.2s | MoNuSAC train set | Deterministic 3-fold slide split of train package; no test data |
| `consep` | segmentation | colorectal nuclei | 18 ROIs/fold | 9 ROIs/fold | 27 total images | 3 folds | 17.2s | CoNSeP | Deterministic 3-fold split of official Train folder; Test folder not read |
| `pannuke` | segmentation | multi-organ nuclei | Fold1 | Fold2 | 2656 images | 2523 images | 284.2s | PanNuke folds | Fixed Fold1/Fold2 protocol; Fold3 not scored |
| `her2` | slide response classification | breast | ~45 slides/fold | ~23 slides/fold | full 20x/512 grid | 3 folds | 80.4s | PathoBench `herroi/response` / HER2-Tumor-ROIs | 3-fold balanced logistic AUROC over fold-0 train using all cached tiles; 17-slide test fold held out |
| `ucla_lung` | slide progression classification | lung | 60 slides/fold | 30 slides/fold | full 20x/512 grid | 3 folds | 60.2s | PathoBench `ucla_lung/progression_regression` / IDR idr0082 | 3-fold balanced logistic AUROC over fold-0 train using the full tissue grid; 22-slide test fold held out |
| `surgen` | mutation classification | colorectal | ~207 slides/fold | ~104 slides/fold | 1,167,089 cached tiles | 3 folds | 2201.9s | PathoBench SR386 / SurGen, mirrored as pre-extracted HF parquet | 3-fold validation over PathoBench fold-0 train; fold-0 test sealed |
| `crc_survival` | survival | colorectal | ~91 slides/fold | ~45 slides/fold | full 20x/512 grid | 3 folds | 275.5s | PathoBench PFS_VALENTINO / BioStudies | 3-fold Coxnet `l1_ratio={0.5,1.0}`, `alpha={0.03,0.1}` validation over fold-0 train; PathoBench test held out |
| `pathorob` | robustness | breast lymph node + esophagus | n/a | n/a | 22402 + 16300 patches | n/a | 99.4s | PathoROB HF datasets | Robustness index over camelyon/tolkach_esca; TCGA subset excluded |

## Files

Split metadata used directly by `probe.py`:

- `bracs.json`
- `break_his.json`
- `mhist.json`
- `her2.json`
- `ucla_lung.json`
- `surgen.json`
- `crc_survival.json`
- `her2_pathobench.tsv`

Datasets without JSON here are split directly by code: PCam uses fixed subsets of official H5 train/valid files, PanNuke uses Fold1/Fold2, MoNuSAC and CoNSeP use deterministic train-folder splits, and PathoROB reads its public parquet subsets directly.

Dataset-specific notes:

- [break_his.md](break_his.md)
- [bracs.md](bracs.md)
- [mhist.md](mhist.md)
- [pcam.md](pcam.md)
- [monusac.md](monusac.md)
- [consep.md](consep.md)
- [pannuke.md](pannuke.md)
- [her2.md](her2.md)
- [ucla_lung.md](ucla_lung.md)
- [surgen.md](surgen.md)
- [crc_survival.md](crc_survival.md)
- [pathorob.md](pathorob.md)
