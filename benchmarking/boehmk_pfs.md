# BoehmK Survival

## Role In Nanopath

`boehmk_pfs` is an ovarian slide-level survival probe. Upstream PathoBench calls the endpoint PFS, short for progression-free survival. It contributes one scalar to `mean_probe_score`: Harrell's validation c-index.

## Source

- Labels: `MahmoodLab/Patho-Bench`, file `boehmk_/PFS/k=all.tsv`
- Upstream metadata: `task_type: survival`, `metrics: cindex`, with `PFS_event` and `PFS_days`
- Raw WSIs: BOEHMK Synapse project at `https://www.synapse.org/Synapse:syn25946117/wiki/611576`
- Portable setup mirror: `medarc/nanopath`, under `probes/boehmk_pfs/`

## Split And Patches

Nanopath vendors `boehmk_pfs.json`, derived from PathoBench BOEHMK survival/PFS fold_0. PathoBench fold_0 test remains held out; Nanopath uses deterministic 3-fold event-stratified validation over the fold_0 train pool.

| split | cases/slides | event labels | cached patches |
|---|---:|---|---:|
| train pool | 146 | 96 event / 50 censored | 271,467 cached 20x/512 tissue tiles |
| per-fold train | 97-98 | reused | reused |
| per-fold val | 48-49 | reused | reused |
| held-out PathoBench test | 37 | 24 event / 13 censored | not read |

## Implementation

`prepare.py` normally downloads the pre-extracted `medarc/nanopath` parquet cache: `patches.parquet`, `labels.tsv`, and `tiling_version.txt`. `fetch_boehmk_pfs_from_synapse()` is the regeneration helper for rebuilding that mirror after the user has accepted the BOEHMK Synapse access terms. It downloads the Synapse `data.tar.gz`, extracts a deterministic 20x, 512 px, 0-overlap tissue grid, and writes one combined `patches.parquet`.

`probe.py` streams a deterministic raster-spaced sub-bag of up to 768 cached patches per slide with a no-crop square resize, mean-pools patch embeddings by slide, and sweeps `sksurv.linear_model.CoxnetSurvivalAnalysis` over `l1_ratio={0.5,1.0}` and PathoBench's `alpha={0.01,0.02,0.07}` grid without extra fold-wise feature standardization. It reports mean validation Harrell's c-index at the best mean Coxnet setting. Exact `l1_ratio=0.0` is omitted because Coxnet requires a positive `l1_ratio`.

## Difference From Original Usage

PathoBench's BOEHMK survival task reports Harrell's c-index. PathoBench is designed for standardized task evaluation across folds and pools Trident patch embeddings. Nanopath keeps the same 20x/512 patch-grid cache, uses a deterministic up-to-768-tile sub-bag for final-probe runtime, runs a deterministic Coxnet hyperparameter sweep on mean-pooled custom-backbone embeddings, then uses repeated train-derived internal validation for fast iteration and does not score the PathoBench test fold. The tissue mask is a lightweight deterministic thumbnail mask rather than Trident HEST segmentation.

## Runtime

On May 12, 2026 H100 survival-only probes, the deterministic 768-tile sub-bag embedded 85,803 tiles and took 83.7s for DINOv2-S random, 84.0s for DINOv2-S, 148.9s for DINOv2-G, 148.9s for OpenMidnight, 148.5s for H-optimus-0, and 435.8s for GenBio-PathFM. The previous full-grid H100 leader-derived DINOv2-S/Nanopath smoke run took 335.5s for BoehmK survival and 844.3s for the full probe.
