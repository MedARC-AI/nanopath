# HER2-Tumor-ROIs

## Role In Nanopath

`her2` is a breast slide-level response classification probe. It contributes one scalar to `mean_probe_score`: validation AUROC from a balanced logistic linear probe on mean-pooled slide embeddings.

## Source

- Raw images: HER2-Tumor-ROIs via TCIA PathDB collection id `533`
- Split provenance: PathoBench `herroi/response`, vendored as `her2_pathobench.tsv`
- URL endpoint used by `prepare.py`: `https://pathdb.cancerimagingarchive.net/listofimages/533`

## Split And Tiles

PathoBench fold 0 has 68 train slides and 17 test slides. Nanopath runs deterministic stratified 3-fold validation over the 68 fold-0 train slides with seed 1337. The held-out test slides are not tiled or scored, and case ids are unique.

| split | slides | cached tiles |
|---|---:|---:|
| train pool | 68 | full 20x/512 tissue grid |
| per-fold train | 45-46 | reused |
| per-fold val | 22-23 | reused |

## Implementation

`prepare.py` downloads the PathDB SVS files named by `her2.json`, extracts a full deterministic 20x, 512 px, 0-overlap tissue grid into `root/tiles/<slide_id>/*.jpg`, and keeps raw slides under `root/raw` for resumable setup. `probe.py` embeds every cached tile once with a no-crop square resize, mean-pools tile embeddings per slide, then sweeps a balanced logistic linear probe across the three slide folds and reports AUROC.

## Difference From Original Usage

PathoBench defines external fold metadata for treatment-response evaluation, runs linear probing on mean-pooled Trident patch embeddings, and reports AUROC. Nanopath's balanced logistic linear probe matches that head class on custom-backbone features. Nanopath follows the 20x/512 uncapped patch-grid shape, derives repeated train-fold validation from fold metadata, and does not evaluate the held-out test fold in `mean_probe_score`. The tissue mask is a lightweight deterministic thumbnail mask rather than Trident HEST segmentation.

## Runtime

HER2 must be remeasured after the 20x/512 retile and no-crop slide transform.

| model | wall |
|---|---:|
| DINOv2-S | remeasure |
| OpenMidnight | remeasure |
| H-optimus-0 | remeasure |
