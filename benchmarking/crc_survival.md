# CRC Survival

## Role In Nanopath

`crc_survival` is a colorectal slide-level survival probe. It contributes one scalar to `mean_probe_score`: Harrell's validation c-index.

## Source

- Labels: `MahmoodLab/Patho-Bench`, file `crc_outcomes/PFS_VALENTINO/k=all.tsv`
- Raw WSIs: EBI BioStudies VALENTINO files at `https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/407/S-BIAD1407/Files/VALENTINO`

## Split And Patches

Nanopath vendors `crc_survival.json`, derived from PathoBench PFS_VALENTINO fold_0 train slides. PathoBench fold_0 test remains held out; Nanopath uses deterministic 3-fold event-stratified validation over the fold-0 train slides.

| split | slides | event labels | cached patches |
|---|---:|---|---:|
| train pool | 136 | 123 event / 13 censored | full 20x/512 tissue grid |
| per-fold train | 90-91 | reused | reused |
| per-fold val | 45-46 | reused | reused |

## Implementation

`prepare.py` downloads the PathoBench label TSV, downloads fold-0 train VALENTINO TIFFs, and extracts a full deterministic 20x, 512 px, 0-overlap tissue grid into `patches.parquet`. `probe.py` embeds each patch once with a no-crop square resize, mean-pools patch embeddings by slide, standardizes features within each training fold, sweeps `sksurv.linear_model.CoxnetSurvivalAnalysis` over `l1_ratio={0.5,1.0}` and `alpha={0.03,0.1}`, and reports mean validation Harrell's c-index at the best mean Coxnet setting. Exact `l1_ratio=0.0` is omitted because Coxnet requires a positive `l1_ratio`; lower-alpha substitutes were slow and numerically fragile for 1536-dimensional giant-model embeddings. This mirrors the classifier probes' validation-selected hyperparameter sweeps while keeping one Cox implementation.

## Difference From Original Usage

PathoBench's survival tasks report Harrell's c-index, so Nanopath's Coxnet-sweep c-index matches the metric definition. PathoBench is designed for standardized task evaluation across folds and pools Trident patch embeddings without a fixed patch-count cap. Nanopath follows the uncapped 20x/512 patch-grid shape, runs a deterministic Coxnet hyperparameter sweep on mean-pooled custom-backbone embeddings, then uses repeated train-derived internal validation for fast iteration and does not score the PathoBench test fold. The tissue mask is a lightweight deterministic thumbnail mask rather than Trident HEST segmentation.

## Runtime

| model | wall |
|---|---:|
| DINOv2-S | 191.6s |
| OpenMidnight | 274.9s |
| H-optimus-0 | 275.5s |
