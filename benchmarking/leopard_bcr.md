# LEOPARD BCR

## Role In Nanopath

`leopard_bcr` is a prostatectomy slide-level survival probe for biochemical recurrence after radical prostatectomy. It contributes Harrell's validation c-index as one of the two datasets averaged into the README survival column.

## Source

- Dataset: LEOPARD Grand Challenge public training set at `https://leopard.grand-challenge.org/`
- Labels: public S3 object `https://leopard-challenge.s3.us-west-2.amazonaws.com/training`
- Raw WSIs: public S3 prefix `https://leopard-challenge.s3.us-west-2.amazonaws.com/training/`
- Endpoint: `event` and `follow_up_years`, converted to `BCR_event` and `BCR_days`
- Local cluster cache: `/data/leopard_bcr`

## Split And Patches

Nanopath vendors `leopard_bcr.json`, derived from the public 508-case LEOPARD training CSV. This is a balanced long-follow-up subset, not the full official LEOPARD challenge cohort: the probe keeps all 87 recurrence-event cases and pairs them with the 87 censored cases with longest follow-up. No recurrence event is dropped. The 334 unused public-training cases are all shorter-follow-up censored cases, which are less useful as negative controls in a compact survival probe because many may simply not have been followed long enough to recur. Deterministic 3-fold event-stratified validation uses seed 1337.

| split | cases/slides | event labels | cached patches |
|---|---:|---|---:|
| probe pool | 174 | 87 event / 87 long-follow-up censored | 133,632 cached 20x/512 tissue tiles |
| per-fold train | 116 | 58 event / 58 censored | reused |
| per-fold val | 58 | 29 event / 29 censored | reused |
| unused public training cases | 334 | 0 event / 334 shorter-follow-up censored | not read |

## Implementation

`prepare.py download=True` downloads `patches.parquet`, `labels.tsv`, and `tiling_version.txt` from the `medarc/nanopath` probe mirror. The mirrored cache was built from official S3 TIFFs by extracting a deterministic 20x, 512 px, 0-overlap tissue grid using the shared lightweight thumbnail tissue mask, then keeping a raster-spaced sub-bag of 768 tiles per slide. The selected raw slides total about 751 GB of transient S3 transfer during maintainer-side cache generation; normal users download only the final parquet cache.

`probe.py` streams the cached patches with a no-crop square resize, mean-pools patch embeddings by slide/case, z-scores features with train-fold statistics, and fits `sksurv.linear_model.CoxPHSurvivalAnalysis(alpha=2.0)` on the full standardized feature matrix. It reports mean validation Harrell's c-index across the three folds.

## Null Distribution Audit

The cheap label-independent controls over the selected 174-case subset are near null: case number best-direction c-index 0.5147, slide file size best-direction c-index 0.5321, and mask file size best-direction c-index 0.5006. The exact cached-tile randomized-DINOv2 audit is not clean: 20 randomized DINOv2-small seeds scored mean c-index 0.6333, std 0.0116, min 0.6062, max 0.6593.

Reference frozen probes under the same head scored: DINOv2-small 0.6206, DINOv2-giant 0.6774, GigaPath 0.6597, H-optimus-0 0.6636, GenBio-PathFM 0.7016, and the current main DINOv2 KDE checkpoint 0.6784. This means LEOPARD BCR adds clinically relevant recurrence survival coverage, but in the current 174-case subset and full-dimensional CoxPH probe it has substantial random-feature sensitivity. Treat it as an intentionally easier, signal-bearing survival probe paired with the harder CPTAC-PDA OS probe, not as a claim to reproduce the official full-cohort LEOPARD leaderboard.
