# Downstream Probe Benchmark

Nanopath uses a 12-dataset downstream benchmark. The goal is to make `mean_probe_score` generalizable across tile classification, segmentation, slide-level classification, mutation prediction, survival, and robustness while still finishing the full probe on one H100 in the final training window of 15 minutes.

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

Each dataset contributes exactly once, with no weighting by dataset size, number of tiles, number of heads, or wall time. `probe.py` asserts that the configured probe suite is exactly this set before running.

## Probe Contract

All probes keep the backbone frozen. The probe worker loads the requested checkpoint, embeds train/validation examples with the frozen encoder, and trains only small downstream heads or nonparametric probes. No probe updates the foundation model.

Tile and slide classification datasets use three heads on cached embeddings:

- Linear probe: an AdamW-trained linear classifier, LR sweep over `1e-3`, `1e-4`, and `1e-5`, scored by validation macro F1.
- KNN: cosine KNN on frozen embeddings, best `k` selected on validation macro F1.
- Few-shot: SimpleShot-style class prototypes for 1, 2, 4, 8, and 16 shots with 1000 trials; the dataset score uses the mean macro F1 over shot counts.

For these datasets, the dataset scalar is:

```text
dataset_score = mean(linear_val_f1, knn_val_f1, fewshot_val_f1)
```

Segmentation datasets train a small MaskTransformer decoder on frozen patch tokens for 30 epochs. The head is selected by validation dice loss and scored by validation macro Jaccard using the Thunder-compatible per-image weighting implemented in `probe.py`.

SurGen trains a logistic regression head on mean-pooled slide embeddings and reports validation AUROC for extended-RAS status. CRC survival trains a Coxnet survival head on mean-pooled slide embeddings and reports Harrell's validation c-index. PathoROB reports the mean of its camelyon and tolkach_esca neighbor-based robustness indices.

## Dataset Details

| dataset | task | probe scalar | split used for score | standardization notes |
|---|---|---:|---|---|
| `break_his` | tile classification | mean of linear/KNN/few-shot macro F1 | checked-in `train`/`val` JSON | Thunder-style deterministic split; `test` JSON exists for provenance but is not read |
| `bracs` | ROI classification | mean of linear/KNN/few-shot macro F1 | checked-in `train`/`val` JSON | Thunder-style deterministic split; `test` JSON exists for provenance but is not read |
| `mhist` | tile classification | mean of linear/KNN/few-shot macro F1 | checked-in `train`/`val` JSON | Thunder-style deterministic split; HF mirror is for portable setup after users satisfy MHIST access terms; `test` JSON is not read |
| `pcam` | tile classification | mean of linear/KNN/few-shot macro F1 | official `train` and `valid` H5 files | fixed subset: 3072 train and 768 valid examples; downloaded `test` H5 is not read |
| `monusac` | nucleus segmentation | macro Jaccard | deterministic 80/20 split of official train slides | slide-disjoint validation from the train package; no official test data is used |
| `consep` | nucleus segmentation | macro Jaccard | deterministic 80/20 split of official `Train` images | validation comes only from `Train`; archive may contain `Test`, but `probe.py` does not read it |
| `pannuke` | nucleus segmentation | macro Jaccard | Fold1 train, Fold2 validation | fixed fold protocol for speed/reproducibility; Fold3 may be downloaded but is not scored |
| `her2` | slide-level response classification | mean of linear/KNN/few-shot macro F1 | PathoBench-derived train/validation slides | derived from PathoBench fold_0 train slides; fold test is held out |
| `ucla_lung` | slide-level progression/regression classification | mean of linear/KNN/few-shot macro F1 | checked-in `train`/`val` slide JSON | deterministic slide split; JSON `test` split is provenance only and is not read |
| `surgen` | slide-level mutation classification | validation AUROC | checked-in `train`/`val` slide JSON | extended-RAS split; JSON `test` split is not read |
| `crc_survival` | slide-level survival | validation c-index | PathoBench fold_0 train-derived train/validation slides | PathoBench fold_0 test remains held out |
| `pathorob` | robustness | mean robustness index | PathoROB camelyon and tolkach_esca parquet subsets | follows the PathoROB neighbor-index idea with published per-subset `k` values; TCGA subset is excluded |

## Test Split Policy

The leaderboard does not evaluate on standardized test splits.

This is deliberate. Nanopath is an iterative research harness, so public leaderboard iteration should not consume official test labels. The current benchmark is a fixed validation benchmark: enough to compare ideas quickly and consistently, but not a final locked external evaluation.

Concretely:

- `probe.py` calls classification and slide datasets only with `"train"` and `"val"`.
- BRACS, BreaKHis, MHIST, UCLA Lung, and SurGen JSON files may include `test`, but the probe worker never reads those keys.
- PCam's downloader fetches the official test H5 for dataset completeness, but `ClassificationDataset` maps evaluation to the official `valid` split only.
- PanNuke reads Fold1 for training and Fold2 for validation. Fold3 is not part of `mean_probe_score`.
- MoNuSAC and CoNSeP validation sets are deterministic splits from official train data.
- HER2 and CRC survival are derived from PathoBench train folds; their PathoBench test folds remain held out.
- PathoROB is not a supervised train/test evaluation in this repo; it is a frozen-embedding robustness index over the selected public subsets.

If Nanopath later needs a final publication-grade number, it should add a separate locked evaluation path that is not used during routine development and is not mixed into the iterative `mean_probe_score`.

## What Is Standardized

The current probe is standardized in the practical sense needed for this repo:

- Fixed dataset list in `MEAN_PROBE_DATASETS`.
- Fixed train/validation splits or deterministic split seeds.
- Fixed image preprocessing: resize to 256, center crop to 224 for classification-style probes; segmentation uses the 256x256 label grid with frozen 224x224 center patch-token extraction.
- Fixed probe heads, optimizer choices, LR sweep, KNN `k` candidates, few-shot shot counts, and random seeds.
- Fixed aggregation: one scalar per dataset, unweighted mean across the 12 datasets.
- Fixed output keys in `metrics.jsonl`, including `probe_<dataset>_score` for every dataset.

It is not claiming that every dataset is evaluated with its canonical external-test protocol. Some datasets have well-known official or published splits, while others are adapted into deterministic validation probes to fit the Nanopath loop. The important rule is that every model is measured under the same frozen-backbone validation protocol, and no official test split contributes to the leaderboard score.

## Data Setup

`prepare.py <config> download=True` is the supported setup path. It downloads the TCGA tile parquet shards, verifies every configured probe root, and fetches or prepares any missing probe dataset. Most probes come from official public sources. MHIST and CoNSeP use the `medarc/nanopath` HF probe mirror for noninteractive cluster setup, and `prepare.py` prints notices that users must satisfy the upstream access terms before using those mirrored files.

Several slide-level probes are pre-extracted during setup so the final H100 probe does not spend time reading whole-slide images:

- UCLA Lung: deterministic bags of 64 tiles per train/validation slide.
- HER2: tissue tiles from the PathoBench-derived train/validation slides.
- SurGen: up to 256 tissue tiles per train/validation slide.
- CRC survival: up to 256 tissue patches per train/validation slide.

This keeps the final probe deployable across machines while preserving a clear split boundary: setup may cache raw or derived files, but `mean_probe_score` still reads only the train/validation data described above.

## Runtime

The current 12-probe suite has been smoke-tested as a final probe on one H100. Untouched DINOv2-S ran in about 8.3 minutes, and the larger OpenMidnight / H-optimus-0 ViT-G baselines each completed just under 15 minutes. Segmentation runs in a background thread while the main worker handles classification, slide, survival, and robustness probes, so CPU-heavy embedding/head work can overlap with segmentation head training on the same loaded frozen backbone.

## Why This Is Better

The expanded benchmark is less likely to reward a model that only improves one easy corner of pathology representation learning. It now checks:

- tile-level morphology classification,
- ROI-level classification,
- nucleus segmentation,
- slide-level response/progression classification,
- mutation prediction,
- survival modeling,
- robustness to biological/site confounding.

The tradeoff is that the score is no longer comparable to old six-probe results. Old leaderboard rows should be re-run before being compared to any current `mean_probe_score`.
