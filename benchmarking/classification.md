# Classification protocol

The classification family follows THUNDER's frozen-feature evaluation on 12
development tasks. It uses official training data to fit heads and official
validation data exactly once for scoring. Official test paths and labels are
not present in the [THUNDER manifest](thunder_v2.json) or the downloadable evaluation
snapshot.

## Development selections

| Task | Tissue / labels | Selected train | Selected validation | Split and sampling contract |
|---|---|---:|---:|---|
| BACH | breast; normal, benign, in-situ, invasive | 218 | 50 | Complete THUNDER train and validation splits |
| BRACS | breast ROI; 7 lesion classes | 512 | 312 | Class-stratified train cap; complete validation; case-like filename prefixes are disjoint |
| BreaKHis | breast; fibroadenoma, tubular adenoma, ductal carcinoma, mucinous carcinoma | 936 | 196 | Complete patient-disjoint 40x THUNDER train and validation splits |
| CRC | colorectal; 9 tissue classes | 4,096 | 2,048 | Class-stratified subsets of NCT-CRC-HE-100K's THUNDER train/validation partition |
| ESCA | esophagus; 11 morphology classes | 4,096 | 2,048 | Class/source-aware cap; training contains WNS and TCGA, validation is UKK-only |
| MHIST | colorectal polyps; hyperplastic polyp and sessile serrated adenoma | 1,743 | 432 | Complete THUNDER train/validation split derived from the original training partition |
| PCam | lymph node; normal and metastatic | 3,072 | 1,024 | Balanced deterministic indices from official train and validation H5 files |
| SPIDER breast | breast; 18 classes | 3,072 | 1,024 | Class/source-slide-aware cap within official train and validation |
| SPIDER colorectal | colorectal; 13 classes | 3,072 | 1,024 | Class/source-slide-aware cap within official train and validation |
| SPIDER skin | skin; 24 classes | 4,096 | 2,048 | Class/source-slide-aware cap within official train and validation |
| SPIDER thorax | thorax; 14 classes | 3,072 | 1,024 | Class/source-slide-aware cap within official train and validation |
| WILDS Camelyon17 | lymph node; normal and tumor | 4,096 | 2,048 | Balanced cap across patient/node groups from official `train` and `val` environments |

All capped selections use seed 1337 and preserve the original split identity.
Training caps are stratified and retain at least 16 examples from every class;
validation retains every class available in the official validation split and
at least one selected example from each. SPIDER source slides and WILDS
patient/node groups are disjoint across selected train and validation data.

The validation lower bound is intentionally one, not 16. SPIDER-Skin's complete
official validation pool contains only one example of its rarest class, so a
larger guarantee would require moving data between official splits or changing
the label distribution. Neither is allowed.

Split membership and task definitions follow the
[THUNDER dataset registry](https://mics-lab.github.io/thunder/). Primary dataset
references are [BACH](https://iciar2018-challenge.grand-challenge.org/Dataset/),
[BRACS](https://arxiv.org/abs/2111.04740),
[BreaKHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/),
[CRC-100K](https://zenodo.org/records/1214456),
[MHIST](https://bmirds.github.io/MHIST/),
[PatchCamelyon](https://zenodo.org/records/2546921),
[SPIDER](https://github.com/HistAI/SPIDER), and
[Camelyon17-WILDS](https://wilds.stanford.edu/datasets/). The manifest is a
path-and-label selection, not a relicensed copy of those datasets; their
original research-use terms still apply.

CRC's external CRC-VAL-HE-7K cohort is THUNDER's test set and is absent. MHIST's
original test partition, SPIDER's separate test folders, WILDS `test`, and all
other official test records are likewise absent. CCRCC, TCGA CRC-MSI,
TCGA-TILs, and TCGA-Uniform are deliberately excluded because their evaluation
images are explicitly TCGA. ESCA remains because its scored validation data is
entirely UKK, despite TCGA being one source in the head-training pool.

## Frozen embeddings

Images use the model's fixed probe transform and normalization. The backbone
runs once per split under fp16 autocast; the resulting `probe_features()` vectors
are stored as float32 and shared by every head. The backbone is never updated.

For dataset `d`, the benchmark computes three equally weighted head scores:

```text
linear_d     = mean macro-F1 over 9 fixed Adam heads
knn_d        = mean macro-F1 over 8 fixed k values
simpleshot_d = macro-F1 of one 1,000-draw majority-vote classifier
dataset_d    = mean(linear_d, knn_d, simpleshot_d)

classification_mean_f1 = mean(dataset_d for all 12 datasets)
```

Thus each dataset has equal family weight and each of the three probe types has
equal weight inside a dataset. The 9 linear cells and 8 KNN cells do not count
as 17 additional top-level tasks.

### Linear heads

- One linear layer per cell in
  `lr={1e-3, 1e-4, 1e-5} × weight_decay={0, 1e-3, 1e-4}`.
- Adam, batch size 64, 200 epochs, seed 0.
- THUNDER's embedding-loader shuffle stream is reproduced.
- Each cell's final-epoch validation macro-F1 is reported; their mean is the
  linear score. Validation is never used for checkpoint or cell selection.

### KNN

- Cosine similarity after L2-normalizing train and validation vectors.
- Majority vote at `k={1, 3, 5, 10, 20, 30, 40, 50}`.
- Every validation macro-F1 is reported; their mean is the KNN score.

### 16-shot SimpleShot

- Seed 0 recreates THUNDER's random support-index stream through 1, 2, 4, 8,
  and 16 shots; only the 16-shot result is scored.
- For each of 1,000 balanced support draws, subtract that support set's global
  mean from support and validation vectors, form normalized class centroids,
  and predict by cosine similarity.
- Majority vote across the 1,000 predictions produces one validation prediction
  per image and one macro-F1.

## Deliberate difference from official THUNDER

Official THUNDER can choose the best linear hyperparameter/epoch and KNN `k` on
visible validation data, then report a separate held-out test score. Nanopath
keeps test data sealed, so it has no independent split on which to report a
validation-selected winner. Choosing the maximum cell and reporting that same
validation cell would reward selection noise. Nanopath instead fixes the
official grids and marginalizes them, using no nested split, refit, or hidden
selection rule. This is simpler, deterministic, and honest about the available
data, while retaining the relative behavior of THUNDER's three probe types.

The tradeoff is that `classification_mean_f1` is not numerically calibrated to
THUNDER's selected-head test aggregate. Its purpose is to preserve comparative
model ordering. Per-head and per-cell metrics remain in the result JSON so a
failure mode is observable rather than hidden by the family mean.
