# NanoPath probe protocol v2

NanoPath v2 is a fast development-set proxy for held-out THUNDER, HEST, and
CPTAC evaluation. It never runs HEST, adds no CPTAC task beyond the existing
CPTAC-PDA survival probe, and contains no official test records.

## Final score

All values remain on their original 0–1 scales:

```text
classification = mean(dataset × {linear, KNN, 16-shot SimpleShot} macro-F1)
predictive_mean = mean(classification, segmentation, progression, mutation, survival)
robustness_quality = mean((PathoROB robustness index + biological balanced accuracy) / 2)
mean_probe_score = 0.95 * predictive_mean + 0.05 * robustness_quality
```

Each predictive family contributes 19% and robustness contributes 5%.
`mean_probe_score` and the summary alias `final_probe_score` are the only v2
scalar; v1 is not computed.

## Fixed suite

| Family | Datasets | Scored metric |
|---|---|---|
| Classification | BACH, BRACS, BreaKHis, CRC, ESCA, MHIST, PCam, SPIDER breast/colorectal/skin/thorax, WILDS | macro-F1 from linear, KNN, and 16-shot SimpleShot |
| Segmentation | SegPath epithelial and lymphocytes | THUNDER per-image weighted macro-F1 |
| Progression | UCLA Lung | macro-OVR AUC |
| Mutation | SurGen RAS | macro-OVR AUC |
| Survival | LEOPARD BCR and CPTAC-PDA OS | c-index |
| Robustness | PathoROB Camelyon and Tolkach ESCA | quality-adjusted robustness |

The complete suite must finish within 25 minutes on one H100.

## THUNDER development subsets

[thunder_v2.json](thunder_v2.json) is the only classification/segmentation
manifest read at runtime. Every task has exactly `root`, `train`, and `val`; no
test key or path is present.

| Classification task | Train | Validation |
|---|---:|---:|
| BACH | 218 | 50 |
| BRACS | 512 | 312 |
| BreaKHis | 936 | 196 |
| CRC | 4,096 | 2,048 |
| ESCA | 4,096 | 2,048 |
| MHIST | 1,743 | 432 |
| PCam | 3,072 | 1,024 |
| SPIDER breast | 3,072 | 1,024 |
| SPIDER colorectal | 3,072 | 1,024 |
| SPIDER skin | 4,096 | 2,048 |
| SPIDER thorax | 3,072 | 1,024 |
| WILDS | 4,096 | 2,048 |

Selections use seed 1337. Uncapped splits remain complete. Capped splits are
class-stratified with at least 16 examples per available class; SPIDER samples
are spread across source slides, WILDS across patient/node groups, and ESCA
across sources within each class. Validation keeps the official split identity.
The selected ESCA validation samples are UKK rather than TCGA.

CCRCC, TCGA CRC-MSI, TCGA-TILs, and TCGA-Uniform are excluded because their
evaluation images are explicitly TCGA. OCELOT is excluded for the same reason.
PanNuke is also excluded: the released arrays mix TCGA and local-hospital data
without provenance that lets NanoPath construct a verifiably non-TCGA
validation split. MoNuSAC is not a substitute because its released cohort is
TCGA-derived.

SegPath uses source-disjoint official development splits: 32,768/4,518
epithelial crops and 20,906/2,164 lymphocyte crops. The two-task proxy was kept
because the previous 12-model study preserved 57 of 66 pairwise orderings of
the full official THUNDER segmentation aggregate (86.4% concordance; Spearman
0.902). Adding PanNuke preserved 58 of 66, too small a gain to justify the
unverifiable TCGA mixture. These numbers motivate the panel; the revised
end-to-end protocol must be rerun before its leaderboard values are promoted.

## Probe protocols

Classification embeddings are frozen fp16 features. Nine Adam linear heads use
THUNDER's `lr={1e-3,1e-4,1e-5}` ×
`weight_decay={0,1e-3,1e-4}`, batch 64, and 200 epochs. Their nine final
validation macro-F1 values are averaged. KNN likewise averages the fixed
`k={1,3,5,10,20,30,40,50}` cells. This marginalization avoids selecting a
hyperparameter on the same validation samples used for the score. SimpleShot
matches THUNDER's seed-0 stream of 1,000 balanced 16-shot support draws and
majority vote. Every dataset × head cell has equal weight within the single
classification family.

Segmentation retains THUNDER's two-layer MaskTransformer, Dice objective, Adam,
batch 64, and weighted per-image macro-F1/Jaccard calculation. It uses fixed
schedules rather than validation checkpoint selection: epithelial trains 9
epochs at `lr=1e-4`, `weight_decay=1e-3`; lymphocytes trains 21 epochs at
`lr=1e-3`, `weight_decay=1e-4`. Dense tokens are cached as signed int8 vectors
with fp16 scales. All feature extraction is microbatched at 512 images so models
may aggregate multiple intermediate layers or test-time views without excessive
peak memory. For DINO-family encoders, expanded spatial grids are area-pooled to
the native patch grid before the shared decoder, while all model-defined
concatenated layer channels are retained. This leaves ordinary dense outputs
unchanged and preserves test-time depth aggregation without multiplying the
decoder's quadratic spatial cost. The decoder width is 192 to meet the runtime
budget. Only the small decoder uses PyTorch's deterministic math attention
kernel; frozen encoders keep their model-defined fast attention path.

UCLA progression and SurGen mutation use raw pooled features and fixed
`LogisticRegression(C=0.5, class_weight="balanced", random_state=0)` in three
development folds. SurGen reads at most 768 source-spaced tiles per slide.
Survival standardizes within each training fold and fits CoxNet with
`l1_ratio=0.5` at 0.1, 0.2, and 0.7 times that fold's `alpha_max`; every
dataset × alpha × fold c-index is averaged and numerical failures are errors.
LEOPARD reads at most 768 tiles per slide and CPTAC-PDA retains every prepared
tile.

PathoROB uses only Camelyon and non-TCGA Tolkach ESCA records. It reports the
published fixed-k robustness index, biological-class balanced KNN accuracy, and
their mean as `robustness_quality`.

Runtime roots are the canonical shared locations declared in
`configs/main.yaml`: `/data/thunder-data`, `/data/ucla-lung`, `/data/surgen`,
`/data/leopard_bcr`, `/data/CPTAC-PDA`, and `/data/pathorob`.

## Promotion gates

Promotion requires two deterministic single-H100 runs under 1,500 seconds and
fresh matched-model comparisons against the official evaluations. Report
Pearson and Spearman, but decide rank fidelity from Kendall tau, all-model and
cross-family pairwise concordance, the explicit NanoPath-vs-GigaPath and
NanoPath-vs-H-Optimus-0 comparisons, and the NanoPath-family residual. Existing
v2 rows from the superseded 16-classification/four-segmentation protocol are not
comparable and must not be reused.
