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
`mean_probe_score` and the summary alias `final_probe_score` are the only
reported scalar.

## Fixed suite

| Family | Datasets | Scored metric |
|---|---|---|
| Classification | BACH, BRACS, BreaKHis, CRC, ESCA, MHIST, PCam, SPIDER breast/colorectal/skin/thorax, WILDS | macro-F1 from linear, KNN, and 16-shot SimpleShot |
| Segmentation | PanNuke, SegPath epithelial, SegPath lymphocytes | THUNDER per-image weighted macro-F1 |
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
MoNuSAC is not a substitute because its released cohort is TCGA-derived.

PanNuke uses the complete official Fold1 training array (2,656 images) and
Fold2 validation array (2,523 images). Fold3 is absent from the manifest and is
never opened. PanNuke mixes TCGA and local-hospital material without per-image
provenance, so it cannot be filtered to a verifiably non-TCGA subset; it is
retained as the explicitly accepted mixed-source exception after SegPath alone
failed to preserve NanoPath-family segmentation ordering. SegPath uses
source-disjoint official development splits: 32,768/4,518 epithelial crops and
20,906/2,164 lymphocyte crops.

## Probe protocols

NanoPath exposes two model-defined test-time readouts. `probe_features()` feeds
classification, progression, mutation, and survival, so recipes may aggregate
layers or views there. `encode_image()` feeds segmentation; all model-defined
channels are retained, while an expanded spatial grid is pooled back to the
native patch grid to bound decoder memory and runtime. PathoROB deliberately
does not use `probe_features()`: its published adapter remains fixed to final
CLS plus mean patch tokens, including any normalization applied by the
backbone's ordinary forward pass.

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
`lr=1e-3`, `weight_decay=1e-4`; PanNuke trains 30 epochs at `lr=1e-3`,
`weight_decay=1e-4`. Dense tokens are cached as signed int8 vectors with fp16
scales. All feature extraction is microbatched at 512 images so models may
aggregate multiple intermediate layers or test-time views without excessive
peak memory. For DINO-family encoders, expanded spatial grids are area-pooled
to the native patch grid before the shared decoder, while all model-defined
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
`/data/leopard_bcr`, `/data/CPTAC-PDA`, and `/data/pathorob`. Missing roots are
downloaded from the immutable `medarc/nanopath-evals` snapshot; on machines
without shared `/data`, preparation localizes them under the clone's ignored
`data/` directory. That snapshot contains only the selected development data,
not official test records.

## Promotion gates

Promotion required two deterministic single-H100 runs under 1,500 seconds and
fresh matched-model comparisons against the official evaluations. Report
Pearson and Spearman, but decide rank fidelity from Kendall tau, all-model and
cross-family pairwise concordance, the explicit NanoPath-vs-GigaPath and
NanoPath-vs-H-Optimus-0 comparisons, and the NanoPath-family residual.

The primary segmentation correlation compares the three THUNDER tasks present
in NanoPath with their official held-out results. The full four-task THUNDER
aggregate remains an out-of-distribution ordering diagnostic, but cannot be the
matched correlation target because its fourth task is the deliberately excluded
all-TCGA OCELOT dataset. Both the pinned harness aggregate and the published
THUNDER aggregate are reported because their GigaPath and Midnight-12K values
materially disagree.
