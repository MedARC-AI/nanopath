# Slide-level progression, mutation, and survival probes

These probes use only development cohorts. UCLA Lung, SurGen, and CPTAC-PDA
come from the original PathoBench fold-0 training pools; their fold-0 test
records are absent. LEOPARD uses public challenge training labels and never
opens the challenge test set.

The `train` and `val` keys in the UCLA and SurGen manifests are storage
partitions inherited from preparation. At runtime they are pooled back together
and split into the fixed development folds described below. Their manifest
`val` key is therefore not a separately scored official validation set.
LEOPARD and CPTAC-PDA instead store their three fixed folds directly. This is
distinct from the THUNDER train/validation protocol.

Task metadata comes from
[PathoBench](https://huggingface.co/datasets/MahmoodLab/Patho-Bench). Primary
image/cohort sources are [IDR idr0082](https://idr.openmicroscopy.org/search/?query=Name:idr0082)
for UCLA Lung, the [SurGen SR386 release](https://github.com/CraigMyles/SurGen-Dataset),
the [LEOPARD challenge](https://leopard.grand-challenge.org/), and
[TCIA CPTAC-PDA](https://www.cancerimagingarchive.net/collection/cptac-pda/).
Original dataset terms continue to apply to the mirrored development assets.

## Cohorts and cached tiles

| Probe | Development cohort | Outcome | Units / balance | Tile use |
|---|---|---|---|---|
| UCLA Lung progression | PathoBench `ucla_lung/progression_regression` fold-0 train pool; IDR idr0082 | progression class | 90 slides; 35 / 55 | All 26,714 cached tiles; 12–1,453 per slide |
| SurGen RAS mutation | PathoBench `sr386_/ras_mutant_binary` fold-0 train pool | wild type / RAS mutant | 311 slides; 201 / 110 | Deterministic source-spaced cap of 768 per slide; 219,505 embedded |
| LEOPARD BCR | public LEOPARD challenge training cohort | biochemical recurrence survival | 174 cases; all 87 events plus 87 longest-follow-up censored cases | Exactly 768 per slide; 133,632 embedded |
| CPTAC-PDA OS | PathoBench `cptac_pda/OS` fold-0 train pool | overall survival | 77 cases; 56 events / 21 censored; 184 slides | Full cached grid; 131,136 embedded |

The prepared parquets contain deterministic 20x, 512-pixel, zero-overlap
tissue grids. They follow PathoBench's tile geometry but use a lightweight
thumbnail tissue mask rather than Trident's HEST tissue segmenter. Cached JPEGs
are inference inputs, not additional labels. SurGen and LEOPARD use at most 768
raster/source-spaced tiles per slide to bound runtime; UCLA and CPTAC-PDA use
their full prepared grids.

For every task, the frozen encoder's `probe_features()` runs under fp16 autocast
and the resulting float32 tile vectors are averaged to slides. Survival then
averages multiple slide vectors belonging to the same case. No attention MIL
model or end-to-end finetuning is used.

## Progression and mutation

UCLA and SurGen each use three `StratifiedKFold` splits with shuffle seed 1337
over the complete development pool. On every fold:

- fit raw, unstandardized pooled features with
  `LogisticRegression(C=0.5, class_weight="balanced", random_state=0,
  max_iter=5000)`;
- report macro one-vs-rest validation AUC;
- average the three fold scores.

The head and `C` are fixed. There is no validation hyperparameter selection,
matching PathoBench's raw-feature linear protocol more closely than selecting
`C` on the same folds being reported.

## Survival

LEOPARD has three checked-in event-balanced 116/58 case folds. CPTAC-PDA has
three checked-in event-stratified folds with 51–52 training and 25–26
validation cases. Within each fold:

1. Fit `StandardScaler` on training cases only and transform train/validation.
2. Estimate the fold's CoxNet `alpha_max` on training data.
3. Fit three separate CoxNet heads at `0.1`, `0.2`, and `0.7` times that
   `alpha_max`, with `l1_ratio=0.5` and `max_iter=100000`.
4. Report Harrell's validation c-index for every fraction.

`survival_mean_cindex` averages every dataset × alpha fraction × fold c-index.
No best alpha is selected. Convergence warnings and numerical errors fail the
evaluation rather than being suppressed or replaced with a fallback result.

An absolute alpha scale is not comparable across models with different feature
dimensions and magnitudes. Fixed fractions of each training fold's `alpha_max`
preserve the same relative regularization strengths without looking at
validation outcomes.

## CPTAC interpretation

CPTAC-PDA survival is not the official CPTAC tile-classification evaluation.
Using a labeled CPTAC-PDA development cohort as one of two survival probes is
legitimate: its evaluation fold is internal to the declared development pool,
its held-out PathoBench records remain sealed, and no official CPTAC
classification sample or label is used.

It does mean the final score is not completely domain-independent of CPTAC.
Consequently, correlation with official CPTAC classification is treated as
post-freeze external validation of transfer and ordering, not as an untouched
estimate of generalization to an unseen institution or organ. CPTAC-PDA
survival contributes half of the 18%-weighted survival family, or 9% of the
final scalar.
