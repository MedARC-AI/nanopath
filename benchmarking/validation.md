# Protocol-v2 validation record

Validation asks two different questions:

1. Does the local implementation reproduce the intended official computation
   on identical frozen inputs?
2. Does the resulting development score preserve useful model ordering on
   official held-out suites?

The first is an implementation-parity check. The second is post-freeze evidence
about proxy fidelity, not permission to tune weights or datasets against test
outcomes.

The scoring implementation was frozen at commit
`c21c9d8e1b824018badbf2a88b7693491f4daa4d` before the final official-result
audit. The selected THUNDER manifest SHA-256 is
`fc9a92587f78078c1d3c880f95a795ff229affb61c67de1a7886221dc99a0b8b`. Later
release commits changed packaging, baseline launchers, comments, and
documentation without changing the manifest, probe math, or scalar.

## Leakage and manifest audit

The release audit verifies that:

- every THUNDER manifest entry has exactly `root`, `train`, and `val`;
- every referenced path exists and train/validation records are disjoint;
- capped training sets retain at least 16 examples per class and all official
  validation classes remain represented;
- SPIDER source slides, WILDS patient/node groups, and SegPath source images are
  disjoint across train and validation;
- ESCA validation is entirely UKK;
- PanNuke Fold3 and every official THUNDER test path are absent;
- PathoBench fold-0 test records, LEOPARD challenge test records, HEST, and
  CPTAC classification data are absent;
- the downloadable Tolkach data excludes its TCGA center.

The sole unresolved source-level overlap is PanNuke Fold2: its release mixes
TCGA and local-hospital images without recoverable per-image provenance. It is
retained as an explicit exception, not represented as TCGA-free.

## Frozen-input parity

Before official comparisons, protocol-v2 heads were checked against the
corresponding THUNDER/PathoBench computation on identical synthetic or cached
embeddings:

| Component | Result |
|---|---|
| KNN predictions and macro-F1 | exact |
| 1,000-draw centered SimpleShot | exact |
| Nine-head Adam linear probe | maximum score difference 2.89e-8 |
| MaskTransformer forward path | maximum absolute output difference 1.86e-5 |
| Multiclass Dice objective | exact |
| Fixed balanced logistic probe | matched |
| Fold-standardized CoxNet protocol | verified |

Segmentation additionally uses the same present-class per-image F1/Jaccard and
foreground/background image weighting as the pinned official THUNDER harness.

## Runtime and determinism

The current complete protocol was run in independent clean processes on one
80 GB H100 with 16 CPUs:

| Model / feature policy | Wall time | Final score | Note |
|---|---:|---:|---|
| Representative NanoPath ViT-S, run 1 | 1,156.7 s | 0.637050 | clean process |
| Representative NanoPath ViT-S, run 2 | 1,198.9 s | 0.636834 | clean process |
| DINOv2-S reference | 1,018 s | 0.6104 | pretrained frozen baseline |
| I-JEPA contig-patch NanoPath | 1,187 s | 0.6439 | ordinary feature adapter |
| block-strided-cls NanoPath | 1,188.6 s | 0.6411 | test-time aggregation exercised |
| robust-norm NanoPath | 1,366.3 s | 0.6464 | 49,554 MiB peak; aggregation exercised |

The two independent representative scores differ by 0.000216, below the 0.001
determinism gate. Every listed run is below the 1,500-second release limit,
including the two feature-aggregation variants that motivated bounded spatial
pooling. Runtime depends on image-cache warmth, backbone size, feature width,
and CPU decode throughput; the limit is a release qualification on the target
H100, not a promise for arbitrary hardware.

## Training-seed audit and promotion margin

Two NanoPath recipes were independently trained at seeds 17, 29, and 43 while
the data split and probe randomness stayed fixed:

| Recipe | Seed 17 | Seed 29 | Seed 43 | Mean | Sample SD |
|---|---:|---:|---:|---:|---:|
| Main DINOv2/KDE | 0.623746 | 0.621826 | 0.616112 | 0.620561 | 0.003971 |
| robust-norm | 0.648837 | 0.646516 | 0.646570 | 0.647308 | 0.001325 |

The estimated standard error between two independent three-run means is
0.002417. Its one-sided 95% Welch margin is 0.006262; rounding up to the next
0.001 fixes the protocol-v2 promotion margin at **0.007**. This number is not
recomputed for each candidate. A discovery run is excluded, all three fixed
confirmation seeds count, and a promoted panel becomes the stored incumbent.
No official evaluation result was used in this calibration.

## Official-suite ordering fidelity

The original promotion study contains five NanoPath checkpoints and seven
principal baselines. Official results were read only after the protocol,
manifests, and scalar were frozen.

Pairwise concordance is the fraction of non-tied model pairs ordered the same
way by NanoPath and the official target. Cross-family concordance restricts
that calculation to NanoPath-versus-baseline pairs, directly testing the offset
that motivated v2. Pearson measures score-shape agreement; Spearman and Kendall
measure rank agreement. None alone is treated as sufficient.

| Proxy / official target | Pearson | Spearman | All-pair concordance | Cross-family concordance |
|---|---:|---:|---:|---:|
| Classification / THUNDER classification | 0.987 | 0.993 | 0.985 | 1.000 |
| Segmentation / matched 3-task THUNDER segmentation | 0.735 | 0.608 | 0.758 | 0.857 |
| Segmentation / pinned full 4-task THUNDER segmentation | 0.656 | 0.515 | 0.720 | 0.829 |
| Final score / existing official composite | 0.935 | 0.916 | 0.879 | 0.943 |

Classification preserves all 10 pairwise orderings among the five NanoPath
checkpoints. Matched-task segmentation preserves 6 of 10 NanoPath-only pairs;
its strongest evidence is cross-family separation, not exact within-family
ordering. The full four-task segmentation diagnostic includes all-TCGA OCELOT,
which is deliberately unavailable to NanoPath. The published THUNDER aggregate
is also tracked because published GigaPath and Midnight-12K values differ from
the pinned harness; it yields 0.719 Pearson and 0.818 all-pair concordance.

The final score never places a studied NanoPath checkpoint above GigaPath or
H-Optimus-0 when the existing official composite places it below that baseline.
Relative to v1, the standardized NanoPath-family residual shrinks for every
target: 89.3% for THUNDER classification, 74.9% for THUNDER segmentation, 35.7%
for HEST, 72.7% for CPTAC classification, and 53.3% for the official composite
(65.2% mean reduction).

An expanded 19-model table adds DINOv2-S/B/L/G, Kaiko-S/16, GigaPath-Flash, and
EXAONE-Path-2.5-B:

| Comparison | Pearson | Spearman | All-pair concordance | Cross-family concordance |
|---|---:|---:|---:|---:|
| Classification / THUNDER, 19 models | 0.988 | 0.991 | 0.977 | 1.000 |
| Segmentation / THUNDER, 17 matched results | 0.586 | 0.798 | 0.844 | 0.900 |
| Final score / HEST, 19 models | 0.878 | 0.837 | — | — |
| Final score / CPTAC classification, 19 models | 0.752 | 0.782 | — | — |

The lower expanded segmentation Pearson is driven chiefly by an EXAONE score-
magnitude miss; removing that row raises Pearson to 0.831. DINOv2-S and
DINOv2-G segmentation are omitted because only different-size published DINO
proxies were available. These exclusions are fixed by result identity, not
model performance.

The exact comparison input is
[`proxy_fidelity_v2.csv`](proxy_fidelity_v2.csv). The original 12 rows use the
assembled fixed v2 result, which combines PanNuke with both SegPath tasks. This
is important: their retained non-segmentation artifact paths happen to contain
intermediate two-SegPath metrics and must not be reopened as if those partial
runs were the final v2 score. The seven added rows use their completed fixed
suite results. Empty DINOv2-S/G segmentation cells prevent different-size
published proxies from silently entering the 17-model statistic.

## Random-feature null audit

Protocol v2 was also run with independently randomized DINOv2-S backbones. This
checks that heads do not obtain implausibly strong scores from class balance,
spatial priors, slide leakage, or validation selection alone. The null audit
uses the exact production manifests, transforms, heads, folds, and scalar; only
the backbone initialization changes. Results are reported across ten seeds
rather than from a favorable draw; raw values are checked in as
[`random_dinov2_s_v2.csv`](random_dinov2_s_v2.csv). The existing
[`baselines/dinov2_random_baseline.py`](../baselines/dinov2_random_baseline.py)
is the runner, so the benchmark does not carry a second stale null script.

| Component | Null mean | Sample SD | Min–max |
|---|---:|---:|---:|
| Final score | 0.5347 | 0.0023 | 0.5302–0.5375 |
| Classification | 0.3706 | 0.0026 | 0.3661–0.3739 |
| Segmentation | 0.5128 | 0.0047 | 0.5067–0.5202 |
| Progression | 0.6684 | 0.0085 | 0.6576–0.6841 |
| Mutation | 0.5502 | 0.0038 | 0.5437–0.5558 |
| Survival | 0.5985 | 0.0076 | 0.5842–0.6100 |
| Robustness quality | 0.4322 | 0.0022 | 0.4293–0.4361 |

All trained or pretrained reference final scores in
[`proxy_fidelity_v2.csv`](proxy_fidelity_v2.csv) exceed the largest random
final score by at least 0.073. Classification, mutation, and robustness provide
clear separation. The segmentation null is numerically high because
background and spatial priors earn F1. Every listed trained reference except
EXAONE is at least 0.036 above the random maximum; EXAONE's 0.368 is well below
it, consistent with the score-magnitude miss in the official comparison.

Progression does **not** pass a clean random-feature interpretation: randomized
features average 0.668 AUC and outperform multiple trained references. Survival
also has weak separation, with a random mean of 0.598 and maximum of 0.610.
Those components may measure cohort/image shortcuts or useful random nonlinear
features as much as learned representation quality. They remain only as parts
of the five-family mean, not trustworthy standalone claims. This null evidence
is a release limitation; it was not used to reweight the already frozen scalar.

Nine null runs finished in 18:52–19:22. One took 26:17 while all ten jobs
contended for the shared image caches concurrently; it is retained in the null
distribution but is not a runtime-qualification run. The clean-process runtime
gate above remains the relevant 25-minute evidence.

## Known limitations

- PanNuke validation cannot be proven disjoint from TCGA pretraining at the
  image-source level.
- Segmentation is substantially better aligned than v1 but does not perfectly
  preserve ordering among closely spaced NanoPath checkpoints.
- Validation-set marginalization avoids selection leakage but does not reproduce
  the absolute score of THUNDER's validation-selected, test-reported heads.
- SPIDER-Skin has a one-example rare class in official validation, so its macro-
  F1 can move sharply when that example changes status.
- CPTAC-PDA survival makes the suite partly familiar with the CPTAC domain,
  though no CPTAC classification records or labels are used.
- A 0–1 weighted mean is transparent but not statistically calibrated across
  metrics with different variance. Official results are not used to fit weights.

For those reasons, v2 should guide efficient hill climbing and baseline
placement, not replace final evaluation on the intended official suites.
