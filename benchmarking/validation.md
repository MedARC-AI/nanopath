# Validation record

Validation asks two different questions:

1. Does the local implementation reproduce the intended official computation
   on identical frozen inputs?
2. Does the resulting development score preserve useful model ordering on
   official held-out suites?

The first is an implementation-parity check. The second is post-freeze evidence
about proxy fidelity, not permission to tune weights or datasets against test
outcomes.

The benchmark components and manifests were frozen at commit
`c21c9d8e1b824018badbf2a88b7693491f4daa4d` before the final official-result
audit. The updated 25/15/25/15/10/10 weights are a scoring-policy choice;
stored component results were rescored arithmetically. The selected THUNDER
manifest SHA-256 is
`fc9a92587f78078c1d3c880f95a795ff229affb61c67de1a7886221dc99a0b8b`. Later
release commits changed packaging, baseline launchers, comments, and
documentation without changing the manifest or component protocols.

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

Before official comparisons, benchmark heads were checked against the
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

The complete benchmark was run in independent clean processes on one
80 GB H100 with 16 CPUs:

| Model / feature policy | Wall time | Final score | Note |
|---|---:|---:|---|
| Representative nanopath ViT-S, run 1 | 1,156.7 s | 0.656388 | clean process |
| Representative nanopath ViT-S, run 2 | 1,198.9 s | 0.656217 | clean process |
| DINOv2-S reference | 1,018 s | 0.6198 | pretrained frozen baseline |
| H0-mini reference | 1,273.1 s | 0.7021 | official CLS-plus-mean readout |
| I-JEPA contig-patch nanopath | 1,187 s | 0.6648 | ordinary feature adapter |
| block-strided-cls nanopath | 1,188.6 s | 0.6591 | test-time aggregation exercised |
| robust-norm nanopath | 1,366.3 s | 0.6733 | 49,554 MiB peak; aggregation exercised |

The two independent representative scores differ by 0.000170, below the 0.001
determinism gate. Every listed run is below the 1,500-second release limit,
including the two feature-aggregation variants that motivated bounded spatial
pooling. Runtime depends on image-cache warmth, backbone size, feature width,
and CPU decode throughput; the limit is a release qualification on the target
H100, not a promise for arbitrary hardware.

## Training-seed audit and promotion margin

Two nanopath recipes were independently trained at seeds 17, 29, and 43 while
the data split and probe randomness stayed fixed:

| Recipe | Seed 17 | Seed 29 | Seed 43 | Mean | Sample SD |
|---|---:|---:|---:|---:|---:|
| Main DINOv2/KDE | 0.637656 | 0.637649 | 0.632825 | 0.636043 | 0.002787 |
| robust-norm | 0.671812 | 0.669810 | 0.670370 | 0.670664 | 0.001033 |

The pooled within-recipe run SD is 0.002102. A three-run mean therefore has an
estimated standard error of 0.001213, and the difference between two
independent three-run means has an estimated standard error of 0.001716. Its
one-sided 95% normal margin is 0.002823; the existing **0.004** promotion
margin remains a conservative fixed policy. This operating threshold does not
assume that matching seeds cancel variance between recipes and is not
recomputed for each candidate.
A discovery run is excluded, all three fixed confirmation seeds count, and a
promoted panel becomes the stored incumbent. No official evaluation result was
used in this calibration.

## Official-suite ordering fidelity

The promotion study contains six nanopath checkpoints and seven
principal baselines. Official results were read only after the benchmark,
manifests, and scalar were frozen.

Pairwise concordance is the fraction of non-tied model pairs ordered the same
way by nanopath and the official target. Cross-family concordance restricts
that calculation to nanopath-versus-baseline pairs, directly testing the
cross-family offset the benchmark is intended to detect. Pearson measures
score-shape agreement; Spearman and Kendall
measure rank agreement. None alone is treated as sufficient.

| Proxy / official target | Pearson | Spearman | All-pair concordance | Cross-family concordance |
|---|---:|---:|---:|---:|
| Classification / THUNDER classification | 0.987 | 0.995 | 0.987 | 1.000 |
| Segmentation / matched 3-task THUNDER segmentation | 0.743 | 0.637 | 0.782 | 0.857 |
| Segmentation / pinned full 4-task THUNDER segmentation | 0.668 | 0.558 | 0.753 | 0.833 |
| Final score / existing official composite, 12 models | 0.932 | 0.916 | 0.879 | 0.943 |

Classification preserves all 15 pairwise orderings among the six nanopath
checkpoints. Matched-task segmentation preserves 11 of 15 nanopath-only pairs;
its strongest evidence is cross-family separation, not exact within-family
ordering. The full four-task segmentation diagnostic includes all-TCGA OCELOT,
which is deliberately unavailable to nanopath. The published THUNDER aggregate
is also tracked because published GigaPath and Midnight-12K values differ from
the pinned harness; it yields 0.719 Pearson and 0.818 all-pair concordance.

Across those 12 pre-existing composite rows, the final score never places a
studied nanopath checkpoint above GigaPath or H-Optimus-0 when the composite
places it below that baseline.

An expanded 20-model table adds H0-mini, DINOv2-S/B/L/G, Kaiko-S/16, and
GigaPath-Flash:

| Comparison | Pearson | Spearman | All-pair concordance | Cross-family concordance |
|---|---:|---:|---:|---:|
| Classification / THUNDER, 20 models | 0.988 | 0.992 | 0.979 | 1.000 |
| Segmentation / THUNDER, 18 matched results | 0.850 | 0.792 | 0.848 | 0.903 |
| Final score / HEST, 20 models | 0.941 | 0.941 | — | — |
| Final score / CPTAC classification, 20 models | 0.851 | 0.872 | — | — |

DINOv2-S and DINOv2-G segmentation are omitted because only different-size
published DINO proxies were available. These exclusions are fixed by result
identity, not model performance.

The exact comparison input is
[proxy-fidelity data](proxy_fidelity_v2.csv). The 13 promotion rows use the
assembled fixed result, which combines PanNuke with both SegPath tasks. This
is important: their retained non-segmentation artifact paths happen to contain
intermediate two-SegPath metrics and must not be reopened as if those partial
runs were the final score. The seven added rows use their completed fixed
suite results. Empty DINOv2-S/G segmentation cells prevent different-size
published proxies from silently entering the 18-model statistic.

## Random-feature null audit

The benchmark was also run with independently randomized DINOv2-S backbones. This
checks that heads do not obtain implausibly strong scores from class balance,
spatial priors, slide leakage, or validation selection alone. The null audit
uses the exact production manifests, transforms, heads, folds, and scalar; only
the backbone initialization changes. Results are reported across ten seeds
rather than from a favorable draw; raw values are checked in as
[the random-feature audit](random_dinov2_s_v2.csv). The existing
[`baselines/dinov2_random_baseline.py`](../baselines/dinov2_random_baseline.py)
is the runner, so the benchmark does not carry a second stale null script.

| Component | Null mean | Sample SD | Min–max |
|---|---:|---:|---:|
| Final score | 0.5223 | 0.0028 | 0.5174–0.5261 |
| Classification | 0.3706 | 0.0026 | 0.3661–0.3739 |
| Segmentation | 0.5128 | 0.0047 | 0.5067–0.5202 |
| Progression | 0.6684 | 0.0085 | 0.6576–0.6841 |
| Mutation | 0.5502 | 0.0038 | 0.5437–0.5558 |
| Survival | 0.5985 | 0.0076 | 0.5842–0.6100 |
| Robustness quality | 0.4322 | 0.0022 | 0.4293–0.4361 |

All trained or pretrained reference final scores in
[the proxy-fidelity data](proxy_fidelity_v2.csv) exceed the largest random
final score by at least 0.094. Classification, mutation, and robustness provide
clear separation. The segmentation null is numerically high because
background and spatial priors earn F1. Every listed trained reference is at
least 0.036 above the random maximum.

Progression does **not** pass a clean random-feature interpretation: randomized
features average 0.668 AUC and outperform multiple trained references. Survival
also has weak separation, with a random mean of 0.598 and maximum of 0.610.
Those components may measure cohort/image shortcuts or useful random nonlinear
features as much as learned representation quality. They remain parts of the
fixed scalar, not trustworthy standalone claims. This null evidence is a
release limitation.

Nine null runs finished in 18:52–19:22. One took 26:17 while all ten jobs
contended for the shared image caches concurrently; it is retained in the null
distribution but is not a runtime-qualification run. The clean-process runtime
gate above remains the relevant 25-minute evidence.

## Known limitations

- PanNuke validation cannot be proven disjoint from TCGA pretraining at the
  image-source level.
- Segmentation does not perfectly preserve ordering among closely spaced
  nanopath checkpoints.
- Validation-set marginalization avoids selection leakage but does not reproduce
  the absolute score of THUNDER's validation-selected, test-reported heads.
- SPIDER-Skin has a one-example rare class in official validation, so its macro-
  F1 can move sharply when that example changes status.
- CPTAC-PDA survival makes the suite partly familiar with the CPTAC domain,
  though no CPTAC classification records or labels are used.
- A 0–1 weighted mean is transparent but not statistically calibrated across
  metrics with different variance. The robustness weight reflects a governance
  preference rather than a fit to official results.

For those reasons, the benchmark should guide efficient hill climbing and baseline
placement, not replace final evaluation on the intended official suites.
