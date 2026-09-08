# Validation record

Validation asks two different questions:

1. Does the local implementation reproduce the intended official computation
   on identical frozen inputs?
2. Does the resulting development score preserve useful model ordering on
   official held-out suites?

The first is an implementation-parity check. The second is post-freeze evidence
about proxy fidelity, not permission to tune weights or datasets against test
outcomes.

The selected THUNDER manifest SHA-256 is
`fc9a92587f78078c1d3c880f95a795ff229affb61c67de1a7886221dc99a0b8b`.
The 30/15/17.5/15/7.5/15 weights are a fixed scoring-policy choice.

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
| CRoMa `m=5` sample margins | maximum difference from upstream 1.06e-11 across 58 checks |

Segmentation additionally uses the same present-class per-image F1/Jaccard and
foreground/background image weighting as the pinned official THUNDER harness.

### CRoMa revision audit (2026-09-05)

CRoMa was evaluated on the existing all-row Camelyon and non-TCGA Tolkach
cohorts for 20 reference encoders, six training-seed checkpoints, and three
random controls. The implementation was matched to upstream commit
`3f58d5e4bd9ecf74c34d0c76eb88d80cee9fb706`: all samples are evaluation units,
same-slide neighbors are excluded, the nearest five `SO` and `OS` distances are
averaged, and the median signed margin is the cohort score. Across both cohorts
and all 29 encoders, production-`m=5` sample margins matched upstream within
1.06e-11 on deterministic 1,024-row subsets. An independent check of the final
production function reproduced the full-cohort study scores and matched upstream
median, quantile, tail mean, and F(0) summaries within 2.32e-12 across all 58
cohort/model subsets.

The final scalar uses CRoMa alone, not the previous biological-accuracy mixture:
the two signed cohort medians are independently mapped by `(1 + croma) / 2` and
averaged. Float64 distance arithmetic is intentional. On the OpenMidnight
precision audit, float32 shifted the Camelyon and Tolkach medians by -0.00329
and -0.01440, respectively. Full study inputs, cached embeddings, scripts, and
results are under `/data/paul/nanopath/croma-study-20260905/`.

## Runtime and determinism

The complete benchmark was run in independent clean processes on one
80 GB H100 with 16 CPUs:

| Model / feature policy | Wall time | Note |
|---|---:|---|
| Representative nanopath ViT-S, run 1 | 1,156.7 s | clean process |
| Representative nanopath ViT-S, run 2 | 1,198.9 s | clean process |
| DINOv2-S reference | 1,018 s | pretrained frozen baseline |
| H0-mini reference | 1,273.1 s | official CLS-plus-mean readout |
| I-JEPA contig-patch nanopath | 1,187 s | ordinary feature adapter |
| block-strided-cls nanopath | 1,188.6 s | test-time aggregation exercised |
| robust-norm nanopath | 1,366.3 s | 49,554 MiB peak; aggregation exercised |

The fresh `main-repro` CRoMa run completed the entire production probe suite in
1,146.8 seconds (19:07) after 999,936 training tile presentations on one H100.
The requested `robust-norm-v2-repro` retraining completed 993,792 presentations
and the full suite in 1,135.4 seconds (18:55). Their measured final scores are
0.627725 and 0.646750, respectively; both replace their historical Labless
evaluations and do not establish a new leader.

The two independent representative pre-CRoMa scores differ by 0.000170, below
the 0.001 determinism gate. The timings in the table predate the metric revision,
but the expensive image decoding and encoder paths are unchanged; across the
29 cached study encoders, float64 CRoMa took 0.87–3.34 seconds for Camelyon and
0.81–1.84 seconds for Tolkach. Every listed run is below the 1,500-second release limit,
including the two feature-aggregation variants that motivated bounded spatial
pooling. Runtime depends on image-cache warmth, backbone size, feature width,
and CPU decode throughput; the limit is a release qualification on the target
H100, not a promise for arbitrary hardware.

## Training-seed validation

A maintainer reruns promising candidates with three different randomly
selected training seeds. The median run must beat the incumbent by at least
**0.004** to become the validated leader. The discovery run is excluded;
`robust-norm-s9876` is the approved exception. This is a fixed promotion
policy, not a margin recalculated from each candidate's measured variance.

### Progression stability

Across six three-seed groups (18 completed checkpoints), UCLA's pooled
within-group sample SD is 0.011269. The groups cover related main,
robust-norm and drop-local recipes on the same 90-slide cohort.
Observed within-group AUC ranges are 0.015750–0.025879, contributing
0.002756–0.004529 to the weighted score. Repeated splits do not eliminate
training-seed variability or the need for independent training runs.

Seven 1,000-repeat checks yield weighted SD 0.00030–0.00050 across disjoint
100-repeat blocks. One hundred complete-protocol label permutations each
for main and UNI2-h average 0.48908 and 0.49258 AUC. The 300-fit head takes
about 3 seconds for main and at most 27 seconds across the 20 reference
encoders on one CPU thread. Frozen-input head and aggregation checks cover
41 distinct encoders, including training seeds and random controls.

All 26,714 UCLA tiles are used. Per-slide counts range from 12 to 1,453
(median 240.5); sixteen slides have fewer than 64 tiles. In one training
triplet, held-out probability SD averages 0.0345 below 64 tiles and 0.0338
otherwise. This does not establish tissue coverage as the dominant source
of instability, nor exclude a coverage problem. The missing patient mapping
and high random-feature AUC remain limitations of this cohort.

## Official-suite ordering fidelity

The earlier promotion study contained six nanopath checkpoints and seven
principal baselines. Its table below records the pre-CRoMa protocol: official
results were read after that benchmark, its manifests, and its scalar were
frozen. The current CRoMa comparisons follow in the expanded 20-model table.

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
| Final score / existing official composite, 12 models | 0.876 | 0.860 | 0.848 | 0.886 |

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

| Comparison, 20 models | Pearson | Kendall |
|---|---:|---:|
| Classification / THUNDER | 0.988 | 0.958 |
| Segmentation / THUNDER | 0.870 | 0.741 |
| Final score / THUNDER classification + segmentation | 0.923 | 0.758 |
| Final score / HEST | 0.896 | 0.768 |
| Final score / CPTAC classification | 0.797 | 0.684 |

The exact comparison input is
[proxy-fidelity data](proxy_fidelity_v2.csv). Final scores use the assembled
fixed result, including PanNuke and both SegPath tasks. THUNDER segmentation
uses complete same-checkpoint results for all 20 models.

## Random-feature null audit

The benchmark was also run with independently randomized DINOv2-S backbones. This
checks that heads do not obtain implausibly strong scores from class balance,
spatial priors, slide leakage, or validation selection alone. The null audit
uses the production manifests, transforms, heads, folds, and scalar; only
the backbone initialization changes. All ten seeds were rescored with CRoMa;
the original component results and revised scores are retained in
[the random-feature audit](random_dinov2_s_v2.csv). The existing
[`baselines/dinov2_random_baseline.py`](../baselines/dinov2_random_baseline.py)
is the runner, so the benchmark does not carry a second stale null script.

| Component | Null mean | Sample SD | Min–max |
|---|---:|---:|---:|
| Final score | 0.4669 | 0.0021 | 0.4631–0.4701 |
| Classification | 0.3706 | 0.0026 | 0.3661–0.3739 |
| Segmentation | 0.5128 | 0.0047 | 0.5067–0.5202 |
| Progression | 0.6330 | 0.0024 | 0.6285–0.6361 |
| Mutation | 0.5502 | 0.0038 | 0.5437–0.5558 |
| Survival | 0.5985 | 0.0076 | 0.5842–0.6100 |
| CRoMa robustness | 0.2706 | 0.0071 | 0.2569–0.2795 |

All trained or pretrained reference final scores in
[the comparison data](proxy_fidelity_v2.csv) exceed the largest random
final score by at least 0.122. Classification, mutation, and robustness provide
clear separation. The segmentation null is numerically high because
background and spatial priors earn F1. Every listed trained reference is at
least 0.036 above the random maximum.

Progression does **not** pass a clean random-feature interpretation: randomized
features average 0.633 AUC and outperform multiple trained references. Survival
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
