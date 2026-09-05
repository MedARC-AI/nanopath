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
audit. The current 30/15/17.5/15/7.5/15 weights are a scoring-policy choice;
stored predictive results were rescored arithmetically and robustness was
recomputed from the same frozen embeddings with CRoMa. The selected THUNDER
manifest SHA-256 is
`fc9a92587f78078c1d3c880f95a795ff229affb61c67de1a7886221dc99a0b8b`. Later
release commits changed packaging, baseline launchers, comments, and
documentation without changing the manifest or predictive component protocols.

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

## Training-seed audit and promotion margin

Two nanopath recipes were independently trained at seeds 17, 29, and 43 while
the data split and probe randomness stayed fixed. These are audit seeds, not a
fixed promotion panel:

| Recipe | Seed 17 | Seed 29 | Seed 43 | Mean | Sample SD |
|---|---:|---:|---:|---:|---:|
| Main DINOv2/KDE | 0.619722 | 0.620240 | 0.615958 | 0.618640 | 0.002337 |
| robust-norm | 0.645294 | 0.645239 | 0.646023 | 0.645518 | 0.000438 |

CRoMa's weighted three-seed range is 0.000593 for main and 0.001206 for
robust-norm; the complete revised score ranges are 0.004282 and 0.000784.
The pooled within-recipe run SD is 0.001681. The **0.004** promotion margin is a
fixed conservative policy and is not recomputed per candidate. A maintainer
reruns a candidate with three different randomly selected seeds; the median run
must clear the margin. The discovery run is excluded. No official evaluation
result was used in this calibration.

### Progression variance audit (2026-09-05)

The same six completed checkpoints above were re-embedded using their saved
model implementations. UCLA's 90-slide manifest (35/55 labels), all 26,714
tiles, raw mean pooling, and balanced logistic head (`C=0.5`) stayed fixed.
Re-extraction reproduced all six released progression AUCs within 1e-15.
Only the AUC estimator changed; no official test image, label, or score entered
the comparison. The manifest also exactly matched PathoBench's fold-0 training
IDs and labels at revision `60fde3a9138b2fb27a163ed6f3e2cf0ef7e8f387`.

| Recipe / training seed | V2: 3 folds | 20 × 3 folds | 20 × 5 folds | Leave-pair-out |
|---|---:|---:|---:|---:|
| Main / 17 | 0.520305 | 0.572976 | 0.573571 | 0.569091 |
| Main / 29 | 0.524935 | 0.575204 | 0.569286 | 0.571688 |
| Main / 43 | 0.515676 | 0.555293 | 0.546558 | 0.540000 |
| robust-norm / 17 | 0.586922 | 0.600416 | 0.595130 | 0.581558 |
| robust-norm / 29 | 0.583784 | 0.598404 | 0.595000 | 0.577922 |
| robust-norm / 43 | 0.582241 | 0.617034 | 0.614740 | 0.592987 |
| **Main sample SD** | **0.004630** | **0.010909** | **0.014518** | **0.017593** |
| **robust-norm sample SD** | **0.002386** | **0.010225** | **0.011360** | **0.007861** |

Repeated folds used `RepeatedStratifiedKFold(random_state=1337)`; AUCs were
averaged within folds, never pooled across separately fitted heads. Each
leave-pair-out head excluded one positive and one negative slide, then ranked
that pair; the score averaged all 1,925 comparisons, with ties worth 0.5.
The first 20 repetitions were specified before results. Extending to 200
repetitions did not reverse either recipe's variance increase. Seeds 59/71
were also inspected but excluded: main stopped at step 5,000 and robust-norm
at step 3,000, short of their completed-run budgets.

**None of these replacements qualifies as a training-seed variance fix.**
At the study's then-current 25% progression weight, main's 0.009259 AUC range
contributed 0.002315 to `final_score`; 20 × 3 folds increased it to 0.004978. More folds reduce
sensitivity to a chosen partition, but do not create independent patients or
necessarily reduce sensitivity to backbone training. Three training seeds per
recipe are discovery evidence, not a precise population variance estimate.
Fold SD is neither training-seed SD nor a confidence interval.

Three freshly randomized DINOv2-S backbones still scored 0.6304–0.6345 under
leave-pair-out, above the pretrained DINOv2-S score of 0.5338. Changing the
estimator therefore did not resolve the random-feature limitation either.
GPU work used at most four independent single-GPU jobs; head comparisons ran
on CPU. Local scripts, cached features, raw fold/pair results, and provenance
checks are under `/data/paul/nanopath/progression-study-20260905/`.

The progression estimator and promotion rule remain unchanged. Independent development
patients and UCLA patient grouping remain priorities (see
[slide_probes.md](slide_probes.md)); the cohort study below evaluates four
concrete additions under the then-current progression category's 25% weight.

### Progression cohort audit (2026-09-05)

Four PathoBench fold-0 training cohorts were evaluated on the exact 20 frozen
encoders and official targets used by `threepanel-v2.png`, plus the six completed
training-seed replicates above and three random DINOv2-S controls. The primary
protocol uses at most 128 deterministic, source-spaced tiles per slide, frozen
mean pooling, and the existing three-fold head. Classification uses balanced
raw-feature logistic regression (`C=0.5`); Valentino PFS uses the existing
censoring-aware Cox head. Caps 64/256 are sensitivities, not tuned against the
official scores. Each addition shares the then-current 25% progression weight
equally with UCLA: `new_final = old_final + 0.125 * (new_component - UCLA_AUC)`.

| Progression component | THUNDER Kendall τ | HEST Kendall τ | CPTAC Kendall τ |
|---|---:|---:|---:|
| Current UCLA | 0.789 | 0.821 | 0.716 |
| UCLA + VisioMel relapse, 1,073 patients | 0.779 | 0.789 | 0.663 |
| UCLA + breast residual burden, 128 patients | 0.821 | 0.789 | 0.642 |
| UCLA + HER2 response, 68 patients | 0.663 | 0.653 | 0.568 |
| UCLA + Valentino PFS, 136 patients | 0.842 | 0.768 | 0.663 |
| VisioMel replaces UCLA, sensitivity | 0.663 | 0.716 | 0.589 |

**None qualifies as a reliable addition under this fast protocol.** VisioMel
reduces main's progression contribution SD from 0.001157 to 0.000463, but raises
robust-norm's from 0.000596 to 0.001468. Its Nanopath-only Kendall values remain
unchanged; concordant Nanopath-versus-reference pairs fall from 75/78/74 to
75/74/70 out of 84. Its 256-tile sensitivity retains the same three correlations
and reduces both recipes' contribution SD by only about 3%. The smaller cohorts
all increase main's progression variation. These are three-seed observations,
not precise population variance estimates. VisioMel's paired model-bootstrap
intervals for Kendall changes include zero; this is insufficient evidence for
adoption, not a statistically decisive proof of harm.

VisioMel's selected 1,073 patients (169 positive) match two identical archived
2023 challenge-training label files and exclude all 541 original test IDs as
well as PathoBench fold-0 test IDs. Repeated PathoBench partitions reuse patients;
this reserves fold-0 tests, not the union of every repeated test partition.
Eighteen positive cases have released relapse times beyond 60 months, so the
released binary labels are retained without claiming an adjudicated strict
five-year endpoint. One nearly achromatic image required a documented tissue-mask
threshold correction; all patients remain included. Random encoders score
0.713–0.718 and simple colour/tissue features score 0.678 on VisioMel.

The unmodified production slide probe adds 95–101 seconds on the main and
robust-norm readouts using an 11.67 GB prepared cache. Recurring cost is feasible;
one-time source preparation transferred approximately 1.32 TB. Since the candidate
failed the ordering/stability comparison, no new full-suite timing run was needed.
Existing official aggregate scores were reused for this retrospective analysis;
no official test image or label entered the new probes. Choosing a new protocol
using these aggregates would still require independent validation, versioning,
and promotion-margin recalibration. The current V2 score remains unchanged.

The full report, per-model CSV, figures, scripts, source/split audits, cached
features, and raw results are under
`/data/paul/nanopath/progression-cohorts-20260905/` (`report.md`, `analysis.json`,
`model-results.csv`). GPU work used at most four independent single-GPU jobs.

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
| Final score / existing official composite, 12 models | 0.902 | 0.888 | 0.864 | 0.914 |

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
| Final score / THUNDER classification + segmentation | 0.927 | 0.800 |
| Final score / HEST | 0.929 | 0.789 |
| Final score / CPTAC classification | 0.814 | 0.684 |

The exact comparison input is
[proxy-fidelity data](proxy_fidelity_v2.csv). Final scores use the assembled
fixed result, including PanNuke and both SegPath tasks. THUNDER segmentation
uses complete same-checkpoint results for all 20 models.

## Random-feature null audit

The benchmark was also run with independently randomized DINOv2-S backbones. This
checks that heads do not obtain implausibly strong scores from class balance,
spatial priors, slide leakage, or validation selection alone. The null audit
uses the exact production manifests, transforms, heads, folds, and scalar; only
the backbone initialization changes. All ten seeds were rescored with CRoMa;
the original component results and revised scores are retained in
[the random-feature audit](random_dinov2_s_v2.csv). The existing
[`baselines/dinov2_random_baseline.py`](../baselines/dinov2_random_baseline.py)
is the runner, so the benchmark does not carry a second stale null script.

| Component | Null mean | Sample SD | Min–max |
|---|---:|---:|---:|
| Final score | 0.4731 | 0.0028 | 0.4682–0.4764 |
| Classification | 0.3706 | 0.0026 | 0.3661–0.3739 |
| Segmentation | 0.5128 | 0.0047 | 0.5067–0.5202 |
| Progression | 0.6684 | 0.0085 | 0.6576–0.6841 |
| Mutation | 0.5502 | 0.0038 | 0.5437–0.5558 |
| Survival | 0.5985 | 0.0076 | 0.5842–0.6100 |
| CRoMa robustness | 0.2706 | 0.0071 | 0.2569–0.2795 |

All trained or pretrained reference final scores in
[the proxy-fidelity data](proxy_fidelity_v2.csv) exceed the largest rescored random
final score by at least 0.118. Classification, mutation, and robustness provide
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
