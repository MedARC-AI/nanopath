# nanopath benchmark

This directory is the audit trail for nanopath's fixed downstream benchmark.
The benchmark is a fast development-data proxy for model ordering on held-out
THUNDER, HEST, and CPTAC evaluations. It is not a replacement for those
evaluations and does not expose or score their test samples.

The executable definition is [`probe.py`](../probe.py) plus the five checked-in
JSON manifests in this directory. The prose here explains that definition; if
the code, manifests, and documentation ever disagree, the release is not valid.
Labless locks `probe.py`, this entire directory, and the probe configuration for
comparable public runs.

## Score definition

All components remain on their natural 0–1 scales:

```text
classification = mean over 12 datasets of
                 mean(linear-grid mean F1, KNN-grid mean F1, SimpleShot F1)

segmentation    = mean(PanNuke F1, SegPath epithelial F1,
                       SegPath lymphocyte F1)
progression     = UCLA Lung macro-OVR AUC
mutation        = SurGen RAS macro-OVR AUC
survival        = mean(LEOPARD BCR c-index, CPTAC-PDA OS c-index)

predictive_mean = mean(classification, segmentation, progression,
                       mutation, survival)
robustness_quality = mean over PathoROB subsets of
                     (robustness index + biological balanced accuracy) / 2

mean_probe_score = 0.95 * predictive_mean + 0.05 * robustness_quality
```

`mean_probe_score` and `final_probe_score` are identical public aliases. Each
predictive family contributes 19% of the final score and robustness contributes
5%. Classification's datasets, heads, and hyperparameter cells remain visible
for diagnosis but do not become extra top-level families.

## Fixed suite

| Family | Datasets | Scored metric | Protocol details |
|---|---|---|---|
| Classification | BACH, BRACS, BreaKHis, CRC, ESCA, MHIST, PCam, SPIDER breast/colorectal/skin/thorax, WILDS | macro-F1 | [classification.md](classification.md) |
| Segmentation | PanNuke, SegPath epithelial, SegPath lymphocytes | THUNDER weighted per-image macro-F1 | [segmentation.md](segmentation.md) |
| Progression | UCLA Lung | macro-OVR AUC | [slide_probes.md](slide_probes.md) |
| Mutation | SurGen RAS | macro-OVR AUC | [slide_probes.md](slide_probes.md) |
| Survival | LEOPARD BCR, CPTAC-PDA OS | Harrell c-index | [slide_probes.md](slide_probes.md) |
| Robustness | PathoROB Camelyon, Tolkach ESCA | quality-adjusted robustness | [pathorob.md](pathorob.md) |

The complete fixed suite is mandatory. `prepare_probe_state()` rejects partial,
reordered, added, or substituted task lists.

## Data boundary

[The THUNDER manifest](thunder_v2.json) is the only classification and
segmentation manifest used at runtime. Every dataset entry has exactly `root`,
`train`, and `val`; there is no `test` key. The four slide manifests contain
only development records. UCLA, SurGen, and CPTAC-PDA use the original
PathoBench fold-0 training pool, and LEOPARD uses public challenge training
labels. Their original test records are absent.

The downloadable evaluation snapshot is
[`medarc/nanopath-evals`](https://huggingface.co/datasets/medarc/nanopath-evals)
pinned in `prepare.py` to revision
`635a83330b0dc2917d7524644f11b04188a63e53`. It is about 192 GiB and contains
only the selected development assets. HEST is absent. No CPTAC classification
task is present. CPTAC appears only as the pre-existing CPTAC-PDA survival
development probe. PanNuke Fold3 and the unused TCGA center in Tolkach ESCA are
absent.

On the MedARC cluster, evaluation reads canonical shared roots under `/data`.
A shared upstream root may contain unrelated official assets, but `probe.py`
opens only the paths named by these manifests. A fresh download of the pinned
nanopath snapshot contains no such unrelated assets. `prepare.py` verifies the
snapshot metadata, the explicit `contains_official_test_records: false`
contract, manifest hashes, and every referenced file before a run starts.

TCGA-pretraining overlap is handled explicitly:

- CCRCC, TCGA CRC-MSI, TCGA-TILs, TCGA-Uniform, and OCELOT are excluded because
  their evaluation images are explicitly TCGA.
- ESCA training may contain TCGA images, but its scored validation selection is
  UKK-only.
- MoNuSAC is excluded because the released cohort is TCGA-derived.
- PanNuke mixes TCGA and local-hospital material and does not expose reliable
  per-image source provenance. It is the one accepted mixed-source exception;
  this limitation is documented in [segmentation.md](segmentation.md).

## Frozen-backbone contract

The benchmark measures frozen representations. Classification and slide probes
consume `model.probe_features()`, allowing a recipe to define bounded test-time
feature aggregation. Segmentation consumes all non-register patch channels from
`model.encode_image()`. If a model emits an expanded spatial grid, it is
area-pooled back to its native patch grid before the shared decoder; feature
channels are not discarded. PathoROB intentionally bypasses
`probe_features()` and uses its fixed published-style CLS-plus-mean-patch
adapter so model-specific aggregation cannot alter the robustness protocol.

Encoder inference uses fp16 autocast and caches classification/slide embeddings
as float32. Segmentation patch vectors are cached as per-vector signed int8 plus
fp16 scales to fit the one-GPU runtime and memory envelope. The complete suite
must finish in less than 1,500 seconds on one 80 GB H100. See
[validation.md](validation.md) for parity, determinism, runtime, rank-fidelity,
and null-model evidence.

## Files

| File | Role |
|---|---|
| [THUNDER manifest](thunder_v2.json) | Exact classification and segmentation train/validation records |
| [`ucla_lung.json`](ucla_lung.json) | UCLA fold-0 development-pool slide labels |
| [`surgen.json`](surgen.json) | SurGen fold-0 development-pool slide labels |
| [`leopard_bcr.json`](leopard_bcr.json) | LEOPARD public-training cohort and fixed folds |
| [`cptac_pda_os.json`](cptac_pda_os.json) | CPTAC-PDA fold-0 development-pool survival labels and fixed folds |
| [Proxy-fidelity data](proxy_fidelity_v2.csv) | Frozen 12- and 18-model proxy/official comparison values |
| [Random-feature audit](random_dinov2_s_v2.csv) | Ten-seed exact-suite randomized-backbone audit |
| [classification.md](classification.md) | Dataset provenance, sampling, head math, and THUNDER deviations |
| [segmentation.md](segmentation.md) | Source boundary, decoder, loss, metric, and PanNuke caveat |
| [slide_probes.md](slide_probes.md) | Tile caching, pooling, folds, AUROC, and survival protocols |
| [pathorob.md](pathorob.md) | Fixed adapter, neighbor construction, and quality correction |
| [validation.md](validation.md) | Implementation parity, runtime, null checks, and official-suite fidelity |

## Interpretation

The final scalar is a hill-climbing signal: it rewards improvements shared
across five predictive families while preventing robustness from dominating.
It is strongest as a predictor of model ordering, not as a calibrated estimate
of an official score. A difference should be interpreted alongside per-family,
per-dataset, per-head, fold-variance, raw robustness, Jaccard, and timing fields.
Small differences remain susceptible to training-seed and probe noise. Public
leader promotion therefore requires the candidate's three-run mean to exceed
the incumbent's stored three-run mean by the fixed margin of 0.004. The
discovery run is excluded.

Official THUNDER, HEST, and CPTAC results were consulted only after the
benchmark, manifests, and scalar were frozen. They are release-validation
evidence, never inputs to a run and never weights or dataset-selection targets.
