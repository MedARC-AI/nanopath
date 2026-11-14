# nanopath-evals

This directory describes and audits nanopath's fixed downstream benchmark, termed `nanopath-evals`, that completes a diverse suite of tile-level and slide-level evaluations in approximately 20 minutes on a single H100 GPU. This benchmark is a fast development-data proxy for performance on official THUNDER, HEST, and CPTAC evaluations; note that no test samples from these official benchmarks are touched by `nanopath-evals`.

The executable definition is [`probe.py`](../probe.py) plus the five checked-in JSON manifests in this directory. The prose here explains that definition; if the code, manifests, and documentation ever disagree, the release is not valid as a nanopath model. Labless locks `probe.py`, this entire directory, and the probe configuration for comparable public runs.

## Score definition

All components enter the scalar on 0–1 scales:

```text
classification = mean over 12 datasets of
                 mean(linear-grid mean F1, KNN-grid mean F1, SimpleShot F1)

segmentation    = mean(PanNuke F1, SegPath epithelial F1,
                       SegPath lymphocyte F1)
progression     = UCLA Lung macro-OVR AUC
mutation        = SurGen RAS macro-OVR AUC
survival        = mean(LEOPARD BCR c-index, CPTAC-PDA OS c-index)

cohort_croma  = median over tiles of (mean OS distance - mean SO distance)
                / (mean OS distance + mean SO distance)
robustness    = mean over PathoROB cohorts of (1 + cohort_croma) / 2

final_score = 0.40 * classification + 0.20 * segmentation
            + 0.10 * progression + 0.10 * mutation
            + 0.05 * survival + 0.15 * robustness
```

Classification, segmentation, progression, mutation, survival, and robustness contribute 40%, 20%, 10%, 10%, 5%, and 15%, respectively.

## Fixed suite

| Family | Datasets | Scored metric | Protocol details |
|---|---|---|---|
| Classification | BACH, BRACS, BreaKHis, CRC, ESCA, MHIST, PCam, SPIDER breast/colorectal/skin/thorax, WILDS | macro-F1 | [classification.md](classification.md) |
| Segmentation | PanNuke, SegPath epithelial, SegPath lymphocytes | THUNDER weighted per-image macro-F1 | [segmentation.md](segmentation.md) |
| Progression | UCLA Lung | macro-OVR AUC | [slide_probes.md](slide_probes.md) |
| Mutation | SurGen RAS | macro-OVR AUC | [slide_probes.md](slide_probes.md) |
| Survival | LEOPARD BCR, CPTAC-PDA OS | Harrell c-index | [slide_probes.md](slide_probes.md) |
| Robustness | PathoROB Camelyon, Tolkach ESCA | CRoMa | [pathorob.md](pathorob.md) |

The complete fixed suite is mandatory. `prepare_probe_state()` rejects partial, reordered, added, or substituted task lists.

## Data boundary

[The THUNDER manifest](thunder_v2.json) is the only classification and segmentation manifest used at runtime. Every dataset entry has exactly `root`, `train`, and `val`; there is no `test` key.

The downloadable evaluation snapshot is [`medarc/nanopath-evals`](https://huggingface.co/datasets/medarc/nanopath-evals) pinned in `prepare.py` to revision `5c8f7848298fa55e09e0bcc58a90a8e9b1c8d426`. It is about 192 GiB and contains only the selected development assets. If you are on our MedARC cluster, you do not need to download this snapshot because contents are already under `/data`.

We try to reduce overlap between our pretraining dataset (TCGA) and downstream data distributions:

- CCRCC, TCGA CRC-MSI, TCGA-TILs, TCGA-Uniform, and OCELOT are excluded because their evaluation images are explicitly TCGA.
- We still include ESCA because while its training samples contain some TCGA images, its validation subset is UKK-only.
- We still include PanNuke because it mixes TCGA and local hospital slides without specifying which slides belong to which source (this limitation is documented in [segmentation.md](segmentation.md)).

## Frozen-backbone contract

`nanopath-evals` operates on frozen representations from one eval-mode forward pass per tile. Protocol v3 fixes the readout of nanopath models: classification, slide and CRoMa probes consume the final-layer CLS token `model(x)["cls"]`, and segmentation consumes the final-layer patch tokens `model(x)["patches"]` on the native patch grid. Before probing, `probe.py` runs a fixed 4-tile batch and asserts that every weight-carrying module runs once per pass on exactly those tiles, that both outputs are `embed_dim` wide, and that eval-mode output equals train-mode output. Frozen baselines keep their published `probe_features()` / `encode_image()` adapters, and CRoMa reads `probe_features()` like the other pooled probes.

Encoder inference uses fp16 autocast and caches classification/slide embeddings as float32. Segmentation patch vectors are cached as per-vector signed int8 plus fp16 scales. See [validation.md](validation.md) for release timing and validation evidence.

## Files

| File | Role |
|---|---|
| [THUNDER manifest](thunder_v2.json) | Exact classification and segmentation train/validation records |
| [`ucla_lung.json`](ucla_lung.json) | UCLA fold-0 development-pool slide labels |
| [`surgen.json`](surgen.json) | SurGen fold-0 development-pool slide labels |
| [`leopard_bcr.json`](leopard_bcr.json) | LEOPARD public-training cohort and fixed folds |
| [`cptac_pda_os.json`](cptac_pda_os.json) | CPTAC-PDA fold-0 development-pool survival labels and fixed folds |
| [Proxy-fidelity data](proxy_fidelity_v2.csv) | Frozen proxy/official comparison values for 24 models |
| [Random-feature audit](random_dinov2_s_v2.csv) | Ten-seed predictive and CRoMa audit |
| [classification.md](classification.md) | Dataset provenance, sampling, head math, and THUNDER deviations |
| [segmentation.md](segmentation.md) | Source boundary, decoder, loss, metric, and PanNuke caveat |
| [slide_probes.md](slide_probes.md) | Tile caching, pooling, folds, AUROC, and survival protocols |
| [pathorob.md](pathorob.md) | Readout and CRoMa neighbor-margin computation |
| [validation.md](validation.md) | Implementation parity, runtime, null checks, and official-suite fidelity |
