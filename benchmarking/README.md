# NanoPath probe protocol v2

NanoPath v2 is a fast, fully train/validation-derived proxy for held-out THUNDER, HEST, and CPTAC evaluation. HEST and new CPTAC tasks are not part of NanoPath. Official test samples are never listed, instantiated, or opened.

## Final score

All values remain on their original 0–1 scales:

```text
classification = mean(dataset × {linear, KNN, 16-shot SimpleShot} macro-F1)
predictive_mean = mean(classification, segmentation, progression, mutation, survival)
robustness_quality = mean((PathoROB robustness index + biological balanced accuracy) / 2)
mean_probe_score = 0.95 * predictive_mean + 0.05 * robustness_quality
```

Each predictive family therefore contributes 19% and robustness contributes 5%. `mean_probe_score` and the summary alias `final_probe_score` are the only protocol-v2 scalar; protocol v1 is not computed.

## Fixed suite

| Family | Datasets | Scored metric | Protocol |
|---|---|---|---|
| THUNDER classification | all 16 official aggregate tasks | macro-F1 | Adam linear probe, cosine KNN, and 1,000 deterministic 16-shot SimpleShot draws |
| THUNDER segmentation | PanNuke, OCELOT, SegPath epithelial, SegPath lymphocytes | per-image macro-F1 | frozen dense tokens plus THUNDER MaskTransformer and Dice loss |
| Progression | UCLA Lung | macro-OVR AUC | raw-feature `LogisticRegression(C=0.5, class_weight="balanced")` |
| Mutation | SurGen RAS | macro-OVR AUC | raw-feature `LogisticRegression(C=0.5, class_weight="balanced")` |
| Survival | LEOPARD BCR, CPTAC-PDA OS | c-index | raw-feature CoxNet at alpha 0.01, 0.02, and 0.07, `l1_ratio=0.5` |
| Robustness | PathoROB camelyon, tolkach_esca | quality-adjusted robustness | published fixed-k robustness index and biological-class balanced KNN accuracy |

Classification and segmentation selections are frozen in [thunder_v2.json](thunder_v2.json). The manifest has exactly `train` and `val` records. It contains the same 16 equally weighted tasks as THUNDER's published classification aggregate: BACH, BRACS, BreaKHis, CCRCC, CRC, ESCA, MHIST, PCam, SPIDER breast/colorectal/skin/thorax, TCGA CRC-MSI/TILs/Uniform, and WILDS. BACH, BreaKHis, and MHIST retain their complete official train/validation splits. Every larger task is capped at 1024/256 with seed-1337 proportional class stratification; naturally smaller splits remain complete. Every training class has at least 16 examples and every validation class is represented.

The segmentation panel uses the exact four current THUNDER datasets and split semantics. PanNuke keeps all 2656/2523 train/validation patches and OCELOT all 6400/2178 crops. SegPath epithelial keeps four source-balanced training crops and two validation crops per selected source (32768/4518); lymphocytes keeps two per source (20906/2164). Source images never cross splits. One crop per lymphocyte source was insufficient in validation diagnostics, so the frozen panel retains two.

Classification mirrors THUNDER's visible-validation protocol: fp16 frozen embeddings and the cached loader's seed-0 minibatch stream feed nine Adam heads using `lr={1e-3,1e-4,1e-5}` × `weight_decay={0,1e-3,1e-4}` for at most 200 epochs. KNN selects from the published `k` grid, and SimpleShot reconstructs THUNDER's seed-0 support-index stream before taking the 16-shot draws. The unweighted mean of all 16 × 3 macro-F1 cells is one classification family. The official test split is absent from the manifest and runtime.

Segmentation retains THUNDER's two-layer MaskTransformer, Dice objective, Adam, batch 64, official per-image macro-F1/Jaccard implementation, and 30/30/9/21 epoch limits. Dense tokens are cached with one signed-int8 vector and one fp16 scale per token. For the runtime gate the decoder width is 192 instead of 768. The pre-frozen task schedules are PanNuke `1e-3/1e-4`, OCELOT `1e-4/0`, epithelial `1e-4/1e-3`, and lymphocytes `1e-3/1e-4` for LR/weight decay. As an efficient analogue of THUNDER's validation checkpoint selection, each epoch is compared by Dice loss on 256 evenly spaced validation examples before the selected checkpoint is scored on the complete validation subset.

All frozen feature extraction uses fp16, matching the official evaluation drivers, with image batches of 256. Runtime roots are the canonical shared locations declared in `configs/main.yaml`: `/data/thunder-data`, `/data/ucla-lung`, `/data/surgen`, `/data/leopard_bcr`, `/data/CPTAC-PDA`, and `/data/pathorob`. `prepare.py download=False` verifies every selected path, source-disjoint splits, and classification class counts.

## Promotion discipline

Code, manifest, and formula are frozen before comparing against official aggregate results. Post-freeze study artifacts belong under `/data/paul/nanopath/probe-v2-study`. Existing leaderboard rows are protocol-v1 legacy and are not comparable to protocol v2; reference tables are repopulated only from complete v2 runs.
