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
| THUNDER classification | BreaKHis, MHIST, PCam, CCRCC, TCGA-Uniform, SPIDER-Skin | macro-F1 | Adam linear probe, cosine KNN, and 1,000 deterministic 16-shot SimpleShot draws |
| THUNDER segmentation | PanNuke, OCELOT, SegPath epithelial, SegPath lymphocytes | per-image macro-F1 | frozen dense tokens plus THUNDER MaskTransformer and Dice loss |
| Progression | UCLA Lung | macro-OVR AUC | raw-feature `LogisticRegression(C=0.5, class_weight="balanced")` |
| Mutation | SurGen RAS | macro-OVR AUC | raw-feature `LogisticRegression(C=0.5, class_weight="balanced")` |
| Survival | LEOPARD BCR, CPTAC-PDA OS | c-index | raw-feature CoxNet at alpha 0.01, 0.02, and 0.07, `l1_ratio=0.5` |
| Robustness | PathoROB camelyon, tolkach_esca | quality-adjusted robustness | published fixed-k robustness index and biological-class balanced KNN accuracy |

Classification and segmentation selections are frozen in [thunder_v2.json](thunder_v2.json). The manifest has exactly `train` and `val` records. Full official BreaKHis and MHIST splits are retained; the other classification caps are 3072/768 PCam, 4096/1024 CCRCC, 8192/2048 TCGA-Uniform, and 4096/1024 SPIDER-Skin. Each segmentation task uses 1024/256 examples. Seed 1337 controls capped stratification and every inner split.

SPIDER-Skin's official slide-disjoint validation split contains only one example for one biological class. NanoPath retains that example rather than leaking a training slide into validation; the at-least-16 rule applies to training support pools, where it is required by SimpleShot. Every validation class remains represented and the complete test split stays held out.

Linear hyperparameters and the stopping epoch are selected on an inner split of the selected training data, then refit on all selected training data before validation is scored once. KNN selects `k` on the same inner split. Segmentation likewise selects its epoch on an inner training split, refits, and scores validation once. PanNuke/OCELOT allow at most 30 epochs; SegPath epithelial/lymphocytes use 9/21.

Runtime roots are the canonical shared locations declared in `configs/main.yaml`: `/data/thunder-data`, `/data/surgen`, `/data/leopard_bcr`, `/data/CPTAC-PDA`, and `/data/pathorob`. `prepare.py download=False` verifies every selected path, split disjointness, and classification class counts.

## Promotion discipline

Code, manifest, and formula are frozen before comparing against official aggregate results. Post-freeze study artifacts belong under `/data/paul/nanopath/probe-v2-study`. Existing leaderboard rows are protocol-v1 legacy and are not comparable to protocol v2; reference tables are repopulated only from complete v2 runs.

The versioned `null_plots/` images are protocol-v1 historical artifacts only; v2 neither reads nor regenerates them.
