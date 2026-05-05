# probe_data_splits

Cached downstream train/val splits that are small enough to ship with the repo.

Tile classification splits for bracs / break_his / mhist are deterministic
outputs of the dataset directory layout plus Thunder's `generate-data-splits`
seed. They use:
```
{"train": {"images": ["relative/path.png", ...], "labels": [int, ...]},
 "val":   {...}, "test": {...}, "train_few_shot": {...}}
```

Image paths are relative to the corresponding `DATASET_ROOTS[dataset]` in
`probe.py`. Only `train` and `val` are used by the inline cls probes.

Slide-level JSONs use slide ids instead of image paths. `her2.json` and
`ucla_lung.json` are mean-pooled per slide before linear / KNN / few-shot;
`surgen.json` is the SurGen extended-RAS AUROC split. `crc_survival.json`
carries slide ids plus `(event, days)` labels for the PFS_VALENTINO c-index
probe. Test folds in these JSONs are provenance only and are not read by
`probe.py`. `her2_pathobench.tsv` is the upstream PathoBench split file used to
derive `her2.json`.

pcam doesn't have a JSON here — it's subsampled inline at probe time
(`probe.py:ClassificationDataset` for `dataset == "pcam"`).
pannuke is read directly from `Fold1`/`Fold2` npy files; monusac and consep
are split deterministically from their official train folds at probe time; and
pathorob is read directly from `pathorob/{camelyon,tolkach_esca}/data/*.parquet`.
