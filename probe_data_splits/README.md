# probe_data_splits

Cached classification data splits for the bracs / break_his / mhist probes.
These are deterministic outputs of the dataset directory layout + Thunder's
`generate-data-splits` seed; we ship them here so a fresh checkout can run
probes immediately after `python prepare.py <config>` without needing
Thunder installed.

Each JSON has the shape:
```
{"train": {"images": ["relative/path.png", ...], "labels": [int, ...]},
 "val":   {...}, "test": {...}, "train_few_shot": {...}}
```

Image paths are relative to the corresponding `DATASET_ROOTS[dataset]` in
`probe.py`. Only `train` and `val` are used by the inline cls probes.

pcam doesn't have a JSON here — it's subsampled inline at probe time
(`probe.py:ClassificationDataset` for `dataset == "pcam"`).
pannuke is read directly from `Fold1`/`Fold2` npy files; no JSON needed.
pathorob is read directly from `pathorob/{camelyon,tolkach_esca}/data/*.parquet`
(see `probe.py:inline_pathorob`); no JSON needed.
