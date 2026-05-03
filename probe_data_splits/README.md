# probe_data_splits

Cached probe splits for classification (break_his, mhist), cell-level segmentation
(monusac, consep), and slide-level (chimera, surgen). They are deterministic
outputs of the dataset directory layout + a fixed seed; shipping them here lets a
fresh checkout run probes immediately after `python prepare.py <config>` without
re-running the split generators.

All splits use a four-way train / tune / val / test convention:
```
{"train": {"images": [...], "labels": [...]},
 "tune":  {...}, "val": {...}, "test": {...}}
```
(consep ships an empty `tune` since it's too small to support its own tune split
— it reuses MoNuSAC's chosen lr at fixed wd.)

Image / case paths are relative to the corresponding `DATASET_ROOTS[dataset]` in
`probe.py`. Slide-level splits (chimera, surgen) key by case_id rather than
image path; the slide pool is materialised at runtime by globbing per-case tile
folders under `dataset_roots.{chimera_tiles,surgen_tiles}`.

pcam doesn't have a JSON here — it's subsampled inline at probe time
(`probe.py:ClassificationDataset` for `dataset == "pcam"`).
pannuke is read directly from `Fold1`/`Fold2`/`Fold3` npy files; no JSON needed
(see `PANNUKE_ROTATIONS` in probe.py for the 3-fold rotation logic).
