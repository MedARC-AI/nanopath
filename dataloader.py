# TCGA tiles from memory-mapped, uncompressed Arrow IPC files.
# Shards retain original JPEG bytes, paths, and cached tissue fractions.
# Readers are opened lazily in each worker.
#
# Patients (not tiles) are hashed by TCGA barcode and the bottom `val_fraction`
# of the hash space is held out from training; train.py instantiates the dataset
# twice (`is_train=True` for the training loop, `is_train=False` for the
# lightweight DINO/iBOT/KDE validation pass), so the held-out patient slice
# stays cleanly out-of-distribution from optimization.
#
# Each view uses PIL crop/resize/flips, then optional HED jitter, color jitter,
# grayscale/blur, and normalization.
#
# This file is the *pretraining* input pipeline only. The downstream probes
# (probe.py) do not import anything from here.

from bisect import bisect_right
from collections import OrderedDict
import hashlib
import io
import os
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


HED_FROM_RGB = torch.tensor(
    [
        [1.87798274, -1.00767869, -0.55611582],
        [-0.06590806, 1.13473037, -0.1355218],
        [-0.60190736, -0.48041419, 1.57358807],
    ],
    dtype=torch.float32,
)
RGB_FROM_HED = torch.tensor(
    [
        [0.65, 0.7, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78],
    ],
    dtype=torch.float32,
)
LOG_1E6 = float(np.log(1e-6))
TILE_SIZE = 224


# Patients (not tiles) are the split unit so train/val never share a case.
def patient_in_val(patient_id, seed, val_fraction):
    key = f"{seed}:{patient_id}".encode()
    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") / 2**64
    return value < float(val_fraction)


# Path entries start with the SVS stem (TCGA-XX-XXXX-...); the first three dash parts are the patient barcode.
def patient_id_from_relpath(rel):
    return "-".join(rel.split("/", 1)[0].split("-")[:3])


# Lightweight stain-space jitter; this is the stain augmentation hook for pretraining tiles.
class HEDJitter(nn.Module):
    # Store conversion matrices as buffers so transforms move with the module dtype/device if needed.
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
        self.register_buffer("hed_from_rgb", HED_FROM_RGB)
        self.register_buffer("rgb_from_hed", RGB_FROM_HED)

    # Perturb HED channels, then convert back to RGB while the crop is still in [0, 1].
    def forward(self, x):
        rgb = x.movedim(-3, -1).clamp_min(1e-6)
        hed = (torch.log(rgb) / LOG_1E6) @ self.hed_from_rgb.to(dtype=x.dtype)
        hed = hed.clamp_min(0.0)
        shape = (*x.shape[:-3], 1, 1, 3)
        shift = torch.randn(shape, dtype=x.dtype, device=x.device) * self.sigma
        scale = 1.0 + torch.randn(shape, dtype=x.dtype, device=x.device) * self.sigma
        hed = hed * scale + shift
        log_rgb = -(hed * (-LOG_1E6)) @ self.rgb_from_hed.to(dtype=x.dtype)
        return torch.exp(log_rgb).clamp_(0.0, 1.0).movedim(-1, -3)


class GPUAugment(nn.Module):
    def __init__(self, data):
        super().__init__()
        self.hed = HEDJitter(data["hed_jitter"]) if data["hed_jitter"] > 0 else nn.Identity()
        for name, value in [("mean", data["mean"]), ("std", data["std"]),
                            ("jitter", [data["color_jitter"], data["color_jitter"], data["color_jitter_saturation"]])]:
            self.register_buffer(name, torch.tensor(value).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, views):
        import kornia as K
        shape = views.shape
        x = self.hed(views.flatten(0, 1).float() / 255)
        n = x.shape[0]
        factors = 1 + (torch.rand(n, 3, 1, 1, device=x.device) * 2 - 1) * self.jitter
        order = torch.rand(n, 3, device=x.device).argsort(1)
        # Kornia's high-level jitter shares order; functions preserve per-view orders.
        for position in range(3):
            brightness = K.enhance.adjust_brightness_accumulative(x, factors[:, 0, 0, 0])
            contrast = K.enhance.adjust_contrast_with_mean_subtraction(x, factors[:, 1, 0, 0])
            saturation = K.enhance.adjust_saturation_with_gray_subtraction(x, factors[:, 2, 0, 0])
            op = order[:, position, None, None, None]
            x = torch.where(op == 0, brightness, torch.where(op == 1, contrast, saturation))
        x = torch.where(torch.rand(n, 1, 1, 1, device=x.device) < 0.1, K.color.rgb_to_grayscale(x), x)
        sigma = (0.1 + torch.rand(n, 1, device=x.device) * 1.7).expand(-1, 2)
        blurred = K.filters.gaussian_blur2d(x, (9, 9), sigma, border_type="reflect", separable=True)
        x = torch.where(torch.rand(n, 1, 1, 1, device=x.device) < 0.35, blurred, x)
        return ((x - self.mean) / self.std).reshape(shape)


# Map-style TCGA tile dataset that emits global/local multi-view stacks for train.py.
class TCGATileDataset(Dataset):
    # Glob shards, build a (shard_idx, row_in_shard) index over the requested patient
    # split, and configure augmentations. `is_train=True` keeps the (1 - val_fraction)
    # majority of patient ids; `is_train=False` keeps the held-out `val_fraction` slice.
    def __init__(self, cfg, is_train=True):
        data = cfg["data"]
        train = cfg["train"]
        self.tissue_thresh = float(data["tissue_thresh"]) if is_train else 0.0
        dataset_dir = Path(os.path.expandvars(data["dataset_dir"]))
        self.shards = sorted(dataset_dir.glob("shard-*.arrow"))
        if not self.shards:
            raise FileNotFoundError(f"No Arrow shards (shard-*.arrow) under {dataset_dir}")
        # Lazy IPC handles, opened on first __getitem__ in each worker
        # so fork-children own their own file positions.
        # Each worker retains at most 256 readers and one batch per reader.
        self._readers = OrderedDict()
        self._offsets = []
        # Pull paths and cached fractions once to build the unchanged split index;
        # the JPEG bytes column stays on disk until __getitem__.
        in_split_shard = []
        in_split_row = []
        in_split_tissue = []
        for shard_idx, shard_path in enumerate(self.shards):
            with pa.memory_map(str(shard_path), "r") as source:
                schema = pa.ipc.open_file(source).schema
                fields = [schema.get_field_index(name) for name in ("path", "tissue_fraction")]
                table = pa.ipc.open_file(source, options=pa.ipc.IpcReadOptions(included_fields=fields)).read_all()
            # Prefix sums handle short final batches and nonuniform batch sizes.
            self._offsets.append(np.cumsum([0, *[len(c) for c in table["path"].chunks]], dtype=np.int64))
            paths = table["path"].to_pylist()
            fractions = table["tissue_fraction"].to_numpy()
            for row_idx, p in enumerate(paths):
                # XOR with is_train: training keeps tiles where patient_in_val is False,
                # validation keeps the complement.
                if patient_in_val(patient_id_from_relpath(p), data["split_seed"], data["val_fraction"]) != is_train:
                    in_split_shard.append(shard_idx)
                    in_split_row.append(row_idx)
                    in_split_tissue.append(fractions[row_idx])
        if not in_split_shard:
            raise ValueError(f"no {'train' if is_train else 'val'} tiles found in {dataset_dir}; check val_fraction={data['val_fraction']}")
        # Split indices and fractions (~48 MB for 4M tiles) stay shared across fork-workers.
        self.shard_of = np.asarray(in_split_shard, dtype=np.int32)
        self.row_of = np.asarray(in_split_row, dtype=np.int32)
        self.tissue_fraction = np.asarray(in_split_tissue, dtype=np.float32)
        mean, std = data["mean"], data["std"]
        self.global_views = int(train["global_views"])
        self.local_views = int(train["local_views"])
        # Global and local views differ only in crop scale/size; the stochastic tail is shared.
        augment = [
            v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), v2.ToImage(),
        ]
        if not train["gpu_augment"]:
            augment += [
                v2.ToDtype(torch.float32, scale=True),
                *([HEDJitter(data["hed_jitter"])] if data["hed_jitter"] > 0 else []),
                v2.ColorJitter(data["color_jitter"], data["color_jitter"], data["color_jitter_saturation"], 0.0),
                v2.RandomGrayscale(p=0.1),
                v2.RandomApply([v2.GaussianBlur(9, sigma=(0.1, 1.8))], p=0.35),
                v2.Normalize(mean=mean, std=std),
            ]
        self.global_aug = v2.Compose([v2.RandomResizedCrop(train["global_size"], scale=tuple(data["global_crop_scale"]), antialias=True), *augment])
        self.local_aug = v2.Compose([v2.RandomResizedCrop(train["local_size"], scale=tuple(data["local_crop_scale"]), antialias=True), *augment])

    # Dataset length is the number of tiles in this train/val split.
    def __len__(self):
        return int(self.shard_of.shape[0])

    # Resample from cached fractions, then read and augment one accepted JPEG.
    def __getitem__(self, idx):
        idx = int(idx)
        # Reject from metadata before IO, preserving every resampling draw and limit.
        for _ in range(1000):
            if self.tissue_thresh <= 0 or float(self.tissue_fraction[idx]) >= self.tissue_thresh:
                break
            idx = random.randint(0, self.shard_of.shape[0] - 1)
        else:
            raise RuntimeError(f"no tile met tissue_thresh={self.tissue_thresh} after 1000 samples")
        shard_idx = int(self.shard_of[idx])
        row_idx = int(self.row_of[idx])
        entry = self._readers.get(shard_idx)
        if entry is None:
            reader = pa.ipc.open_file(pa.memory_map(str(self.shards[shard_idx]), "r"))
            entry = [reader, -1, None]
            self._readers[shard_idx] = entry
            if len(self._readers) > 256:
                self._readers.popitem(last=False)
        self._readers.move_to_end(shard_idx)
        batch_idx = bisect_right(self._offsets[shard_idx], row_idx) - 1
        row_in_batch = row_idx - self._offsets[shard_idx][batch_idx]
        if entry[1] != batch_idx:
            entry[1:] = [batch_idx, entry[0].get_batch(batch_idx)]
        table = entry[2]
        rel = table["path"][row_in_batch].as_py()
        jpeg_bytes = table["jpeg"][row_in_batch].as_py()
        with Image.open(io.BytesIO(jpeg_bytes)) as img:
            tile = img.convert("RGB")
        slide_stem = rel.split("/", 1)[0]
        patient_id = patient_id_from_relpath(rel)
        slide_key = int.from_bytes(hashlib.blake2b(slide_stem.encode(), digest_size=8).digest(), "big") & 0x7FFFFFFFFFFFFFFF
        patient_key = int.from_bytes(hashlib.blake2b(patient_id.encode(), digest_size=8).digest(), "big") & 0x7FFFFFFFFFFFFFFF
        # Augmentations are stochastic per view; reproducibility comes from worker seeds.
        global_views = torch.stack([self.global_aug(tile) for _ in range(self.global_views)])
        local_views = torch.stack([self.local_aug(tile) for _ in range(self.local_views)])
        return {
            "global_views": global_views,
            "local_views": local_views,
            "sample_idx": torch.tensor(int(idx), dtype=torch.int64),
            "slide_id": torch.tensor(slide_key, dtype=torch.int64),
            "patient_id": torch.tensor(patient_key, dtype=torch.int64),
        }
