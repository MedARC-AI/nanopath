# TCGA tile input pipeline backed by Parquet shards. Each shard is a parquet
# file of `{path: string, jpeg: binary}` rows. We open the shards via pyarrow
# directly (NOT `datasets.load_dataset`, which copies into ~/.cache) so the
# 112 GB of shards are mmap'd in place with zero duplication. Random access is
# resolved by per-shard ParquetFile.read_row_group; prepare.py packs each
# shard with PARQUET_ROW_GROUP_SIZE=64 rows/group so reading one row group is
# ~2 MB and __getitem__ is ~2-3 ms incl. JPEG decode.
#
# Patients (not tiles) are split train/val by hashing the TCGA barcode parsed
# from the path.
#
# Augmentation per tile: optional HEDJitter (stain-space color perturbation),
# then train.global_views global crops + train.local_views local crops, each
# chained as RandomResizedCrop -> horizontal flip -> vertical flip ->
# ColorJitter -> Normalize. During validation, we deterministically seed the
# augmentation RNG with the tile's index so every val pass produces the same
# augmented views — that way the val loss curve reflects model changes rather
# than augmentation randomness from one eval to the next.
#
# This file is the *pretraining* input pipeline only. The downstream probes
# (probe.py) do not import anything from here.

import hashlib
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
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

    # Perturb HED channels, then convert back to RGB before the crop/flip/color pipeline.
    def forward(self, x):
        rgb = x.permute(1, 2, 0).clamp_min(1e-6)
        hed = (torch.log(rgb) / LOG_1E6) @ self.hed_from_rgb.to(dtype=x.dtype)
        hed = hed.clamp_min(0.0)
        shift = torch.randn((1, 1, 3), dtype=x.dtype) * self.sigma
        scale = 1.0 + torch.randn((1, 1, 3), dtype=x.dtype) * self.sigma
        hed = hed * scale + shift
        log_rgb = -(hed * (-LOG_1E6)) @ self.rgb_from_hed.to(dtype=x.dtype)
        return torch.exp(log_rgb).clamp_(0.0, 1.0).permute(2, 0, 1)


# Map-style TCGA tile dataset that emits global/local multi-view stacks for train.py.
class TCGATileDataset(Dataset):
    # Glob shards, build a per-split (shard_idx, row_in_shard) index, and configure the two augmentation views.
    def __init__(self, cfg, split):
        data = cfg["data"]
        train = cfg["train"]
        self.is_train = split == data["train_split"]
        dataset_dir = Path(data["dataset_dir"])
        self.shards = sorted(dataset_dir.glob("shard-*.parquet"))
        if not self.shards:
            raise FileNotFoundError(
                f"No parquet shards (shard-*.parquet) under {dataset_dir}. Run "
                f"`python prepare.py {cfg['config_path']} download=True` to fetch them from "
                f"the medarc/nanopath HF dataset before training."
            )
        if int(train["global_size"]) > TILE_SIZE:
            raise ValueError(f"global_size must be <= {TILE_SIZE}, got global_size={train['global_size']}")
        # Lazy ParquetFile handles, opened on first __getitem__ in each worker
        # so fork-children own their own file positions.
        self._readers = [None] * len(self.shards)
        # Pull just the path column from each shard once to build the per-split
        # index; the JPEG bytes column stays on disk until __getitem__.
        in_split_shard = []
        in_split_row = []
        for shard_idx, shard_path in enumerate(self.shards):
            paths = pq.read_table(str(shard_path), columns=["path"], memory_map=True)["path"].to_pylist()
            for row_idx, p in enumerate(paths):
                if patient_in_val(patient_id_from_relpath(p), data["split_seed"], data["val_fraction"]) != self.is_train:
                    in_split_shard.append(shard_idx)
                    in_split_row.append(row_idx)
        if not in_split_shard:
            raise ValueError(f"split '{split}' has zero tiles in {dataset_dir}; check val_fraction={data['val_fraction']}")
        # Two parallel int32 arrays (~32 MB total for 4M tiles) shared COW across DataLoader fork-workers.
        self.shard_of = np.asarray(in_split_shard, dtype=np.int32)
        self.row_of = np.asarray(in_split_row, dtype=np.int32)
        mean, std = data["mean"], data["std"]
        self.global_views = int(train["global_views"])
        self.local_views = int(train["local_views"])
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.hed_jitter = HEDJitter(data["hed_jitter"]) if data["hed_jitter"] > 0 else None
        # Global crops carry the high-context view used by the JEPA consistency objective.
        self.global_aug = v2.Compose(
            [
                v2.RandomResizedCrop(train["global_size"], scale=tuple(data["global_crop_scale"]), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.ColorJitter(data["color_jitter"], data["color_jitter"], data["color_jitter"], 0.0),
                v2.Normalize(mean=mean, std=std),
            ]
        )
        # Local crops force the encoder to align small tissue regions with the global context.
        self.local_aug = v2.Compose(
            [
                v2.RandomResizedCrop(train["local_size"], scale=tuple(data["local_crop_scale"]), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.ColorJitter(data["color_jitter"], data["color_jitter"], data["color_jitter"], 0.0),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    # Dataset length is the number of tiles assigned to this split.
    def __len__(self):
        return int(self.shard_of.shape[0])

    # Read one JPEG row, decode, apply augmentations, and return train.py fields.
    def __getitem__(self, idx):
        shard_idx = int(self.shard_of[idx])
        row_idx = int(self.row_of[idx])
        reader = self._readers[shard_idx]
        if reader is None:
            reader = pq.ParquetFile(str(self.shards[shard_idx]), memory_map=True)
            self._readers[shard_idx] = reader
        # Each shard has uniform-size row groups (PARQUET_ROW_GROUP_SIZE in
        # prepare.py); reading one group is ~2 MB and ~2-3 ms incl. JPEG decode.
        rg_size = reader.metadata.row_group(0).num_rows
        rg_idx = row_idx // rg_size
        row_in_rg = row_idx % rg_size
        table = reader.read_row_group(rg_idx, columns=["path", "jpeg"])
        rel = table["path"][row_in_rg].as_py()
        jpeg_bytes = table["jpeg"][row_in_rg].as_py()
        with Image.open(io.BytesIO(jpeg_bytes)) as img:
            tile = self.to_tensor(img.convert("RGB"))
        slide_stem = rel.split("/", 1)[0]
        patient_id = "-".join(slide_stem.split("-")[:3])
        slide_key = int.from_bytes(hashlib.blake2b(slide_stem.encode(), digest_size=8).digest(), "big") & 0x7FFFFFFFFFFFFFFF
        patient_key = int.from_bytes(hashlib.blake2b(patient_id.encode(), digest_size=8).digest(), "big") & 0x7FFFFFFFFFFFFFFF
        if self.is_train:
            # Train augmentations intentionally remain stochastic; reproducibility comes from worker seeds.
            context_tile = self.hed_jitter(tile) if self.hed_jitter is not None else tile
            global_views = torch.stack([self.global_aug(context_tile) for _ in range(self.global_views)])
            local_views = torch.stack([self.local_aug(context_tile) for _ in range(self.local_views)])
        else:
            # Validation uses deterministic augmentations per index so curves reflect model changes.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(((idx + 1) * 1103515245 + 12345) & 0x7FFFFFFF)
                context_tile = self.hed_jitter(tile) if self.hed_jitter is not None else tile
                global_views = torch.stack([self.global_aug(context_tile) for _ in range(self.global_views)])
                local_views = torch.stack([self.local_aug(context_tile) for _ in range(self.local_views)])
        return {
            "global_views": global_views,
            "local_views": local_views,
            "sample_idx": torch.tensor(int(idx), dtype=torch.int64),
            "slide_id": torch.tensor(slide_key, dtype=torch.int64),
            "patient_id": torch.tensor(patient_key, dtype=torch.int64),
        }
