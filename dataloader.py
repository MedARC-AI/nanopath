# TCGA pretraining input pipeline. Reads the OpenMidnight-style sample list at
# data.sample_list (one WSI patch per line: slide_path x y level), assigns each
# patient deterministically to train or val by hashing the patient id, caches
# byte offsets per split, opens slides lazily with lazyslide (LRU per worker),
# and emits the JEPA global+local view stacks with ColorJitter + optional
# HED jitter. The augmentation stack lives here, not in configs, on purpose.

import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import lazyslide as zs
import numpy as np
import torch
import torch.nn as nn
from numpy.lib.format import open_memmap
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
SAMPLE_LIST_PATCH_SIZE = 224


# Assign patients, not tiles, to splits so train/val do not leak slides from the same case.
def patient_in_val(patient_id, seed, val_fraction):
    key = f"{seed}:{patient_id}".encode()
    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") / 2**64
    return value < float(val_fraction)


# TCGA slide filenames begin with the patient barcode; this is the split key.
def patient_id_from_slide_path(slide_path):
    stem = Path(slide_path).name.split(".", 1)[0]
    parts = stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"could not derive TCGA patient id from slide path: {slide_path}")
    return "-".join(parts[:3])


# Parse only the slide path from each sample-list row; x/y/level stay cold until __getitem__.
def sample_list_rows(sample_list):
    with sample_list.open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            line = line.rstrip(b"\r\n")
            if line:
                yield offset, line.decode("utf-8").rsplit(" ", 3)[0]


# Map a slide path to the configured split names while hashing at patient level.
def split_from_slide_path(slide_path, data):
    patient_id = patient_id_from_slide_path(slide_path)
    return data["val_split"] if patient_in_val(patient_id, data["split_seed"], data["val_fraction"]) else data["train_split"]


# Build cache paths that change when the sample list, seed, or val fraction changes.
def split_offset_paths(data):
    cache_dir = Path(data["sample_list_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample_list = Path(data["sample_list"]).resolve()
    sample_list_stat = sample_list.stat()
    sample_list_tag = hashlib.blake2b(
        f"{sample_list}:{sample_list_stat.st_size}:{sample_list_stat.st_mtime_ns}".encode(),
        digest_size=8,
    ).hexdigest()
    val_tag = str(float(data["val_fraction"])).replace(".", "p")
    prefix = f"{sample_list.stem}.{sample_list_tag}.seed{int(data['split_seed'])}.val{val_tag}"
    return {
        data["train_split"]: cache_dir / f"{prefix}.{data['train_split']}.offsets.npy",
        data["val_split"]: cache_dir / f"{prefix}.{data['val_split']}.offsets.npy",
    }


# Resolve microns-per-pixel from lazyslide first, then fall back to common OpenSlide properties.
def resolve_slide_mpp(wsi):
    mpp = wsi.properties.mpp
    if mpp is not None:
        mpp = float(mpp)
        if np.isfinite(mpp) and mpp > 0:
            return mpp
    raw = wsi.properties.to_dict()["raw"]
    props = json.loads(raw) if isinstance(raw, str) else raw
    for key in ("openslide.mpp-x", "openslide.mpp-y", "aperio.MPP"):
        value = props.get(key)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value) and value > 0:
            return value
    return float("nan")


# Precompute byte offsets per split so workers can seek directly into huge sample lists.
def prepare_sample_list_offsets(cfg):
    data = cfg["data"]
    sample_list = Path(data["sample_list"])
    if not sample_list.is_file():
        raise FileNotFoundError(f"sample list not found at {sample_list}")
    if float(data["val_fraction"]) <= 0.0 or float(data["val_fraction"]) >= 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {data['val_fraction']}")
    offset_paths = split_offset_paths(data)
    train_offsets_path = offset_paths[data["train_split"]]
    val_offsets_path = offset_paths[data["val_split"]]
    if train_offsets_path.is_file() and val_offsets_path.is_file():
        return offset_paths
    splits = (data["train_split"], data["val_split"])
    # First pass only counts rows so the memmap arrays can be allocated exactly once.
    counts = {split: 0 for split in splits}
    for _, slide_path in sample_list_rows(sample_list):
        counts[split_from_slide_path(slide_path, data)] += 1
    tmp_tag = f".{os.getpid()}.{time.time_ns()}.tmp"
    tmp_paths = {split: Path(f"{offset_paths[split]}{tmp_tag}") for split in splits}
    offsets = {split: open_memmap(tmp_paths[split], mode="w+", dtype=np.uint64, shape=(counts[split],)) for split in splits}
    # Second pass writes byte offsets, leaving tile decoding to __getitem__.
    write_i = {split: 0 for split in splits}
    for offset, slide_path in sample_list_rows(sample_list):
        split = split_from_slide_path(slide_path, data)
        offsets[split][write_i[split]] = offset
        write_i[split] += 1
    if write_i != counts:
        raise ValueError(f"offset count mismatch after writing sample list cache: wrote {write_i}, expected {counts}")
    for split in splits:
        offsets[split].flush()
        tmp_paths[split].replace(offset_paths[split])
    return offset_paths


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
class RandomTCGADataset(Dataset):
    # Configure split offsets, slide-handle cache, and the two augmentation views.
    def __init__(self, cfg, split):
        data = cfg["data"]
        train = cfg["train"]
        self.is_train = split == data["train_split"]
        self.sample_list_path = Path(data["sample_list"])
        if not self.sample_list_path.is_file():
            raise FileNotFoundError(f"sample list not found at {self.sample_list_path}")
        if int(train["global_size"]) > SAMPLE_LIST_PATCH_SIZE:
            raise ValueError(
                f"global_size must be <= {SAMPLE_LIST_PATCH_SIZE} when using sample_dataset_30.txt, got global_size={train['global_size']}"
            )
        offset_paths = split_offset_paths(data)
        offsets_path = offset_paths[split]
        if not offsets_path.is_file():
            raise FileNotFoundError(f"sample list offset cache missing at {offsets_path}; call prepare_sample_list_offsets first")
        self.offsets = np.load(offsets_path, mmap_mode="r")
        self.sample_list_handle = None
        self.max_open_wsi = int(data["max_open_wsi"])
        self.handles = OrderedDict()
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

    # Dataset length is the number of cached offsets for this split.
    def __len__(self):
        return int(self.offsets.shape[0])

    # Load one WSI tile, apply train or deterministic-val augmentations, and return train.py fields.
    def __getitem__(self, idx):
        # Keep one sample-list handle per worker and seek to the cached row offset.
        if self.sample_list_handle is None:
            self.sample_list_handle = self.sample_list_path.open("rb")
        self.sample_list_handle.seek(int(self.offsets[idx]))
        slide_path, x_str, y_str, level_str = self.sample_list_handle.readline().decode("utf-8").rstrip("\r\n").rsplit(" ", 3)
        patient_id = patient_id_from_slide_path(slide_path)
        slide_key = int.from_bytes(hashlib.blake2b(slide_path.encode("utf-8"), digest_size=8).digest(), "big") & 0x7FFFFFFFFFFFFFFF
        patient_key = int.from_bytes(hashlib.blake2b(patient_id.encode("utf-8"), digest_size=8).digest(), "big") & 0x7FFFFFFFFFFFFFFF
        x = int(x_str)
        y = int(y_str)
        level = int(level_str)
        wsi = self.handles.get(slide_path)
        if wsi is None:
            # WSI handles are expensive; lazily open them and keep a small LRU per worker.
            if not Path(slide_path).is_file():
                raise FileNotFoundError(
                    f"TCGA slide not found at {slide_path}. The path comes from {self.sample_list_path}. "
                    "Follow the TCGA data setup in README.md, for example "
                    "`bash download_TCGA.sh /data/TCGA 8`, then set data.sample_list to "
                    "/data/TCGA/sample_dataset_30.txt or place/symlink slides at the paths listed there."
                )
            wsi = zs.open_wsi(slide_path, store=None, attach_thumbnail=False)
            self.handles[slide_path] = wsi
        self.handles.move_to_end(slide_path)
        while len(self.handles) > self.max_open_wsi:
            self.handles.popitem(last=False)[1].close()
        slide_mpp = resolve_slide_mpp(wsi)
        level_downsample = np.asarray(wsi.properties.level_downsample, dtype=np.float64)
        if level < 0 or level >= len(level_downsample):
            raise ValueError(f"invalid level {level} for {slide_path} with {len(level_downsample)} pyramid levels")
        tile = np.asarray(wsi.reader.get_region(x, y, SAMPLE_LIST_PATCH_SIZE, SAMPLE_LIST_PATCH_SIZE, level=level))
        # Normalize the lazyslide/open-slide output to an RGB 224x224 tensor before augmentation.
        if tile.shape[-1] == 4:
            tile = tile[..., :3]
        if tile.shape[0] != SAMPLE_LIST_PATCH_SIZE or tile.shape[1] != SAMPLE_LIST_PATCH_SIZE:
            tile = np.asarray(Image.fromarray(tile).resize((SAMPLE_LIST_PATCH_SIZE, SAMPLE_LIST_PATCH_SIZE), Image.Resampling.BILINEAR))
        tile = self.to_tensor(tile.copy())
        sampled_mpp = slide_mpp * float(level_downsample[level])
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
            "sampled_mpp": torch.tensor(sampled_mpp, dtype=torch.float32),
            "sample_idx": torch.tensor(int(idx), dtype=torch.int64),
            "slide_id": torch.tensor(slide_key, dtype=torch.int64),
            "patient_id": torch.tensor(patient_key, dtype=torch.int64),
        }
