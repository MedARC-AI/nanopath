# This file is the TCGA input path for pretraining.
# It reads the same OpenMidnight-style sample list used to choose TCGA patches,
# builds cached train/val line-offset indices from that list by hashing TCGA patient IDs,
# opens WSIs with lazyslide, and turns each listed patch into the JEPA global/local views.
# The patch source is therefore fixed by `/block/TCGA/sample_dataset_30.txt`, while the
# multi-view augmentation stack remains specific to this repo rather than copying OpenMidnight.

import hashlib
import json
import math
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


def assign_split(patient_id, seed, val_fraction):
    key = f"{seed}:{patient_id}".encode()
    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") / 2**64
    return "val" if value < val_fraction else "train"


def patient_id_from_slide_path(slide_path):
    stem = Path(slide_path).name.split(".", 1)[0]
    parts = stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"could not derive TCGA patient id from slide path: {slide_path}")
    return "-".join(parts[:3])


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


def resolve_slide_mpp(wsi, slide_path):
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
    train_count = 0
    val_count = 0
    with sample_list.open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            line = line.rstrip(b"\r\n")
            if not line:
                continue
            slide_path = line.decode("utf-8").rsplit(" ", 3)[0]
            split = assign_split(patient_id_from_slide_path(slide_path), data["split_seed"], data["val_fraction"])
            if split == data["train_split"]:
                train_count += 1
            elif split == data["val_split"]:
                val_count += 1
            else:
                raise ValueError(f"unexpected split {split} while counting sample list offsets at byte {offset}")
    tmp_tag = f".{os.getpid()}.{time.time_ns()}.tmp"
    train_tmp = Path(f"{train_offsets_path}{tmp_tag}")
    val_tmp = Path(f"{val_offsets_path}{tmp_tag}")
    train_offsets = open_memmap(train_tmp, mode="w+", dtype=np.uint64, shape=(train_count,))
    val_offsets = open_memmap(val_tmp, mode="w+", dtype=np.uint64, shape=(val_count,))
    train_i = 0
    val_i = 0
    with sample_list.open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            line = line.rstrip(b"\r\n")
            if not line:
                continue
            slide_path = line.decode("utf-8").rsplit(" ", 3)[0]
            split = assign_split(patient_id_from_slide_path(slide_path), data["split_seed"], data["val_fraction"])
            if split == data["train_split"]:
                train_offsets[train_i] = offset
                train_i += 1
            elif split == data["val_split"]:
                val_offsets[val_i] = offset
                val_i += 1
            else:
                raise ValueError(f"unexpected split {split} while writing sample list offsets at byte {offset}")
    if train_i != train_count or val_i != val_count:
        raise ValueError(f"offset count mismatch after writing sample list cache: train {train_i}/{train_count}, val {val_i}/{val_count}")
    train_offsets.flush()
    val_offsets.flush()
    train_tmp.replace(train_offsets_path)
    val_tmp.replace(val_offsets_path)
    return offset_paths


class HEDJitter(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
        self.register_buffer("hed_from_rgb", HED_FROM_RGB)
        self.register_buffer("rgb_from_hed", RGB_FROM_HED)

    def forward(self, x):
        rgb = x.permute(1, 2, 0).clamp_min(1e-6)
        hed = (torch.log(rgb) / LOG_1E6) @ self.hed_from_rgb.to(dtype=x.dtype)
        hed = hed.clamp_min(0.0)
        shift = torch.randn((1, 1, 3), dtype=x.dtype) * self.sigma
        scale = 1.0 + torch.randn((1, 1, 3), dtype=x.dtype) * self.sigma
        hed = hed * scale + shift
        log_rgb = -(hed * (-LOG_1E6)) @ self.rgb_from_hed.to(dtype=x.dtype)
        return torch.exp(log_rgb).clamp_(0.0, 1.0).permute(2, 0, 1)


class RandomTCGADataset(Dataset):
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
        worker_budget = max(1, int(train["num_workers"]))
        self.max_open_wsi = max(1, int(math.ceil(float(data["max_open_wsi"]) / float(worker_budget))))
        self.handles = OrderedDict()
        mean, std = data["mean"], data["std"]
        self.global_views = int(train["global_views"])
        self.local_views = int(train["local_views"])
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.hed_jitter = HEDJitter(data["hed_jitter"]) if data["hed_jitter"] > 0 else None
        self.global_aug = v2.Compose(
            [
                v2.RandomResizedCrop(train["global_size"], scale=tuple(data["global_crop_scale"]), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.ColorJitter(data["color_jitter"], data["color_jitter"], data["color_jitter"], 0.0),
                v2.Normalize(mean=mean, std=std),
            ]
        )
        self.local_aug = v2.Compose(
            [
                v2.RandomResizedCrop(train["local_size"], scale=tuple(data["local_crop_scale"]), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.ColorJitter(data["color_jitter"], data["color_jitter"], data["color_jitter"], 0.0),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self):
        return int(self.offsets.shape[0])

    def sample_line(self, idx):
        if self.sample_list_handle is None:
            self.sample_list_handle = self.sample_list_path.open("rb")
        self.sample_list_handle.seek(int(self.offsets[idx]))
        return self.sample_list_handle.readline().decode("utf-8").rstrip("\r\n")

    def __getitem__(self, idx):
        seed = ((idx + 1) * 1103515245 + 12345) & 0x7FFFFFFF
        slide_path, x_str, y_str, level_str = self.sample_line(idx).rsplit(" ", 3)
        patient_id = patient_id_from_slide_path(slide_path)
        slide_key = int.from_bytes(hashlib.blake2b(slide_path.encode("utf-8"), digest_size=8).digest(), "big") & 0x7FFFFFFFFFFFFFFF
        patient_key = int.from_bytes(hashlib.blake2b(patient_id.encode("utf-8"), digest_size=8).digest(), "big") & 0x7FFFFFFFFFFFFFFF
        x = int(x_str)
        y = int(y_str)
        level = int(level_str)
        wsi = self.handles.get(slide_path)
        if wsi is None:
            wsi = zs.open_wsi(slide_path, store=None, attach_thumbnail=False)
            self.handles[slide_path] = wsi
        self.handles.move_to_end(slide_path)
        while len(self.handles) > self.max_open_wsi:
            self.handles.popitem(last=False)[1].close()
        slide_mpp = resolve_slide_mpp(wsi, slide_path)
        level_downsample = np.asarray(wsi.properties.level_downsample, dtype=np.float64)
        if level < 0 or level >= len(level_downsample):
            raise ValueError(f"invalid level {level} for {slide_path} with {len(level_downsample)} pyramid levels")
        tile = wsi.reader.get_region(x, y, SAMPLE_LIST_PATCH_SIZE, SAMPLE_LIST_PATCH_SIZE, level=level)
        tile = np.asarray(tile).copy()
        if tile.shape[-1] == 4:
            tile = tile[..., :3]
        if tile.shape[0] != SAMPLE_LIST_PATCH_SIZE or tile.shape[1] != SAMPLE_LIST_PATCH_SIZE:
            tile = np.asarray(Image.fromarray(tile).resize((SAMPLE_LIST_PATCH_SIZE, SAMPLE_LIST_PATCH_SIZE), Image.Resampling.BILINEAR)).copy()
        tile = self.to_tensor(tile)
        sampled_mpp = slide_mpp * float(level_downsample[level])
        if self.is_train:
            context_tile = self.hed_jitter(tile) if self.hed_jitter is not None else tile
            return {
                "global_views": torch.stack([self.global_aug(context_tile) for _ in range(self.global_views)]),
                "local_views": torch.stack([self.local_aug(context_tile) for _ in range(self.local_views)]),
                "sampled_mpp": torch.tensor(sampled_mpp, dtype=torch.float32),
                "sample_idx": torch.tensor(int(idx), dtype=torch.int64),
                "slide_id": torch.tensor(slide_key, dtype=torch.int64),
                "patient_id": torch.tensor(patient_key, dtype=torch.int64),
            }
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
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
