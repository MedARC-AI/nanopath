# TCGA tile input pipeline for the JPEG dataset produced by preprocessing.py.
# Reads {data.dataset_dir}/manifest.txt (one relative path per line of the
# form `{slide_stem}/{x}_{y}_{level}.jpg`), splits patients (not tiles)
# train/val by hashing the TCGA barcode parsed from each slide stem, and
# emits the JEPA global+local view stacks with ColorJitter + optional HED
# jitter. No WSI decoding at train time; JPEG tiles decode in ~1 ms each so
# throughput is dataloader-bound by IO + augmentation, not WSI seeking.

import hashlib
from pathlib import Path

import numpy as np
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


# Manifest entries start with the SVS stem (TCGA-XX-XXXX-...); the first three dash parts are the patient barcode.
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
    # Filter the manifest to one split and configure the two augmentation views.
    def __init__(self, cfg, split):
        data = cfg["data"]
        train = cfg["train"]
        self.is_train = split == data["train_split"]
        self.dataset_dir = Path(data["dataset_dir"])
        manifest_path = self.dataset_dir / "manifest.txt"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Tile manifest not found at {manifest_path}. Run "
                f"`python preprocessing.py {cfg['config_path']}` to materialize JPEG tiles "
                f"from the SVS sample list before training."
            )
        if int(train["global_size"]) > TILE_SIZE:
            raise ValueError(f"global_size must be <= {TILE_SIZE}, got global_size={train['global_size']}")
        # Patient-level filter; numpy bytes array stays COW-shared across DataLoader fork workers.
        in_split = []
        with manifest_path.open() as f:
            for rel in f:
                rel = rel.rstrip("\n")
                if not rel:
                    continue
                if patient_in_val(patient_id_from_relpath(rel), data["split_seed"], data["val_fraction"]) != self.is_train:
                    in_split.append(rel)
        if len(in_split) == 0:
            raise ValueError(f"split '{split}' has zero tiles; check {manifest_path} and val_fraction={data['val_fraction']}")
        self.paths = np.array(in_split, dtype=np.bytes_)
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
        return int(self.paths.shape[0])

    # Decode one JPEG tile, apply train or deterministic-val augmentations, and return train.py fields.
    def __getitem__(self, idx):
        rel = self.paths[idx].decode("utf-8")
        with Image.open(self.dataset_dir / rel) as img:
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
