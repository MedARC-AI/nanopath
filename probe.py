# Inline downstream probes. Protocol v2 weights classification/segmentation/
# progression/mutation/survival/robustness at 30/15/17.5/15/7.5/15 percent.
#
# train.py can snapshot a probe checkpoint at each FLOP milestone to run
# this file as a subprocess (`python probe.py req.json`), whereby training pauses,
# subprocess writes a result JSON, collect_probe_results ingests it back into
# wandb + metrics.jsonl. Inside the subprocess, one loaded frozen backbone serves
# every probe.
#
# Every THUNDER manifest entry is drawn from train/validation only; official
# test paths are intentionally absent from the repository and runtime.

import gc
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from PIL import Image
from timm.layers import trunc_normal_

Image.MAX_IMAGE_PIXELS = None  # Probe ROIs are trusted local pathology images, often >90M pixels.


BENCHMARKING_DIR = Path(__file__).resolve().parent / "benchmarking"
PROBE_PROTOCOL_VERSION = 2
SCORE_WEIGHTS = (
    ("classification_mean_f1", 0.30), ("seg_mean_f1", 0.15), ("slide_mean_auc", 0.175),
    ("auc_mean", 0.15), ("survival_mean_cindex", 0.075), ("robustness_mean", 0.15),
)
THUNDER_V2 = json.loads((BENCHMARKING_DIR / "thunder_v2.json").read_text())
assert THUNDER_V2["protocol_version"] == PROBE_PROTOCOL_VERSION
assert all(set(spec) == {"root", "train", "val"} for family in ("classification", "segmentation") for spec in THUNDER_V2[family].values())
EMBED_BATCH_SIZE = 512
EMBED_NUM_WORKERS = 16
SEGMENTATION_EPOCHS = {"pannuke": 30, "segpath_epithelial": 9, "segpath_lymphocytes": 21}
SEGMENTATION_HYPERPARAMETERS = {
    "pannuke": (1e-3, 1e-4),
    "segpath_epithelial": (1e-4, 1e-3),
    "segpath_lymphocytes": (1e-3, 1e-4),
}
SEGMENTATION_NUM_CLASSES = {"pannuke": 6, "segpath_epithelial": 2, "segpath_lymphocytes": 2}
SEGMENTATION_DECODER_DIM = 192
SEGMENTATION_BATCH_SIZE = 64
SEG_SPLIT_SEED = 1337
THUNDER_PROBE_SEED = 0
REPEATED_FOLDS = 3
LINEAR_PROBE_LRS = (1e-3, 1e-4, 1e-5)
LINEAR_PROBE_WEIGHT_DECAYS = (0.0, 1e-3, 1e-4)
LINEAR_PROBE_EPOCHS = 200
LINEAR_PROBE_BATCH_SIZE = 64
FEWSHOT_SHOT = 16
FEWSHOT_SUPPORT_SETS = 1000
FEWSHOT_SUPPORT_CHUNK = 64
KNN_K_VALS = [1, 3, 5, 10, 20, 30, 40, 50]
KNN_CHUNK_SIZE = 4096
CLASSIFICATION_DATASETS = [
    "bach", "bracs", "break_his", "crc", "esca", "mhist", "pcam",
    "spider_breast", "spider_colorectal", "spider_skin", "spider_thorax",
    "wilds",
]
SEGMENTATION_DATASETS = ["pannuke", "segpath_epithelial", "segpath_lymphocytes"]
SLIDE_DATASETS = ["ucla_lung"]
AUC_DATASETS = ["surgen"]
SURVIVAL_DATASETS = ["leopard_bcr", "cptac_pda_os"]
ROBUSTNESS_DATASETS = ["pathorob"]
PROBE_DATASETS = [
    *CLASSIFICATION_DATASETS, *SEGMENTATION_DATASETS,
    "ucla_lung", "surgen", "leopard_bcr", "cptac_pda_os", "pathorob",
]
TASK_FIELDS = {
    "classification_datasets": ("datasets", CLASSIFICATION_DATASETS),
    "segmentation_datasets": ("segmentation_datasets", SEGMENTATION_DATASETS),
    "slide_datasets": ("slide_datasets", SLIDE_DATASETS),
    "auc_datasets": ("auc_datasets", AUC_DATASETS),
    "survival_datasets": ("survival_datasets", SURVIVAL_DATASETS),
    "robustness_datasets": ("robustness_datasets", ROBUSTNESS_DATASETS),
}
PATHOBENCH_LR_C = 0.5
SURGEN_LR_MAX_ITER = 5000
SURGEN_TILES_PER_SLIDE = 768
SURGEN_ROW_GROUP_SIZE = 64
SURVIVAL_TILES_PER_SLIDE_CAPS = {"leopard_bcr": 768, "cptac_pda_os": 0}  # 0 means uncapped.
SURVIVAL_COXNET_ALPHA_FRACTIONS = (0.1, 0.2, 0.7)
SURVIVAL_COXNET_L1_RATIO = 0.5
SURVIVAL_COXNET_MAX_ITER = 100000
PATHOROB_SUBSETS = ("camelyon", "tolkach_esca")
CROMA_M = 5
# Module-level so dataset adapters can read roots without threading cfg through every call.
# Populated from cfg.probe.dataset_roots by prepare_probe_state() and run_probe_job().
DATASET_ROOTS = {}


# Prefix probe logs with the same timestamp/job id format as train.py.
def console_prefix():
    return f"{time.strftime('%H:%M:%S')} {os.environ.get('SLURM_JOB_ID', str(os.getpid()))}"


# Keep all protocol-v2 sidecar files under one probe directory.
def probe_paths(output_dir):
    probe_dir = Path(output_dir) / "probe"
    return {
        "probe_dir": probe_dir,
        "state_path": probe_dir / "state.json",
        "results_dir": probe_dir / "results",
    }


# Probes are enabled only when the recipe asks for them and names at least one task.
def probe_enabled(cfg):
    return bool(cfg["probe"]["enabled"]) and sum(len(cfg["probe"].get(cfg_key, [])) for cfg_key, _ in TASK_FIELDS.values()) > 0


# Persist probe state so explicitly resumed train.py runs do not relog completed result files.
def write_probe_state(state):
    state["paths"]["state_path"].write_text(json.dumps(state["data"], indent=2) + "\n")


# Deterministic repeated validation folds for small train-derived probes.
def stratified_folds(labels):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(n_splits=REPEATED_FOLDS, shuffle=True, random_state=SEG_SPLIT_SEED).split(np.zeros(len(labels)), labels))


# Validate probe recipe compatibility and initialize the on-disk result tracker.
def prepare_probe_state(cfg, output_dir):
    DATASET_ROOTS.clear()
    DATASET_ROOTS.update({k: Path(v) for k, v in cfg["probe"]["dataset_roots"].items()})
    paths = probe_paths(output_dir)
    for path in [paths["probe_dir"], paths["results_dir"]]:
        path.mkdir(parents=True, exist_ok=True)
    groups = {request_key: [str(x) for x in cfg["probe"].get(cfg_key, [])] for request_key, (cfg_key, _) in TASK_FIELDS.items()}
    data = {
        "version": 17,
        "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        "family": str(cfg["project"]["family"]),
        "count": int(cfg["probe"]["count"]),
        "logged_results": [],
        **groups,
    }
    if paths["state_path"].exists():
        # Explicit resume can continue only if the probe family/datasets/count match the old state.
        previous = json.loads(paths["state_path"].read_text())
        if previous["version"] != 17:
            raise ValueError(f"unsupported probe state version: {previous['version']}")
        if previous["family"] != data["family"]:
            raise ValueError(f"probe family changed from {previous['family']} to {data['family']}")
        for request_key in TASK_FIELDS:
            if previous.get(request_key, []) != data[request_key]:
                raise ValueError(f"{request_key} changed from {previous.get(request_key, [])} to {data[request_key]}")
        if previous["count"] != data["count"]:
            raise ValueError(f"probe count changed from {previous['count']} to {data['count']}")
        data["logged_results"] = previous["logged_results"]
    for request_key, (_, supported) in TASK_FIELDS.items():
        for dataset in data[request_key]:
            if dataset not in supported:
                raise ValueError(f"unsupported {request_key}: {dataset}")
    configured = [d for request_key in TASK_FIELDS for d in data[request_key]]
    assert configured == PROBE_DATASETS, f"probe config must contain exactly {PROBE_DATASETS}, got {configured}"
    state = {"paths": paths, "data": data}
    write_probe_state(state)
    return state


# Snapshot a checkpoint payload and run this file as a separate process for clean GPU memory.
def queue_probe_job(state, checkpoint_payload, checkpoint_step, target_flops, target_fraction):
    step_tag = f"step_{checkpoint_step:07d}"
    slurm_id = os.environ.get("SLURM_JOB_ID", f"local-{os.getpid()}")
    request = {
        "checkpoint_step": int(checkpoint_step),
        "train_step": int(checkpoint_step),
        "target_flops": int(target_flops),
        "target_fraction": float(target_fraction),
        "checkpoint_path": str(state["paths"]["probe_dir"] / f"{step_tag}.pt"),
        "request_path": str(state["paths"]["probe_dir"] / f"{step_tag}.request.json"),
        "result_path": str(state["paths"]["results_dir"] / f"{step_tag}.json"),
        **{request_key: list(state["data"][request_key]) for request_key in TASK_FIELDS},
        "job_id": f"{slurm_id}-{checkpoint_step:07d}",
    }
    for dataset in [d for request_key in TASK_FIELDS for d in request[request_key]]:
        if not DATASET_ROOTS[dataset].exists():
            raise FileNotFoundError(f"missing dataset root for {dataset}: {DATASET_ROOTS[dataset]}")
    torch.save(checkpoint_payload, request["checkpoint_path"])
    Path(request["request_path"]).write_text(json.dumps(request, indent=2) + "\n")
    torch.cuda.empty_cache()
    env = os.environ.copy()
    env.pop("WANDB_SERVICE", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent)
    print(
        f"{console_prefix()} Probe  [{checkpoint_step}]  "
        f"start: {request['job_id']}  target_fraction: {target_fraction:.4f}  "
        f"classification: {','.join(request['classification_datasets']) or '-'}  "
        f"segmentation: {','.join(request['segmentation_datasets']) or '-'}  "
        f"slide: {','.join(request['slide_datasets']) or '-'}  "
        f"auc: {','.join(request['auc_datasets']) or '-'}  "
        f"survival: {','.join(request['survival_datasets']) or '-'}  "
        f"robustness: {','.join(request['robustness_datasets']) or '-'}",
        flush=True,
    )
    subprocess.run([sys.executable, str(Path(__file__).resolve()), request["request_path"]], env=env, check=True)
    print(
        f"{console_prefix()} Probe  [{checkpoint_step}]  "
        f"finished: {request['job_id']}  result: {request['result_path']}",
        flush=True,
    )


# Image dataset adapter for classification probes; dataset-specific split logic lives here.
class ClassificationDataset(torch.utils.data.Dataset):
    # Load the checked-in THUNDER v2 train/validation selections. PCam remains H5-backed.
    def __init__(self, dataset, split, transform):
        import h5py
        import numpy as np

        self.transform = transform
        self.dataset = dataset
        spec = THUNDER_V2["classification"][dataset][split]
        if dataset == "pcam":
            pcam_split = "train" if split == "train" else "valid"
            with h5py.File(DATASET_ROOTS["pcam"] / f"camelyonpatch_level_2_split_{pcam_split}_x.h5", "r") as fx:
                key_x = next(iter(fx.keys()))
                idx = np.asarray(spec["indices"], dtype=np.int64)
                self.images = np.array(fx[key_x][idx])
            self.labels = [int(v) for v in spec["labels"]]
        else:
            self.images = spec["images"]
            self.labels = [int(v) for v in spec["labels"]]
            self.root = DATASET_ROOTS[dataset]

    # Number of labeled examples in this probe split.
    def __len__(self):
        return len(self.labels)

    # Return one transformed RGB image and integer label for embedding.
    def __getitem__(self, idx):
        from PIL import Image
        if self.dataset == "pcam":
            img = Image.fromarray(self.images[idx])
        else:
            img = Image.open(self.root / self.images[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


# Manifest-backed THUNDER segmentation images shared by parallel decode threads.
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, split, transform):
        self.dataset, self.transform = dataset, transform
        self.root = DATASET_ROOTS[dataset]
        spec = THUNDER_V2["segmentation"][dataset][split]
        if dataset == "pannuke":
            import numpy as np
            self.images = np.load(self.root / spec["images"], mmap_mode="r")
            self.masks = np.load(self.root / spec["labels"], mmap_mode="r")
        else:
            self.image_records, self.label_records = spec["images"], spec["labels"]
            self.groups = []
            for i, record in enumerate(self.image_records):
                if not self.groups or self.image_records[self.groups[-1][0]][0] != record[0]:
                    self.groups.append([])
                self.groups[-1].append(i)

    def __len__(self):
        return len(self.images) if self.dataset == "pannuke" else len(self.groups)

    def __getitem__(self, i):
        import numpy as np
        if self.dataset == "pannuke":
            label = np.zeros((256, 256), dtype=np.int64)
            for class_id in range(1, SEGMENTATION_NUM_CLASSES["pannuke"]):
                np.copyto(label, class_id, where=self.masks[i, :, :, class_id - 1] > 0)
            return [(self.transform(Image.fromarray(self.images[i].astype(np.uint8))), torch.from_numpy(label))]
        source_image = Image.open(self.root / self.image_records[self.groups[i][0]][0]).convert("RGB")
        source_label = np.asarray(Image.open(self.root / self.label_records[self.groups[i][0]][0]))
        crops = []
        for index in self.groups[i]:
            _, i0, i1, j0, j1 = self.image_records[index]
            _, l0, l1, m0, m1 = self.label_records[index]
            image = source_image.crop((j0, i0, j1, i1))
            label = source_label[l0:l1, m0:m1].astype(np.int64)
            crops.append((self.transform(image), torch.from_numpy(label.copy())))
        return crops


# Mean-pool cached PathoBench-style tile embeddings to one vector per slide.
def embed_slide_dataset(model, mean, std, dataset, split, device, transform):
    import io
    import numpy as np
    from PIL import Image

    spec = json.loads((BENCHMARKING_DIR / f"{dataset}.json").read_text())
    split_names = (split,) if isinstance(split, str) else tuple(split)
    slides, labels = [], []
    for name in split_names:
        s = spec[name]
        slides += list(s["slide_ids"])
        labels += [int(v) for v in s["labels"]]
    labels = np.asarray(labels, dtype=np.int64)
    paths, slide_idx = [], []
    import pyarrow.parquet as pq
    slide_to_i = {s: i for i, s in enumerate(slides)}
    table = pq.read_table(DATASET_ROOTS[dataset] / "tiles.parquet")
    for sid, jpg in zip(table.column("slide_id").to_pylist(), table.column("jpeg").to_pylist()):
        if sid in slide_to_i:
            paths.append(jpg); slide_idx.append(slide_to_i[sid])

    class _Tiles(torch.utils.data.Dataset):
        def __len__(self): return len(paths)
        def __getitem__(self, i):
            return transform(Image.open(io.BytesIO(paths[i])).convert("RGB")), slide_idx[i]

    loader = torch.utils.data.DataLoader(_Tiles(), batch_size=EMBED_BATCH_SIZE, shuffle=False, num_workers=EMBED_NUM_WORKERS, pin_memory=True)
    sums, counts = None, torch.zeros(len(slides), dtype=torch.long)
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    with torch.no_grad():
        for x, si in loader:
            x = x.to(device, non_blocking=True)
            with autocast:
                e = model.probe_features((x - mean) / std).float().cpu()
            if sums is None:
                sums = torch.zeros(len(slides), e.shape[1])
            sums.index_add_(0, si, e)
            counts.index_add_(0, si, torch.ones_like(si))
    return (sums / counts.unsqueeze(1)).numpy().astype(np.float32), labels


# Run the frozen backbone over one classification split and return numpy embeddings/labels.
def embed_classification_dataset(model, mean, std, dataset, split, device, transform):
    import numpy as np

    loader = torch.utils.data.DataLoader(
        ClassificationDataset(dataset, split, transform),
        batch_size=EMBED_BATCH_SIZE,
        shuffle=False,
        num_workers=EMBED_NUM_WORKERS,
        pin_memory=True,
    )
    embs, labels = [], []
    # THUNDER's published adapter extracts frozen tile embeddings in fp16.
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    # probe_features() supplies each encoder's published frozen representation;
    # none of the DINO/iBOT training heads are involved.
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with autocast:
                e = model.probe_features((x - mean) / std)
            embs.append(e.float().cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(embs, axis=0).astype(np.float32), np.concatenate(labels, axis=0).astype(np.int64)


# Multiclass Dice loss for the THUNDER segmentation probes; mask gates invalid pixels.
# Vendored from Thunder (thunder/src/thunder/utils/dice_loss.py).
def multiclass_dice_loss(pred, label, mask, smooth=1.0):
    pred = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    target = label.clone()
    target[~mask] = num_classes
    target = F.one_hot(target, num_classes=num_classes + 1)[..., :-1].permute(0, 3, 1, 2)
    mask = mask.unsqueeze(1)
    intersection = (pred * target * mask).sum(dim=(0, 2, 3))
    union = (pred * mask).sum(dim=(0, 2, 3)) + (target * mask).sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()


# Pre-LN transformer decoder block (qkv attention + MLP) used inside MaskTransformer.
class _SegBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)
        self.fc1, self.fc2 = nn.Linear(dim, mlp_dim), nn.Linear(mlp_dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        # Flash SDPA backward is numerically non-reproducible enough for the
        # fixed 21-epoch head to cross decision boundaries. The decoder is
        # small, so use the deterministic math kernel here while the frozen
        # encoder retains its fast fused attention path.
        with sdpa_kernel(SDPBackend.MATH):
            attn = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], dropout_p=0.0, scale=(c // self.heads) ** -0.5)
        attn = attn.transpose(1, 2).reshape(b, n, c)
        x = x + self.proj(attn)
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


# Trunc-normal Linear, zero-init bias, identity LayerNorm — Thunder's seg-head init.
def _init_seg_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


# Segmentation decoder vendored from Thunder (thunder/src/thunder/models/task_specific_models.py).
# Project frozen encoder patch tokens into d_model, append n_cls learnable class tokens, run a
# few decoder blocks, then emit low-resolution class masks; the probe upsamples them to each
# task's label resolution.
class MaskTransformer(nn.Module):
    def __init__(self, n_cls, d_encoder, n_layers=2, n_heads=8, d_model=768, d_ff=3072):
        super().__init__()
        self.d_encoder = d_encoder
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5
        self.blocks = nn.ModuleList(_SegBlock(d_model, n_heads, d_ff) for _ in range(n_layers))
        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)
        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)
        self.apply(_init_seg_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    def forward(self, x):
        b, n, _ = x.shape
        gs = int(n ** 0.5)
        x = self.proj_dec(x)
        x = torch.cat([x, self.cls_emb.expand(b, -1, -1)], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        patches, cls_seg = x[:, : -self.n_cls] @ self.proj_patch, x[:, -self.n_cls :] @ self.proj_classes
        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg = cls_seg / cls_seg.norm(dim=-1, keepdim=True)
        masks = self.mask_norm(patches @ cls_seg.transpose(1, 2))
        return masks.reshape(b, gs, gs, self.n_cls).permute(0, 3, 1, 2)


# Load one manifest-selected THUNDER segmentation split and extract dense frozen tokens once.
def embed_segmentation_dataset(model, mean, std, dataset, split, device, transform):
    from concurrent.futures import ThreadPoolExecutor

    # Threads keep PNG decoding parallel without forking an initialized CUDA process.
    tiles = SegmentationDataset(dataset, split, transform)
    feats, scales, labels = [], [], []
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    with torch.no_grad():
        batch_images, batch_labels = [], []
        with ThreadPoolExecutor(max_workers=EMBED_NUM_WORKERS) as pool:
            for i, crops in enumerate(pool.map(tiles.__getitem__, range(len(tiles))), 1):
                for crop_index, (image, label) in enumerate(crops, 1):
                    batch_images.append(image); batch_labels.append(label)
                    if len(batch_images) == EMBED_BATCH_SIZE or (i == len(tiles) and crop_index == len(crops)):
                        images = torch.stack(batch_images).to(device)
                        with autocast:
                            batch_feats = model.encode_image((images - mean) / std)
                        # Preserve model-defined test-time feature aggregation while preventing an
                        # upsampled token grid from multiplying the shared decoder's quadratic cost.
                        # Ordinary outputs are unchanged; expanded spatial grids are area-pooled to
                        # the encoder's native patch grid while every aggregated channel is retained.
                        if hasattr(model, "patch_size"):
                            native_h, native_w = images.shape[-2] // model.patch_size, images.shape[-1] // model.patch_size
                            if batch_feats.shape[1] != native_h * native_w:
                                side = int(batch_feats.shape[1] ** 0.5)
                                assert side * side == batch_feats.shape[1]
                                batch_feats = F.adaptive_avg_pool2d(
                                    batch_feats.transpose(1, 2).reshape(len(images), batch_feats.shape[-1], side, side),
                                    (native_h, native_w),
                                ).flatten(2).transpose(1, 2)
                        batch_feats = batch_feats.float()
                        batch_scales = batch_feats.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-12).div_(127).to(torch.float16)
                        feats.append(torch.clamp(torch.round(batch_feats / batch_scales.float()), -127, 127).to(torch.int8).cpu())
                        scales.append(batch_scales.cpu())
                        labels.append(torch.stack(batch_labels).to(torch.int8))
                        batch_images, batch_labels = [], []
    return torch.cat(feats), torch.cat(scales), torch.cat(labels)


# Train the THUNDER MaskTransformer for a fixed task schedule, then score the
# complete validation split once. Validation never selects a checkpoint.
def inline_segmentation_f1(model, mean, std, dataset, device, transform):
    import numpy as np

    started_at = time.monotonic()
    train_feats, train_scales, train_labels = embed_segmentation_dataset(model, mean, std, dataset, "train", device, transform)
    val_feats, val_scales, val_labels = embed_segmentation_dataset(model, mean, std, dataset, "val", device, transform)
    feature_cache_bytes = train_feats.numel() + val_feats.numel() + 2 * (train_scales.numel() + val_scales.numel())
    features_on_gpu = feature_cache_bytes <= 24 * 1024 ** 3
    if features_on_gpu:
        train_feats, train_scales = train_feats.to(device), train_scales.to(device)
        val_feats, val_scales = val_feats.to(device), val_scales.to(device)
    n_cls = SEGMENTATION_NUM_CLASSES[dataset]
    torch.manual_seed(THUNDER_PROBE_SEED)
    torch.cuda.manual_seed_all(THUNDER_PROBE_SEED)
    torch.set_float32_matmul_precision("high")
    head = MaskTransformer(
        n_cls=n_cls,
        d_encoder=train_feats.shape[-1],
        n_layers=2,
        n_heads=8,
        d_model=SEGMENTATION_DECODER_DIM,
        d_ff=SEGMENTATION_DECODER_DIM * 4,
    ).to(device)
    decoder = torch.compile(head)
    lr, weight_decay = SEGMENTATION_HYPERPARAMETERS[dataset]
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(SEGMENTATION_EPOCHS[dataset]):
        head.train()
        # Match THUNDER's cached loader: draw its iterator seed on CPU, then
        # construct the shuffled CPU indices from a fresh seeded generator.
        torch.empty((), dtype=torch.int64).random_()
        shuffle_seed = int(torch.empty((), dtype=torch.int64).random_().item())
        order = torch.randperm(len(train_feats), generator=torch.Generator().manual_seed(shuffle_seed))
        for start in range(0, len(order), SEGMENTATION_BATCH_SIZE):
            idx = order[start:start + SEGMENTATION_BATCH_SIZE]
            labels = train_labels[idx].to(device=device, dtype=torch.long)
            if features_on_gpu:
                gpu_idx = idx.to(device)
                batch_feats = train_feats[gpu_idx].float() * train_scales[gpu_idx].float()
            else:
                batch_feats = train_feats[idx].to(device).float() * train_scales[idx].to(device).float()
            logits = F.interpolate(decoder(batch_feats), labels.shape[-2:], mode="bilinear")
            loss = multiclass_dice_loss(logits, labels, labels != -1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        # THUNDER constructs a non-shuffled validation iterator after each
        # epoch. Consume its base-seed draw without reading validation data so
        # the next training epoch follows the same RNG stream.
        torch.empty((), dtype=torch.int64).random_()

    rows = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(val_feats), SEGMENTATION_BATCH_SIZE):
            labels = val_labels[start:start + SEGMENTATION_BATCH_SIZE].to(device=device, dtype=torch.long)
            batch_feats = val_feats[start:start + SEGMENTATION_BATCH_SIZE].to(device).float() * val_scales[start:start + SEGMENTATION_BATCH_SIZE].to(device).float()
            logits = decoder(batch_feats)
            pred = F.interpolate(logits, labels.shape[-2:], mode="bilinear").argmax(1)
            valid = labels != -1
            tp, fp, fn = [], [], []
            for class_id in range(n_cls):
                pc, tc = pred == class_id, labels == class_id
                tp.append((pc & tc & valid).sum((1, 2))); fp.append((pc & ~tc & valid).sum((1, 2))); fn.append((~pc & tc & valid).sum((1, 2)))
            tp, fp, fn = torch.stack(tp, 1).double(), torch.stack(fp, 1).double(), torch.stack(fn, 1).double()
            present = tp + fp + fn > 0
            class_f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
            f1 = class_f1.sum(1) / present.sum(1)
            jaccard = (tp / (tp + fp + fn).clamp(min=1)).sum(1) / present.sum(1)
            pixel_counts = valid.sum((1, 2)).double()
            keep = pixel_counts > 0
            rows.append(torch.stack((f1[keep], jaccard[keep], pixel_counts[keep], (labels.masked_fill(~valid, 0).sum((1, 2))[keep] == 0).double()), 1).cpu())
    values = torch.cat(rows).numpy()
    weights = values[:, 2].astype(np.float32); background_only = values[:, 3].astype(bool)
    weights[~background_only] *= max(1.0, background_only.mean() * 16.0)
    result = {
        "seg_val_f1": float(np.average(values[:, 0], weights=weights)),
        "seg_val_jaccard": float(np.average(values[:, 1], weights=weights)),
        "epochs": SEGMENTATION_EPOCHS[dataset],
        "lr": lr,
        "weight_decay": weight_decay,
        "selection_split": None,
        "decoder_dim": SEGMENTATION_DECODER_DIM,
        "encoder_tokens": train_feats.shape[1],
        "encoder_dim": train_feats.shape[2],
        "train_examples": len(train_feats),
        "val_examples": len(val_feats),
        "feature_cache": "gpu" if features_on_gpu else "cpu",
    }
    return result, time.monotonic() - started_at


# CRoMa all-row design, m=5: clemsgrs/croma@3f58d5e4bd9ecf74c34d0c76eb88d80cee9fb706.
# Typed top-k replaces adaptive searches exactly; float64 avoids cancellation near cosine 1.
def croma_score(embeddings, meta, device):
    import numpy as np

    geometry = torch.tensor(embeddings, dtype=torch.float64, device=device)
    geometry /= torch.linalg.vector_norm(geometry, dim=1, keepdim=True)
    slide, label, center = [
        torch.from_numpy(np.unique(meta[key].to_numpy(dtype=str), return_inverse=True)[1]).to(device)
        for key in ("slide_id", "biological_class", "medical_center")
    ]
    margins = []
    for s in range(0, len(meta), KNN_CHUNK_SIZE):
        e = min(s + KNN_CHUNK_SIZE, len(meta))
        distance = (1 - geometry[s:e] @ geometry.T).clamp_(0, 2)
        same_slide, same_label, same_center = [x[s:e, None] == x[None] for x in (slide, label, center)]
        d_so = distance.masked_fill(~(same_label & ~same_center & ~same_slide), float("inf")).topk(CROMA_M, dim=1, largest=False).values.mean(1)
        d_os = distance.masked_fill(~(~same_label & same_center & ~same_slide), float("inf")).topk(CROMA_M, dim=1, largest=False).values.mean(1)
        margins.append((d_os - d_so) / (d_os + d_so))
    sample_croma = torch.cat(margins).cpu().numpy()
    assert np.isfinite(sample_croma).all(), "CRoMa requires five SO/OS neighbors and nonzero distance sums for every tile"
    croma = float(np.median(sample_croma))
    q10 = float(np.percentile(sample_croma, 10))
    return {
        "croma": croma,
        "croma_q10": q10,
        "croma_ltm10": float(sample_croma[sample_croma <= q10].mean()),
        "croma_f0": float((sample_croma <= 0).mean()),
        "croma_m": CROMA_M,
        "robustness": (1 + croma) / 2,
    }


# CRoMa over held-out Camelyon + Tolkach ESCA. The adapter remains fixed to PathoROB's
# final CLS plus mean patch tokens rather than the model-defined pooled probe features.
def inline_pathorob(model, mean, std, device, transform):
    import io
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    started_at = time.monotonic()
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)

    class _Patches(torch.utils.data.Dataset):
        def __init__(self, byts): self.byts = byts
        def __len__(self): return len(self.byts)
        def __getitem__(self, i): return transform(Image.open(io.BytesIO(self.byts[i])).convert("RGB"))

    out = {}
    for name in PATHOROB_SUBSETS:
        tbl = pa.concat_tables([pq.read_table(f) for f in sorted((DATASET_ROOTS["pathorob"] / name).glob("data/*.parquet"))])
        meta = tbl.select(["slide_id", "biological_class", "medical_center"]).to_pandas()
        if name == "tolkach_esca":
            keep = meta.medical_center.to_numpy(dtype=object) != "VALSET3_TCGA"
            tbl = tbl.filter(pa.array(keep))
            meta = meta[keep].reset_index(drop=True)
        byts = [r["bytes"] for r in tbl.column("image").to_pylist()]
        loader = torch.utils.data.DataLoader(_Patches(byts), batch_size=EMBED_BATCH_SIZE, num_workers=EMBED_NUM_WORKERS, pin_memory=True, shuffle=False)
        embs = []
        with torch.no_grad():
            for batch in loader:
                x = batch.to(device, non_blocking=True)
                with autocast:
                    o = model((x - mean) / std)
                    feat = torch.cat([o["cls"], o["patches"].mean(dim=1)], dim=-1)
                embs.append(feat.float().cpu().numpy())
        out[name] = croma_score(np.concatenate(embs), meta, device)
    return out, time.monotonic() - started_at


def inline_surgen_ras_auc(model, mean, std, device, transform):
    import io
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    started_at = time.monotonic()
    splits = json.loads((BENCHMARKING_DIR / "surgen.json").read_text())
    pool_slides = list(splits["train"]["slides"]) + list(splits["val"]["slides"])
    pool_labels = np.asarray([int(v) for split in ("train", "val") for v in splits[split]["labels"]], dtype=np.int64)
    label_of = dict(zip(pool_slides, pool_labels))
    files = sorted((DATASET_ROOTS["surgen"] / "data").glob("surgen-*.parquet"))
    # Prepared rows follow a raster scan; spaced row groups preserve slide coverage without reading the full 102 GB cache.
    row_groups = defaultdict(list)
    for fi, f in enumerate(files):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            stats = pf.metadata.row_group(rg).column(1).statistics
            sids = [stats.min] if stats.min == stats.max else set(pf.read_row_group(rg, columns=["slide_id"]).column("slide_id").to_pylist())
            for sid in sids:
                if sid in label_of:
                    row_groups[sid].append((fi, rg))
    groups_per_slide = (SURGEN_TILES_PER_SLIDE + SURGEN_ROW_GROUP_SIZE - 1) // SURGEN_ROW_GROUP_SIZE
    keep_groups = defaultdict(set)
    for sid in pool_slides:
        groups = row_groups[sid]
        take = range(len(groups)) if len(groups) <= groups_per_slide else np.linspace(0, len(groups) - 1, groups_per_slide, dtype=np.int64)
        for i in take:
            fi, rg = groups[int(i)]
            keep_groups[fi].add(rg)
    selected_groups = [(fi, rg) for fi in sorted(keep_groups) for rg in sorted(keep_groups[fi])]

    class _Tiles(torch.utils.data.IterableDataset):
        def __iter__(self):
            worker = torch.utils.data.get_worker_info()
            if worker is None:
                groups = selected_groups
            else:
                per = (len(selected_groups) + worker.num_workers - 1) // worker.num_workers
                groups = selected_groups[worker.id * per : (worker.id + 1) * per]
            cur_fi, pf = None, None
            for fi, rg in groups:
                if fi != cur_fi:
                    cur_fi, pf = fi, pq.ParquetFile(files[fi])
                table = pf.read_row_group(rg, columns=["jpeg", "slide_id"])
                for b, sid in zip(table.column("jpeg").to_pylist(), table.column("slide_id").to_pylist()):
                    if sid in label_of:
                        yield transform(Image.open(io.BytesIO(b)).convert("RGB")), sid

    loader = torch.utils.data.DataLoader(_Tiles(), batch_size=EMBED_BATCH_SIZE, num_workers=EMBED_NUM_WORKERS, pin_memory=True)
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    sums, counts, tiles = {}, defaultdict(int), 0
    with torch.no_grad():
        for x, sids in loader:
            x = x.to(device, non_blocking=True)
            with autocast:
                batch = model.probe_features((x - mean) / std).float().cpu().numpy()
            for sid, vec in zip(sids, batch):
                sums[sid] = sums.get(sid, 0.0) + vec.astype(np.float64)
                counts[sid] += 1
                tiles += 1
    X = np.stack([sums[s] / counts[s] for s in pool_slides]).astype(np.float32)
    folds = []
    for tr, va in stratified_folds(pool_labels):
        clf = LogisticRegression(C=PATHOBENCH_LR_C, class_weight="balanced", max_iter=SURGEN_LR_MAX_ITER, random_state=0).fit(X[tr], pool_labels[tr])
        folds.append(float(roc_auc_score(pool_labels[va], clf.predict_proba(X[va])[:, 1])))
    return {"val_auc": float(np.mean(folds)), "c": PATHOBENCH_LR_C, "fold_scores": folds, "tiles": tiles, "tiles_per_slide_cap": SURGEN_TILES_PER_SLIDE}, time.monotonic() - started_at


def inline_pathobench_survival(model, mean, std, dataset, device, transform):
    import io
    import warnings
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxnetSurvivalAnalysis

    started_at = time.monotonic()
    spec = json.loads((BENCHMARKING_DIR / f"{dataset}.json").read_text())
    if "case_ids" in spec:
        unit_ids = list(spec["case_ids"])
        unit_slides = [list(x) for x in spec["case_slides"]]
        pool_events = np.asarray([bool(e) for e in spec["events"]])
        pool_days = np.asarray([float(d) for d in spec["days"]])
    else:
        splits = [sid for split in ("train", "val") for sid in spec[split]["slide_ids"]]
        unit_ids = [str(x) for split in ("train", "val") for x in spec[split].get("case_ids", spec[split]["slide_ids"])]
        unit_slides = [[sid] for sid in splits]
        pool_events = np.asarray([bool(e) for split in ("train", "val") for e in spec[split]["events"]])
        pool_days = np.asarray([float(d) for split in ("train", "val") for d in spec[split]["days"]])
    pool_slides = [sid for slides in unit_slides for sid in slides]
    needed = set(pool_slides)
    pf = pq.ParquetFile(DATASET_ROOTS[dataset] / "patches.parquet")
    slide_col = pf.schema_arrow.get_field_index("slide_id")
    row_groups = defaultdict(list)
    for rg in range(pf.num_row_groups):
        stats = pf.metadata.row_group(rg).column(slide_col).statistics
        sids = [stats.min] if stats.min == stats.max else set(pf.read_row_group(rg, columns=["slide_id"]).column("slide_id").to_pylist())
        for sid in sids:
            if sid in needed:
                row_groups[sid].append(rg)
    cap = SURVIVAL_TILES_PER_SLIDE_CAPS[dataset]
    groups_per_slide = (cap + pf.metadata.row_group(0).num_rows - 1) // pf.metadata.row_group(0).num_rows if cap else None
    selected_groups = set()
    for sid in pool_slides:
        groups = row_groups[sid]
        take = range(len(groups)) if groups_per_slide is None or len(groups) <= groups_per_slide else np.linspace(0, len(groups) - 1, groups_per_slide, dtype=np.int64)
        selected_groups.update(groups[int(i)] for i in take)
    selected_groups = sorted(selected_groups)

    class _Tiles(torch.utils.data.IterableDataset):
        def __iter__(self):
            worker = torch.utils.data.get_worker_info()
            groups = selected_groups if worker is None else selected_groups[worker.id::worker.num_workers]
            pf = pq.ParquetFile(DATASET_ROOTS[dataset] / "patches.parquet")
            for rg in groups:
                table = pf.read_row_group(rg, columns=["image", "slide_id"])
                for b, sid in zip(table.column("image").to_pylist(), table.column("slide_id").to_pylist()):
                    if sid in needed:
                        yield transform(Image.open(io.BytesIO(b)).convert("RGB")), sid

    loader = torch.utils.data.DataLoader(_Tiles(), batch_size=EMBED_BATCH_SIZE, num_workers=EMBED_NUM_WORKERS, pin_memory=True)
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    sums, counts, tiles = {}, defaultdict(int), 0
    with torch.no_grad():
        for x, sids in loader:
            x = x.to(device, non_blocking=True)
            with autocast:
                batch = model.probe_features((x - mean) / std).float().cpu().numpy()
            for sid, vec in zip(sids, batch):
                sums[sid] = sums.get(sid, 0.0) + vec.astype(np.float64)
                counts[sid] += 1
                tiles += 1

    slide_vecs = {sid: sums[sid] / counts[sid] for sid in pool_slides}
    X = np.stack([np.stack([slide_vecs[sid] for sid in slides]).mean(0) for slides in unit_slides]).astype(np.float64)
    y = np.array(list(zip(pool_events, pool_days)), dtype=[("event", bool), ("days", float)])
    if "folds" in spec:
        case_to_i = {case_id: i for i, case_id in enumerate(unit_ids)}
        fold_indices = [(np.asarray([case_to_i[c] for c in fold["train"]], dtype=np.int64), np.asarray([case_to_i[c] for c in fold["val"]], dtype=np.int64)) for fold in spec["folds"]]
    else:
        fold_indices = stratified_folds(pool_events.astype(np.int64))
    folds = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for tr, va in fold_indices:
            scaler = StandardScaler().fit(X[tr])
            X_train = scaler.transform(X[tr])
            X_val = scaler.transform(X[va])
            alpha_max = CoxnetSurvivalAnalysis(
                l1_ratio=SURVIVAL_COXNET_L1_RATIO, n_alphas=2,
                alpha_min_ratio=0.99, max_iter=SURVIVAL_COXNET_MAX_ITER,
            ).fit(X_train, y[tr]).alphas_[0]
            for fraction in SURVIVAL_COXNET_ALPHA_FRACTIONS:
                alpha = alpha_max * fraction
                head = CoxnetSurvivalAnalysis(
                    alphas=[alpha], l1_ratio=SURVIVAL_COXNET_L1_RATIO,
                    max_iter=SURVIVAL_COXNET_MAX_ITER,
                ).fit(X_train, y[tr])
                folds.append({"alpha_fraction": fraction, "alpha": float(alpha), "alpha_max": float(alpha_max), "val_cindex": float(head.score(X_val, y[va])), "train_cases": len(tr), "val_cases": len(va)})
    val_cindex = float(np.mean([f["val_cindex"] for f in folds]))
    return {
        "val_cindex": val_cindex,
        "coxnet_alpha_fractions": list(SURVIVAL_COXNET_ALPHA_FRACTIONS),
        "coxnet_l1_ratio": SURVIVAL_COXNET_L1_RATIO,
        "coxnet_max_iter": SURVIVAL_COXNET_MAX_ITER,
        "coxnet_standardize": True,
        "val_cindex_per_alpha_fraction": {str(fraction): float(np.mean([f["val_cindex"] for f in folds if f["alpha_fraction"] == fraction])) for fraction in SURVIVAL_COXNET_ALPHA_FRACTIONS},
        "fold_scores": [float(f["val_cindex"]) for f in folds],
        "folds": folds,
        "tiles": tiles,
        "tiles_per_slide_cap": cap,
    }, time.monotonic() - started_at


# KNN probe marginalized over THUNDER's fixed k grid. No validation cell is selected.
def inline_knn_val_f1(train_embs, train_labels, val_embs, val_labels):
    import numpy as np
    from sklearn.metrics import f1_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = F.normalize(torch.from_numpy(train_embs.astype(np.float32, copy=False)), dim=1).to(device)
    val = F.normalize(torch.from_numpy(val_embs.astype(np.float32, copy=False)), dim=1).to(device)
    labels = torch.from_numpy(train_labels).long().to(device)
    num_classes = int(labels.max()) + 1
    neighbors = torch.empty(len(val), max(KNN_K_VALS), dtype=torch.long, device=device)
    for start in range(0, len(val), 1024):
        neighbors[start:start + 1024] = (val[start:start + 1024] @ train.T).topk(max(KNN_K_VALS), dim=1).indices
    preds_per_k = {
        k: F.one_hot(labels[neighbors[:, :k]], num_classes).sum(1).argmax(1).cpu().numpy()
        for k in KNN_K_VALS
    }
    f1_per_k = {k: float(f1_score(val_labels, preds_per_k[k], average="macro")) for k in KNN_K_VALS}
    return sum(f1_per_k.values()) / len(f1_per_k), f1_per_k


# THUNDER SimpleShot: recreate the published seed-0 support-index stream through
# 1/2/4/8/16 shot, then majority-vote the 1,000 centered 16-shot predictions.
def inline_fewshot_val_f1(train_embs, train_labels, val_embs, val_labels):
    import numpy as np
    from sklearn.metrics import f1_score

    train_embs = train_embs.astype(np.float32, copy=False)
    val_embs = val_embs.astype(np.float32, copy=False)
    labels = np.asarray(sorted(np.unique(train_labels)), dtype=np.int64)
    class_indices = [np.flatnonzero(train_labels == label) for label in labels]
    rng = random.Random(THUNDER_PROBE_SEED)
    support_sets = None
    for shot in (1, 2, 4, 8, FEWSHOT_SHOT):
        support_sets = np.asarray([
            [index for indices in class_indices for index in rng.sample(indices.tolist(), shot)]
            for _ in range(FEWSHOT_SUPPORT_SETS)
        ])
    if torch.cuda.is_available():
        device = torch.device("cuda")
        train_t = torch.from_numpy(train_embs).to(device)
        val_t = torch.from_numpy(val_embs).to(device)
        labels_t = torch.from_numpy(labels).to(device)
        support_sets_t = torch.from_numpy(support_sets).to(device)
        votes = []
        with torch.no_grad():
            for start in range(0, FEWSHOT_SUPPORT_SETS, FEWSHOT_SUPPORT_CHUNK):
                support = train_t[support_sets_t[start : start + FEWSHOT_SUPPORT_CHUNK]]
                mean = support.mean(dim=1)
                cls = (support - mean[:, None]).reshape(len(support), len(labels), FEWSHOT_SHOT, -1).mean(dim=2)
                cls = F.normalize(cls, dim=-1, eps=1e-12)
                val = F.normalize(val_t[None] - mean[:, None], dim=-1, eps=1e-12)
                votes.append(labels_t[torch.einsum("bvd,bcd->bvc", val, cls).argmax(dim=-1)].cpu().numpy())
        votes = np.concatenate(votes, axis=0)
    else:
        votes = np.empty((FEWSHOT_SUPPORT_SETS, len(val_labels)), dtype=np.int64)
        for i, support_idx in enumerate(support_sets):
            support = train_embs[support_idx]
            mean = support.mean(axis=0, keepdims=True)
            cls = (support - mean).reshape(len(labels), FEWSHOT_SHOT, -1).mean(axis=1)
            cls = cls / np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-12)
            val = val_embs - mean
            val = val / np.maximum(np.linalg.norm(val, axis=1, keepdims=True), 1e-12)
            votes[i] = labels[(val @ cls.T).argmax(axis=1)]
    preds = np.asarray([np.bincount(votes[:, i], minlength=int(labels.max()) + 1).argmax() for i in range(votes.shape[1])])
    return float(f1_score(val_labels, preds, average="macro"))


# THUNDER's nine Adam linear heads train together for a fixed 200 epochs. Their
# final macro-F1 values are averaged, so validation never selects a head or epoch.
def inline_linear_val_f1(train_embs, train_labels, val_embs, val_labels):
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(np.max(train_labels)) + 1
    train_embs_t = torch.from_numpy(train_embs).to(device)
    train_labels_t = torch.from_numpy(train_labels).long().to(device)
    val_embs_t = torch.from_numpy(val_embs).to(device)
    val_labels_t = torch.from_numpy(val_labels).long().to(device)
    classes = torch.arange(num_classes, device=device)[None, :, None]
    expected = val_labels_t[None, None] == classes
    hyperparameters = [(lr, decay) for lr in LINEAR_PROBE_LRS for decay in LINEAR_PROBE_WEIGHT_DECAYS]
    torch.manual_seed(THUNDER_PROBE_SEED)
    torch.cuda.manual_seed_all(THUNDER_PROBE_SEED)
    heads = nn.ModuleList(nn.Linear(train_embs.shape[1], num_classes) for _ in hyperparameters).to(device)
    optimizer = torch.optim.Adam([
        {"params": head.parameters(), "lr": lr, "weight_decay": decay}
        for head, (lr, decay) in zip(heads, hyperparameters)
    ])
    for _ in range(LINEAR_PROBE_EPOCHS):
        # Reproduce THUNDER's GPUEmbeddingLoader RNG stream exactly.
        torch.empty((), dtype=torch.int64).random_()
        shuffle_seed = int(torch.empty((), dtype=torch.int64).random_().item())
        order = torch.randperm(
            len(train_embs_t), generator=torch.Generator().manual_seed(shuffle_seed),
        ).to(device)
        for start in range(0, len(order), LINEAR_PROBE_BATCH_SIZE):
            indices = order[start:start + LINEAR_PROBE_BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            weights = torch.stack([head.weight for head in heads])
            biases = torch.stack([head.bias for head in heads])
            outputs = torch.einsum("bd,hcd->hbc", train_embs_t[indices], weights).add_(biases[:, None])
            # One fused kernel is exactly the sum of THUNDER's nine mean CE losses.
            F.cross_entropy(
                outputs.flatten(0, 1), train_labels_t[indices].repeat(len(heads)),
            ).mul(len(heads)).backward()
            optimizer.step()
        # Preserve THUNDER's next-epoch RNG stream without evaluating or
        # selecting on validation at intermediate epochs.
        torch.empty((), dtype=torch.int64).random_()
    with torch.no_grad():
        weights = torch.stack([head.weight for head in heads])
        biases = torch.stack([head.bias for head in heads])
        predictions = torch.einsum("bd,hcd->hbc", val_embs_t, weights).add_(biases[:, None]).argmax(2)
        predicted = predictions[:, None] == classes
        tp = (predicted & expected).sum(2)
        fp = (predicted & ~expected).sum(2)
        fn = (~predicted & expected).sum(2)
        scores = (2 * tp / (2 * tp + fp + fn).clamp(min=1)).mean(1).cpu().numpy()
    per_hyperparameter = {
        f"lr={lr:g},weight_decay={decay:g}": float(score)
        for (lr, decay), score in zip(hyperparameters, scores)
    }
    return float(np.mean(scores)), per_hyperparameter


def classification_head_metrics(train_embs, train_labels, val_embs, val_labels):
    knn_val_f1, knn_all = inline_knn_val_f1(train_embs, train_labels, val_embs, val_labels)
    fewshot_f1 = inline_fewshot_val_f1(train_embs, train_labels, val_embs, val_labels)
    linear_f1, linear_all = inline_linear_val_f1(train_embs, train_labels, val_embs, val_labels)
    return {
        "linear_val_f1": linear_f1,
        "linear_val_f1_per_hyperparameter": linear_all,
        "knn_val_f1": knn_val_f1,
        "knn_val_f1_per_k": {int(k): float(v) for k, v in knn_all.items()},
        "fewshot_val_f1": fewshot_f1,
        "fewshot_val_f1_per_shot": {FEWSHOT_SHOT: fewshot_f1},
        "selection_split": None,
        "support_draw_seed": THUNDER_PROBE_SEED,
    }


def slide_linear_auc_metrics(embs, labels):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    folds = []
    for tr, va in stratified_folds(labels):
        head = LogisticRegression(C=PATHOBENCH_LR_C, class_weight="balanced", max_iter=SURGEN_LR_MAX_ITER, random_state=0).fit(embs[tr], labels[tr])
        probs = head.predict_proba(embs[va])
        folds.append(float(roc_auc_score(labels[va], probs[:, 1] if probs.shape[1] == 2 else probs, multi_class="ovr", average="macro")))
    return {"val_auc": float(np.mean(folds)), "c": PATHOBENCH_LR_C, "fold_scores": folds}


# Worker entry point launched by queue_probe_job(); owns model loading and probe aggregation.
def worker_probe_transforms(cfg):
    # Frozen baselines set transform_policy explicitly; Nanopath training runs fall back to model.py.
    policy = cfg["probe"].get("transform_policy")
    if policy is None:
        from model import probe_transforms
        return probe_transforms()
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    image = {
        "resize_crop_224": transforms.Compose([transforms.Resize(224, antialias=True), transforms.CenterCrop(224), transforms.ToTensor()]),
        "bicubic224_crop224": transforms.Compose([transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True), transforms.CenterCrop(224), transforms.ToTensor()]),
        "bicubic256_crop224": transforms.Compose([transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True), transforms.CenterCrop(224), transforms.ToTensor()]),
        "square_224": transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor()]),
    }[policy]
    return image, image


def run_probe_job(request_path):
    import importlib
    from model import ViT

    # Fixed seeds keep every head, minibatch order, and support draw stable.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(THUNDER_PROBE_SEED)
    torch.manual_seed(THUNDER_PROBE_SEED)
    torch.cuda.manual_seed_all(THUNDER_PROBE_SEED)
    torch.use_deterministic_algorithms(True)
    probe_started_at = time.monotonic()
    request = json.loads(Path(request_path).read_text())
    classification = list(request["classification_datasets"])
    segmentation = list(request["segmentation_datasets"])
    slide = list(request["slide_datasets"])
    auc = list(request["auc_datasets"])
    survival = list(request["survival_datasets"])
    robustness = list(request["robustness_datasets"])
    print(
        f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
        f"start: {request['job_id']}  checkpoint: {request['checkpoint_path']}",
        flush=True,
    )
    checkpoint = None
    if "config" not in request:
        checkpoint = torch.load(request["checkpoint_path"], map_location="cpu", weights_only=False)
    cfg = request["config"] if "config" in request else checkpoint["config"]
    DATASET_ROOTS.clear()
    DATASET_ROOTS.update({k: Path(v) for k, v in cfg["probe"]["dataset_roots"].items()})
    device = torch.device("cuda")
    if cfg["probe"].get("model_loader"):
        module, fn = cfg["probe"]["model_loader"].split(":")
        model = getattr(importlib.import_module(module), fn)(request["checkpoint_path"], device)
    else:
        if checkpoint is None:
            checkpoint = torch.load(request["checkpoint_path"], map_location="cpu", weights_only=False)
        model = ViT(variant=cfg["model"]["type"]).to(device).eval()
        # Recipes can compare live model weights or EMA weights without changing probe code.
        state_key = {"ema": "model_ema", "model": "model"}[str(cfg["probe"]["model_weights"])]
        model.load_state_dict(checkpoint[state_key], strict=True)
    del checkpoint
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    mean = torch.tensor(cfg["data"]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["data"]["std"], device=device).view(1, 3, 1, 1)
    transform, patch_transform = worker_probe_transforms(cfg)

    # Segmentation runs first because its compiled decoders are sensitive to allocator
    # fragmentation left by the other probes. Every head resets its own seed.
    matmul_precision = torch.get_float32_matmul_precision()
    seg_results = {}
    for dataset in segmentation:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"{console_prefix()} ProbeWorker  [{request['train_step']}]  inline_seg_start: {dataset}", flush=True)
        result, wall = inline_segmentation_f1(model, mean, std, dataset, device, patch_transform)
        result["wall_seconds"] = wall
        seg_results[dataset] = result
        print(
            f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
            f"inline_seg_done: {dataset}  f1={result['seg_val_f1']:.4f}  jaccard={result['seg_val_jaccard']:.4f}  "
            f"epochs={result['epochs']}  wall={wall:.2f}s",
            flush=True,
        )
    gc.collect()
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision(matmul_precision)

    inline_metrics = {}
    for dataset in classification:
        # Thunder-style tile probes share embeddings, then evaluate KNN, SimpleShot, and linear heads.
        print(f"{console_prefix()} ProbeWorker  [{request['train_step']}]  inline_start: {dataset}", flush=True)
        embed_started = time.monotonic()
        train_embs, train_labels = embed_classification_dataset(model, mean, std, dataset, "train", device, transform)
        val_embs, val_labels = embed_classification_dataset(model, mean, std, dataset, "val", device, transform)
        inline_metrics[dataset] = classification_head_metrics(train_embs, train_labels, val_embs, val_labels)
        inline_metrics[dataset]["wall_seconds"] = time.monotonic() - embed_started
        print(
            f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
            f"inline_done: {dataset}  linear_f1={inline_metrics[dataset]['linear_val_f1']:.4f}  knn_f1={inline_metrics[dataset]['knn_val_f1']:.4f}  "
            f"fewshot_f1={inline_metrics[dataset]['fewshot_val_f1']:.4f}  wall={time.monotonic()-embed_started:.2f}s",
            flush=True,
        )

    slide_metrics = {}
    for dataset in slide:
        print(f"{console_prefix()} ProbeWorker  [{request['train_step']}]  inline_slide_start: {dataset}", flush=True)
        embed_started = time.monotonic()
        embs, labels = embed_slide_dataset(model, mean, std, dataset, ("train", "val"), device, patch_transform)
        slide_metrics[dataset] = slide_linear_auc_metrics(embs, labels)
        slide_metrics[dataset]["wall_seconds"] = time.monotonic() - embed_started
        print(f"{console_prefix()} ProbeWorker  [{request['train_step']}]  inline_slide_done: {dataset}  auc={slide_metrics[dataset]['val_auc']:.4f}  c={slide_metrics[dataset]['c']}  wall={time.monotonic()-embed_started:.2f}s", flush=True)

    auc_metrics = {}
    for dataset in auc:
        print(f"{console_prefix()} ProbeWorker  [{request['train_step']}]  inline_auc_start: {dataset}", flush=True)
        result, wall = inline_surgen_ras_auc(model, mean, std, device, patch_transform)
        result["wall_seconds"] = wall
        auc_metrics[dataset] = result
        print(f"{console_prefix()} ProbeWorker  [{request['train_step']}]  inline_auc_done: {dataset}  auc={result['val_auc']:.4f}  c={result['c']}  wall={wall:.2f}s", flush=True)

    survival_metrics = {}
    for dataset in survival:
        print(f"{console_prefix()} ProbeWorker  [{request['train_step']}]  inline_survival_start: {dataset}", flush=True)
        result, wall = inline_pathobench_survival(model, mean, std, dataset, device, patch_transform)
        result["wall_seconds"] = wall
        survival_metrics[dataset] = result
        print(f"{console_prefix()} ProbeWorker  [{request['train_step']}]  inline_survival_done: {dataset}  cindex={result['val_cindex']:.4f}  coxnet_alpha_fractions={result['coxnet_alpha_fractions']}  wall={wall:.2f}s", flush=True)

    rob_indices = {}
    for dataset in robustness:
        print(f"{console_prefix()} ProbeWorker  [{request['train_step']}]  inline_robustness_start: {dataset}", flush=True)
        subset_indices, wall = inline_pathorob(model, mean, std, device, patch_transform)
        rob_indices[dataset] = {
            "subsets": subset_indices,
            "croma": float(sum(v["croma"] for v in subset_indices.values()) / len(subset_indices)),
            "croma_ltm10": float(sum(v["croma_ltm10"] for v in subset_indices.values()) / len(subset_indices)),
            "croma_f0": float(sum(v["croma_f0"] for v in subset_indices.values()) / len(subset_indices)),
            "robustness": float(sum(v["robustness"] for v in subset_indices.values()) / len(subset_indices)),
            "wall_seconds": wall,
        }
        print(
            f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
            f"inline_robustness_done: {dataset}  robustness={rob_indices[dataset]['robustness']:.4f}  "
            f"croma={rob_indices[dataset]['croma']:+.4f}  ltm10={rob_indices[dataset]['croma_ltm10']:+.4f}  "
            f"f0={rob_indices[dataset]['croma_f0']:.4f}  wall={wall:.2f}s",
            flush=True,
        )

    # Aggregate per-dataset metrics into the result file consumed by train.py.
    metrics = {}
    results = {}
    per_dataset_score = {}
    fold_scores = {}
    for dataset in classification:
        metrics[f"probe_{dataset}_linear_val_f1"] = inline_metrics[dataset]["linear_val_f1"]
        metrics[f"probe_{dataset}_knn_val_f1"] = inline_metrics[dataset]["knn_val_f1"]
        metrics[f"probe_{dataset}_fewshot_val_f1"] = inline_metrics[dataset]["fewshot_val_f1"]
        per_dataset_score[dataset] = (
            inline_metrics[dataset]["linear_val_f1"]
            + inline_metrics[dataset]["knn_val_f1"]
            + inline_metrics[dataset]["fewshot_val_f1"]
        ) / 3.0
        results[dataset] = inline_metrics[dataset]
    for dataset in slide:
        metrics[f"probe_{dataset}_val_auc"] = slide_metrics[dataset]["val_auc"]
        metrics[f"probe_{dataset}_c"] = slide_metrics[dataset]["c"]
        per_dataset_score[dataset] = slide_metrics[dataset]["val_auc"]
        fold_scores[dataset] = slide_metrics[dataset]["fold_scores"]
        results[dataset] = slide_metrics[dataset]
    for dataset in segmentation:
        metrics[f"probe_{dataset}_seg_val_f1"] = seg_results[dataset]["seg_val_f1"]
        metrics[f"probe_{dataset}_seg_val_jaccard"] = seg_results[dataset]["seg_val_jaccard"]
        metrics[f"probe_{dataset}_epochs"] = seg_results[dataset]["epochs"]
        per_dataset_score[dataset] = seg_results[dataset]["seg_val_f1"]
        results[dataset] = seg_results[dataset]
    for dataset in auc:
        metrics[f"probe_{dataset}_val_auc"] = auc_metrics[dataset]["val_auc"]
        metrics[f"probe_{dataset}_c"] = auc_metrics[dataset]["c"]
        per_dataset_score[dataset] = auc_metrics[dataset]["val_auc"]
        fold_scores[dataset] = auc_metrics[dataset]["fold_scores"]
        results[dataset] = auc_metrics[dataset]
    for dataset in survival:
        metrics[f"probe_{dataset}_val_cindex"] = survival_metrics[dataset]["val_cindex"]
        metrics[f"probe_{dataset}_coxnet_l1_ratio"] = survival_metrics[dataset]["coxnet_l1_ratio"]
        metrics[f"probe_{dataset}_coxnet_max_iter"] = survival_metrics[dataset]["coxnet_max_iter"]
        metrics[f"probe_{dataset}_coxnet_standardize"] = survival_metrics[dataset]["coxnet_standardize"]
        for fraction, score in survival_metrics[dataset]["val_cindex_per_alpha_fraction"].items():
            metrics[f"probe_{dataset}_val_cindex_alpha_fraction_{fraction.replace('.', 'p')}"] = score
        metrics[f"probe_{dataset}_tiles"] = survival_metrics[dataset]["tiles"]
        metrics[f"probe_{dataset}_tiles_per_slide_cap"] = survival_metrics[dataset]["tiles_per_slide_cap"]
        per_dataset_score[dataset] = survival_metrics[dataset]["val_cindex"]
        fold_scores[dataset] = survival_metrics[dataset]["fold_scores"]
        results[dataset] = survival_metrics[dataset]
    for dataset in robustness:
        for subset, subset_metrics in rob_indices[dataset]["subsets"].items():
            for key, value in subset_metrics.items():
                metrics[f"probe_{dataset}_{subset}_{key}"] = value
        metrics[f"probe_{dataset}_croma"] = rob_indices[dataset]["croma"]
        metrics[f"probe_{dataset}_croma_ltm10"] = rob_indices[dataset]["croma_ltm10"]
        metrics[f"probe_{dataset}_croma_f0"] = rob_indices[dataset]["croma_f0"]
        metrics[f"probe_{dataset}_robustness"] = rob_indices[dataset]["robustness"]
        per_dataset_score[dataset] = rob_indices[dataset]["robustness"]
        results[dataset] = rob_indices[dataset]
    for dataset, score in per_dataset_score.items():
        metrics[f"probe_{dataset}_score"] = score
        metrics[f"probe_{dataset}_wall_seconds"] = results[dataset]["wall_seconds"]
    for dataset, scores in fold_scores.items():
        avg = sum(scores) / len(scores)
        var = sum((x - avg) ** 2 for x in scores) / len(scores)
        metrics[f"probe_{dataset}_fold_var"] = var
        metrics[f"probe_{dataset}_fold_std"] = var ** 0.5

    metrics["linear_mean_f1"] = sum(metrics[f"probe_{d}_linear_val_f1"] for d in classification) / len(classification)
    metrics["knn_mean_f1"] = sum(metrics[f"probe_{d}_knn_val_f1"] for d in classification) / len(classification)
    metrics["fewshot_mean_f1"] = sum(metrics[f"probe_{d}_fewshot_val_f1"] for d in classification) / len(classification)
    metrics["classification_mean_f1"] = sum(metrics[f"probe_{d}_{head}_val_f1"] for d in classification for head in ("linear", "knn", "fewshot")) / (3 * len(classification))
    metrics["slide_mean_auc"] = sum(metrics[f"probe_{d}_val_auc"] for d in slide) / len(slide)
    metrics["seg_mean_f1"] = sum(metrics[f"probe_{d}_seg_val_f1"] for d in segmentation) / len(segmentation)
    metrics["seg_mean_jaccard"] = sum(metrics[f"probe_{d}_seg_val_jaccard"] for d in segmentation) / len(segmentation)
    metrics["auc_mean"] = sum(metrics[f"probe_{d}_val_auc"] for d in auc) / len(auc)
    metrics["survival_mean_cindex"] = sum(metrics[f"probe_{d}_val_cindex"] for d in survival) / len(survival)
    metrics["croma_mean"] = sum(metrics[f"probe_{d}_croma"] for d in robustness) / len(robustness)
    metrics["croma_ltm10_mean"] = sum(metrics[f"probe_{d}_croma_ltm10"] for d in robustness) / len(robustness)
    metrics["croma_f0_mean"] = sum(metrics[f"probe_{d}_croma_f0"] for d in robustness) / len(robustness)
    metrics["robustness_mean"] = sum(metrics[f"probe_{d}_robustness"] for d in robustness) / len(robustness)

    metrics["probe_protocol_version"] = PROBE_PROTOCOL_VERSION
    metrics["final_score"] = sum(weight * metrics[key] for key, weight in SCORE_WEIGHTS)

    print(
        f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
        f"result: final_score={metrics.get('final_score')}  "
        f"linear={metrics.get('linear_mean_f1')}  knn={metrics.get('knn_mean_f1')}  "
        f"fewshot={metrics.get('fewshot_mean_f1')}  classification={metrics.get('classification_mean_f1')}  "
        f"slide={metrics.get('slide_mean_auc')}  seg={metrics.get('seg_mean_f1')}  "
        f"auc={metrics.get('auc_mean')}  survival={metrics.get('survival_mean_cindex')}  "
        f"robustness={metrics.get('robustness_mean')}  "
        f"wall: {time.monotonic() - probe_started_at:.2f}s",
        flush=True,
    )

    Path(request["result_path"]).write_text(
        json.dumps(
            {
                "wall_seconds": time.monotonic() - probe_started_at,
                "job_id": request["job_id"],
                "checkpoint_step": request["checkpoint_step"],
                "train_step": request["train_step"],
                "target_flops": request["target_flops"],
                "target_fraction": request["target_fraction"],
                "checkpoint_path": request["checkpoint_path"],
                "probe_protocol_version": PROBE_PROTOCOL_VERSION,
                "classification_datasets": classification,
                "segmentation_datasets": segmentation,
                "slide_datasets": slide,
                "auc_datasets": auc,
                "survival_datasets": survival,
                "robustness_datasets": robustness,
                "metrics": metrics,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


# train.py call: consume probe result JSONs, log metrics, then delete temporary probe checkpoints.
def collect_probe_results(state, wandb_run, metrics_path):
    state["data"] = json.loads(state["paths"]["state_path"].read_text())
    logged = set(state["data"]["logged_results"])
    for result_path in sorted(state["paths"]["results_dir"].glob("step_*.json")):
        result_path_str = str(result_path)
        result = json.loads(result_path.read_text())
        metrics = {key: float(value) for key, value in result["metrics"].items()}
        checkpoint_path = Path(result["checkpoint_path"])
        if result_path_str in logged:
            continue
        event_payload = {
            "event": "probe",
            "step": result["train_step"],
            "target_flops": result["target_flops"],
            "target_fraction": result["target_fraction"],
            "probe_wall_seconds": float(result["wall_seconds"]),
            **metrics,
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(event_payload) + "\n")
        print(
            f"{console_prefix()} Probe  [{result['train_step']}]  "
            f"log_result: final_score={metrics.get('final_score')}  "
            f"wall={result['wall_seconds']:.2f}s",
            flush=True,
        )
        wandb_payload = {"probe/target_flops": int(result["target_flops"]), "probe/wall_seconds": float(result["wall_seconds"])}
        for key, value in metrics.items():
            wandb_payload[f"probe/{key.removeprefix('probe_')}"] = value
        wandb_run.log(wandb_payload, step=int(result["train_step"]))
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        logged.add(result_path_str)
    state["data"]["logged_results"] = sorted(logged)
    write_probe_state(state)


# Flatten the latest successful probe result into summary.json.
def completed_probe_summary(output_dir):
    summary = {}
    final_result = None
    for result_path in sorted(probe_paths(output_dir)["results_dir"].glob("step_*.json")):
        result = json.loads(result_path.read_text())
        if "final_score" not in result["metrics"]:
            continue
        if final_result is None or int(result["train_step"]) > int(final_result["train_step"]):
            final_result = result
    if final_result is None:
        return summary
    summary["final_probe_step"] = int(final_result["train_step"])
    summary["final_probe_target_flops"] = int(final_result["target_flops"])
    summary["final_probe_target_fraction"] = float(final_result["target_fraction"])
    summary["final_probe_wall_seconds"] = float(final_result["wall_seconds"])
    for key, value in final_result["metrics"].items():
        flat = key.removeprefix("probe_")
        summary[key if key == "final_score" else f"final_probe_{flat}"] = float(value)
    return summary


# CLI entry point for probe subprocesses.
def main():
    if len(sys.argv) != 2:
        raise ValueError("usage: python probe.py <request.json>")
    run_probe_job(sys.argv[1])


if __name__ == "__main__":
    main()
