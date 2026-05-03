# Inline downstream probes. mean_probe_score = unweighted mean of six aggregates: linear, KNN,
# and SimpleShot few-shot F1 over break_his/mhist/pcam; segmentation macro Jaccard over PanNuke,
# MoNuSAC, and CoNSeP from the MaskTransformer head; the PathoROB Kömen et al. 2025 robustness
# index (https://arxiv.org/abs/2507.17845) over camelyon/tolkach_esca; and slide-level AUROC
# averaged over chimera_progression (CHIMERA bladder NMIBC progression) and surgen_extras
# (SurGen SR1482 KRAS/NRAS extended-RAS).
#
# Cls + seg probes follow a four-way train / tune / val / test convention:
#   train: fits the head
#   tune:  selects hyperparameters (LR + WD on linear, k on KNN, LR on seg with WD fixed; epoch
#          is also selected by per-epoch tune-loss/F1 in the training loops)
#   val:   reported once at the chosen state — this feeds mean_probe_score
#   test:  sealed; only computed when cfg.probe.compute_test=true (off by default)
#
# Slide probes (chimera, surgen) use 5-fold StratifiedKFold over (train + tune + val) cases for
# routine reporting (mean of 5 fold AUCs reduces the high single-split AUC noise that small
# n=27/60 vals showed in 3-seed variance experiments). Each fold's C is chosen on a 80/20
# inner split of the rest. The sealed test split (~10%) is preserved and only scored when
# compute_test=true at the modal per-fold C.
#
# CoNSeP is too small to support its own tune split, so its head reuses MoNuSAC's selected lr
# at fixed WD and trains on a fixed schedule. PanNuke train=Fold1, tune+val=Fold2 split 50/50,
# test=Fold3. PathoROB has no fit/eval distinction (unsupervised k-NN). Seg WD is fixed at
# SEGMENTATION_WEIGHT_DECAY (sweeping it added selection noise without improving discrimination
# on tune).
#
# train.py snapshots a probe checkpoint at each FLOP milestone and runs
# this file as a subprocess (`python probe.py req.json`); training pauses, the
# subprocess writes a result JSON, collect_probe_results ingests it back into
# wandb + metrics.jsonl. Inside the subprocess, two threads share one GPU and
# one loaded DinoV2ViT: the main thread loops classification datasets (for
# each, embed train+val with the frozen backbone once, then run all three
# heads — KNN, SimpleShot few-shot, and linear — on those cached embeddings)
# then runs PathoROB (embed each HF dataset once with cls+mean(patch tokens),
# cosine kNN with same-slide neighbours filtered, count SO/OS to form the
# robustness index), while a background thread runs PanNuke. Putting both on
# one GPU helps because classification spends a lot of time in plain
# Python/CPU code which leaves the GPU free to crunch PanNuke.
#
# Rough per-task wall on a 1xH100 leader-recipe checkpoint (the full ViT, not smoke). The seg
# (lr, wd) sweep dominates; cls + slide are short tail. (lr × wd) = 9 combos for linear/seg.
#   break_his   ~30s         9 combo × 200 epochs of linear, plus knn + few-shot
#   mhist       ~32s
#   pcam        ~50s         subsampled
#   PanNuke     ~250s        3 LR × 10-epoch sweep at fixed wd + 30-epoch retrain at chosen LR
#   MoNuSAC     ~12s         smaller dataset, same shape
#   CoNSeP      ~3s          no sweep (reuses monusac's chosen lr at fixed wd), fixed schedule
#   PathoROB    ~10-15s      camelyon + tolkach_esca patches → chunked cosine kNN
#   Chimera     ~25s         pre-extracted tiles → mean-pool → 5-fold CV with C sweep per fold
#   SurGen      ~45s         ~102K pre-extracted tiles → mean-pool → 5-fold CV with C sweep
# PanNuke runs in parallel with the classification loop, so probe wall is roughly
# max(PanNuke, sum of cls + PathoROB) plus a small tail. In practice that's ~5-13 min
# depending on whether the OS page cache for PanNuke's npy folds is warm.

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PROBE_DATA_SPLITS = Path(__file__).resolve().parent / "probe_data_splits"
EMBED_BATCH_SIZE = 128
EMBED_NUM_WORKERS = 4
# Segmentation: LR sweep on tune (short SEG_TUNE_EPOCHS), then retrain at the chosen lr for
# SEGMENTATION_EPOCHS scoring tune every epoch and saving the best state, then report Jaccard
# once on val. WD is fixed (was previously swept across {0, 1e-4, 1e-3} but tune signal could
# not reliably distinguish them — sweep landed on different WDs across seeds and added ~6 pp
# pannuke val Jaccard noise without improving tune-loss separation).
SEGMENTATION_EPOCHS = 30
SEG_TUNE_EPOCHS = 10
SEGMENTATION_LRS = (1e-3, 1e-4, 1e-5)
SEGMENTATION_WEIGHT_DECAY = 1e-4
SEGMENTATION_BATCH_SIZE = 64
PANNUKE_NUM_CLASSES = 6
# train = Fold1, tune+val = Fold2 split deterministically 50/50 (PANNUKE_FOLD2_SPLIT_SEED),
# test = Fold3 (sealed, only run when cfg.probe.compute_test=true).
PANNUKE_FOLDS = {"train": "Fold1", "tune_val": "Fold2", "test": "Fold3"}
PANNUKE_FOLD2_SPLIT_SEED = 1337
# MoNuSAC: 5 nuclei classes (0=bg, 1=epithelial, 2=lymphocyte, 3=macrophage, 4=neutrophil).
# Test ROIs sometimes include class 5="Ambiguous" added in the v2 release; we clip to bg.
MONUSAC_NUM_CLASSES = 5
# CoNSeP: 5 nuclei classes after the standard HoVer-Net consolidation of the original 8-class
# type_map. {3,4}→3 (epithelial: healthy + dysplastic), {5,6,7}→4 (spindle: fibroblast + muscle
# + endothelial). The remap is applied at load time; CONSEP_REMAP[i] = consolidated class id.
# CoNSeP has no tune split (27 train ROIs is too small): the head reuses MoNuSAC's selected
# (lr, wd) and is fit on train, scored on val, with test sealed.
CONSEP_NUM_CLASSES = 5
CONSEP_REMAP = (0, 1, 2, 3, 3, 4, 4, 4)
LINEAR_PROBE_LRS = (1e-3, 1e-4, 1e-5)
LINEAR_PROBE_WEIGHT_DECAYS = (0.0, 1e-4, 1e-3)
LINEAR_PROBE_EPOCHS = 200
LINEAR_PROBE_BATCH_SIZE = 64
PCAM_SUBSET_SEED = 1337
# train + tune both come from HF train (disjoint subsets carved from the same RNG draw); val
# from HF valid; test from HF test (sealed).
PCAM_SUBSET_SIZES = {"train": 3072, "tune": 768, "val": 768, "test": 768}
PCAM_SOURCE_SPLIT = {"train": "train", "tune": "train", "val": "valid", "test": "test"}
FEWSHOT_SHOTS = [1, 2, 4, 8, 16]
FEWSHOT_TRIALS = 1000
FEWSHOT_SEED = 1337
KNN_K_VALS = [1, 3, 5, 10, 20, 30, 40, 50]
KNN_CHUNK_SIZE = 4096
CLASSIFICATION_DATASETS = ["break_his", "mhist", "pcam"]
SEGMENTATION_DATASETS = ["pannuke", "monusac", "consep"]
# PathoROB (Kömen et al. 2025) downstream evaluation. Per-dataset median k_opt are the
# values reported in the preprint. We intentionally exclude PathoROB-tcga: TCGA tiles are
# in our pretraining data, so a TCGA-rooted robustness measurement isn't held-out.
ROBUSTNESS_DATASETS = ["camelyon", "tolkach_esca"]
ROBUSTNESS_K = {"camelyon": 11, "tolkach_esca": 46}
# Slide-level outcome probes. Case-grouped 60/15/15/10 train/tune/val/test splits live in
# probe_data_splits/{chimera,surgen}.json; LR `C` is swept on tune and AUROC reported once
# on val. Test is sealed (only run when cfg.probe.compute_test=true).
SLIDE_DATASETS = ["chimera_progression", "surgen_extras"]
SLIDE_LR_CS = (0.01, 0.1, 1.0, 10.0)
SLIDE_LR_MAX_ITER = 1000
# Module-level so ClassificationDataset / inline_pannuke_jaccard / inline_pathorob_robustness_index
# can read it without threading cfg through every call. Populated from cfg.probe.dataset_roots
# by prepare_probe_state() (train.py main process) and run_probe_job() (probe subprocess).
DATASET_ROOTS = {}


# Prefix probe logs with the same timestamp/job id format as train.py.
def console_prefix():
    return f"{time.strftime('%H:%M:%S')} {os.environ.get('SLURM_JOB_ID', str(os.getpid()))}"


# Keep all probe sidecar files under output_dir/thunder for compatibility with old run layouts.
def probe_paths(output_dir):
    probe_dir = Path(output_dir) / "thunder"
    return {
        "probe_dir": probe_dir,
        "state_path": probe_dir / "state.json",
        "results_dir": probe_dir / "results",
    }


# Probes are enabled only when the recipe asks for them and names at least one task.
def probe_enabled(cfg):
    return bool(cfg["probe"]["enabled"]) and (
        len(cfg["probe"]["datasets"])
        + len(cfg["probe"]["segmentation_datasets"])
        + len(cfg["probe"].get("robustness_datasets", []))
        + len(cfg["probe"].get("slide_datasets", []))
    ) > 0


# Persist probe state so explicitly resumed train.py runs do not relog completed result files.
def write_probe_state(state):
    state["paths"]["state_path"].write_text(json.dumps(state["data"], indent=2) + "\n")


# Validate probe recipe compatibility and initialize the on-disk result tracker.
def prepare_probe_state(cfg, output_dir):
    DATASET_ROOTS.clear()
    DATASET_ROOTS.update({k: Path(v) for k, v in cfg["probe"]["dataset_roots"].items()})
    paths = probe_paths(output_dir)
    for path in [paths["probe_dir"], paths["results_dir"]]:
        path.mkdir(parents=True, exist_ok=True)
    classification = [str(x) for x in cfg["probe"]["datasets"]]
    segmentation = [str(x) for x in cfg["probe"]["segmentation_datasets"]]
    robustness = [str(x) for x in cfg["probe"].get("robustness_datasets", [])]
    slide = [str(x) for x in cfg["probe"].get("slide_datasets", [])]
    compute_test = bool(cfg["probe"].get("compute_test", False))
    data = {
        "version": 17,
        "family": str(cfg["project"]["family"]),
        "classification_datasets": classification,
        "segmentation_datasets": segmentation,
        "robustness_datasets": robustness,
        "slide_datasets": slide,
        "compute_test": compute_test,
        "count": int(cfg["probe"]["count"]),
        "logged_results": [],
    }
    if paths["state_path"].exists():
        # Explicit resume can continue only if the probe family/datasets/count match the old state.
        previous = json.loads(paths["state_path"].read_text())
        if previous["version"] != 17:
            raise ValueError(f"unsupported probe state version: {previous['version']}")
        if previous["family"] != data["family"]:
            raise ValueError(f"probe family changed from {previous['family']} to {data['family']}")
        if previous["classification_datasets"] != data["classification_datasets"]:
            raise ValueError(f"classification datasets changed from {previous['classification_datasets']} to {data['classification_datasets']}")
        if previous["segmentation_datasets"] != data["segmentation_datasets"]:
            raise ValueError(f"segmentation datasets changed from {previous['segmentation_datasets']} to {data['segmentation_datasets']}")
        if previous.get("robustness_datasets", []) != data["robustness_datasets"]:
            raise ValueError(f"robustness datasets changed from {previous.get('robustness_datasets', [])} to {data['robustness_datasets']}")
        if previous.get("slide_datasets", []) != data["slide_datasets"]:
            raise ValueError(f"slide datasets changed from {previous.get('slide_datasets', [])} to {data['slide_datasets']}")
        if previous["count"] != data["count"]:
            raise ValueError(f"probe count changed from {previous['count']} to {data['count']}")
        data["logged_results"] = previous["logged_results"]
    for dataset in classification:
        if dataset not in CLASSIFICATION_DATASETS:
            raise ValueError(f"unsupported classification dataset: {dataset}")
    for dataset in segmentation:
        if dataset not in SEGMENTATION_DATASETS:
            raise ValueError(f"unsupported segmentation dataset: {dataset}")
    for dataset in robustness:
        if dataset not in ROBUSTNESS_DATASETS:
            raise ValueError(f"unsupported robustness dataset: {dataset}")
    for dataset in slide:
        if dataset not in SLIDE_DATASETS:
            raise ValueError(f"unsupported slide dataset: {dataset}")
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
        "classification_datasets": list(state["data"]["classification_datasets"]),
        "segmentation_datasets": list(state["data"]["segmentation_datasets"]),
        "robustness_datasets": list(state["data"]["robustness_datasets"]),
        "slide_datasets": list(state["data"]["slide_datasets"]),
        "compute_test": bool(state["data"].get("compute_test", False)),
        "job_id": f"{slurm_id}-{checkpoint_step:07d}",
    }
    for dataset in request["classification_datasets"] + request["segmentation_datasets"]:
        if not DATASET_ROOTS[dataset].exists():
            raise FileNotFoundError(f"missing dataset root for {dataset}: {DATASET_ROOTS[dataset]}")
    if request["robustness_datasets"] and not DATASET_ROOTS["pathorob"].exists():
        raise FileNotFoundError(f"missing dataset root for pathorob: {DATASET_ROOTS['pathorob']}")
    if request["slide_datasets"] and not DATASET_ROOTS["chimera_tiles"].exists():
        raise FileNotFoundError(f"missing dataset root for chimera_tiles: {DATASET_ROOTS['chimera_tiles']}")
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
        f"robustness: {','.join(request['robustness_datasets']) or '-'}  "
        f"slide: {','.join(request['slide_datasets']) or '-'}",
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
    # Loads images for the classification probes. Splits are train / tune / val / test:
    #   train: fits the head
    #   tune: hyperparameter selection (LR, WD, epoch, k)
    #   val: routine probe metric (reported once)
    #   test: sealed; only loaded when cfg.probe.compute_test=true
    # For pcam we subsample h5 data with deterministic indices (train + tune from HF train,
    # val from HF valid, test from HF test). For break_his / mhist we use the JSON in
    # probe_data_splits/ (regenerated to a stratified train/tune/val/test split keeping the
    # original published test list as the sealed test).
    def __init__(self, dataset, split, transform):
        import h5py
        import numpy as np

        self.transform = transform
        self.dataset = dataset
        if dataset == "pcam":
            pcam_split = PCAM_SOURCE_SPLIT[split]
            with h5py.File(DATASET_ROOTS["pcam"] / f"camelyonpatch_level_2_split_{pcam_split}_x.h5", "r") as fx:
                key_x = next(iter(fx.keys()))
                n_total = fx[key_x].shape[0]
                # train + tune both come from HF train and must be disjoint: sample
                # (size_train + size_tune) once with a fixed seed and partition.
                if split in ("train", "tune"):
                    n_pool = PCAM_SUBSET_SIZES["train"] + PCAM_SUBSET_SIZES["tune"]
                    pool = np.random.default_rng(PCAM_SUBSET_SEED).choice(n_total, size=n_pool, replace=False)
                    idx = np.sort(pool[:PCAM_SUBSET_SIZES["train"]] if split == "train" else pool[PCAM_SUBSET_SIZES["train"]:])
                else:
                    seed = PCAM_SUBSET_SEED + (2 if split == "val" else 3)
                    idx = np.sort(np.random.default_rng(seed).choice(n_total, size=PCAM_SUBSET_SIZES[split], replace=False))
                self.images = np.array(fx[key_x][idx])
            with h5py.File(DATASET_ROOTS["pcam"] / f"camelyonpatch_level_2_split_{pcam_split}_y.h5", "r") as fy:
                self.labels = [int(v) for v in np.array(fy[next(iter(fy.keys()))][idx]).reshape(-1)]
        else:
            splits = json.loads((PROBE_DATA_SPLITS / f"{dataset}.json").read_text())[split]
            self.images = splits["images"]
            self.labels = [int(v) for v in splits["labels"]]
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
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    # Probe embeddings use model.probe_features(), which returns the cls token
    # — none of the DINO/iBOT training heads are involved.
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with autocast:
                e = model.probe_features((x - mean) / std)
            embs.append(e.float().cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(embs, axis=0).astype(np.float32), np.concatenate(labels, axis=0).astype(np.int64)


# Multiclass dice loss for the PanNuke segmentation probe; mask gates invalid pixels.
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
        attn = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]).transpose(1, 2).reshape(b, n, c)
        x = x + self.proj(attn)
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


# Trunc-normal Linear, zero-init bias, identity LayerNorm — Thunder's seg-head init.
def _init_seg_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


# Segmentation decoder vendored from Thunder (thunder/src/thunder/models/task_specific_models.py).
# Project frozen encoder patch tokens into d_model, append n_cls learnable class tokens, run a
# few decoder blocks, then emit low-resolution class masks; inline_pannuke_jaccard upsamples to
# PanNuke label resolution.
class MaskTransformer(nn.Module):
    def __init__(self, n_cls, d_encoder, n_layers=2, n_heads=8, d_model=768, d_ff=3072):
        super().__init__()
        self.n_cls = n_cls
        scale = d_model ** -0.5
        self.proj_dec = nn.Linear(d_encoder, d_model)
        self.blocks = nn.ModuleList(_SegBlock(d_model, n_heads, d_ff) for _ in range(n_layers))
        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_patch = nn.Parameter(scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(scale * torch.randn(d_model, d_model))
        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)
        self.apply(_init_seg_weights)
        nn.init.trunc_normal_(self.cls_emb, std=0.02)

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


# Train a lightweight segmentation head on frozen 256x256-image patch features and report Jaccard.
# Shared by inline_pannuke_jaccard / inline_monusac_jaccard / inline_consep_jaccard.
# train_images/labels: fits the head. tune_images/labels (optional): selects (lr, wd, epoch).
# val_images/labels: reported once at the chosen state. fixed_hp (optional): if given, skip the
# (lr, wd) sweep and use the supplied (lr, wd) directly (CoNSeP path — too small to sweep, reuses
# MoNuSAC's selected hyperparameters). If tune is None, fixed schedule (no per-epoch peek).
# Two-stage when sweeping: SEG_TUNE_EPOCHS short runs over LRS×WDS to pick (lr*, wd*), then full
# SEGMENTATION_EPOCHS retrain with per-epoch tune save. Per-image macro Jaccard is reweighted via
# Thunder's bg-only_weight_test=16 so all-background images don't dominate the mean.
def _seg_head_jaccard(model, mean, std, device, train_images, train_labels, val_images, val_labels, n_cls,
                     tune_images=None, tune_labels=None, fixed_hp=None):
    import numpy as np
    from sklearn.metrics import jaccard_score

    @torch.no_grad()
    def extract(images_np):
        feats = []
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        for i in range(0, len(images_np), SEGMENTATION_BATCH_SIZE):
            batch = torch.from_numpy(np.ascontiguousarray(images_np[i : i + SEGMENTATION_BATCH_SIZE, 16:240, 16:240, :])).permute(0, 3, 1, 2).float().to(device) / 255.0
            with autocast:
                feats.append(model.encode_image((batch - mean) / std)[:, model.registers :].float().cpu())
        return torch.cat(feats, dim=0)

    train_feats = extract(train_images)
    val_feats = extract(val_images)
    tune_feats = extract(tune_images) if (tune_images is not None and len(tune_images) > 0) else None
    d_encoder = train_feats.shape[-1]
    train_labels_t = torch.from_numpy(train_labels)
    val_labels_t = torch.from_numpy(val_labels)
    tune_labels_t = torch.from_numpy(tune_labels) if tune_feats is not None else None
    n = len(train_feats)

    def fresh_head(lr, wd):
        h = MaskTransformer(n_cls=n_cls, d_encoder=d_encoder, n_layers=2, n_heads=8, d_model=768, d_ff=3072).to(device)
        return h, torch.optim.Adam(h.parameters(), lr=lr, weight_decay=wd)

    def train_one_epoch(h, opt):
        h.train()
        perm = torch.randperm(n)
        for i in range(0, n, SEGMENTATION_BATCH_SIZE):
            idx = perm[i : i + SEGMENTATION_BATCH_SIZE]
            labels = train_labels_t[idx].to(device)
            logits = F.interpolate(h(train_feats[idx].to(device)), (256, 256), mode="bilinear")
            loss = multiclass_dice_loss(logits, labels, torch.ones_like(labels, dtype=torch.bool))
            opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def tune_loss_eval(h):
        h.eval()
        s, c = 0.0, 0
        for i in range(0, len(tune_feats), SEGMENTATION_BATCH_SIZE):
            labels = tune_labels_t[i : i + SEGMENTATION_BATCH_SIZE].to(device)
            logits = F.interpolate(h(tune_feats[i : i + SEGMENTATION_BATCH_SIZE].to(device)), (256, 256), mode="bilinear")
            s += multiclass_dice_loss(logits, labels, torch.ones_like(labels, dtype=torch.bool)).item()
            c += 1
        return s / max(1, c)

    # Stage 1 — pick LR on tune at fixed WD, unless overridden by caller (CoNSeP path with
    # fixed_hp) or tune is unavailable. WD-sweep was dropped after empirically observing tune
    # signal couldn't reliably distinguish wd ∈ {0, 1e-4, 1e-3} (sweep added pannuke val
    # Jaccard noise without improving selection quality).
    chosen_wd = SEGMENTATION_WEIGHT_DECAY
    if fixed_hp is not None:
        chosen_lr, chosen_wd = fixed_hp
    elif tune_feats is None:
        chosen_lr = SEGMENTATION_LRS[0]
    else:
        best_loss, chosen_lr = float("inf"), None
        for lr in SEGMENTATION_LRS:
            h, opt = fresh_head(lr, chosen_wd)
            for _ in range(SEG_TUNE_EPOCHS):
                train_one_epoch(h, opt)
            tl = tune_loss_eval(h)
            if tl < best_loss:
                best_loss, chosen_lr = tl, lr

    # Stage 2 — full retrain at chosen hp, save best state by tune loss; fixed-schedule + last
    # epoch when no tune is available.
    head, opt = fresh_head(chosen_lr, chosen_wd)
    best_state = None
    if tune_feats is not None:
        best_loss = float("inf")
        for _ in range(SEGMENTATION_EPOCHS):
            train_one_epoch(head, opt)
            tl = tune_loss_eval(head)
            if tl < best_loss:
                best_loss = tl
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        head.load_state_dict(best_state)
    else:
        for _ in range(SEGMENTATION_EPOCHS):
            train_one_epoch(head, opt)
    head.eval()
    per_image_j, per_image_bg_only = [], []
    with torch.no_grad():
        for i in range(0, len(val_feats), SEGMENTATION_BATCH_SIZE):
            preds = F.interpolate(head(val_feats[i : i + SEGMENTATION_BATCH_SIZE].to(device)), (256, 256), mode="bilinear").argmax(dim=1).cpu().numpy()
            true_chunk = val_labels[i : i + SEGMENTATION_BATCH_SIZE]
            for k in range(preds.shape[0]):
                t = true_chunk[k].reshape(-1)
                p = preds[k].reshape(-1)
                per_image_j.append(jaccard_score(t, p, average="macro", zero_division=0))
                per_image_bg_only.append(bool(t.sum() == 0))
    per_image_j = np.asarray(per_image_j, dtype=np.float64)
    per_image_bg_only = np.asarray(per_image_bg_only)
    freq_bg_only = per_image_bg_only.sum() / len(per_image_bg_only)
    weights = np.ones(len(per_image_j))
    weights[~per_image_bg_only] *= max(1.0, freq_bg_only * 16.0)
    return float(np.average(per_image_j, weights=weights)), (chosen_lr, chosen_wd)


# Load PanNuke npy folds. train = Fold1, tune+val = Fold2 split deterministically 50/50,
# test = Fold3 (sealed; only loaded when compute_test=True).
def inline_pannuke_jaccard(model, mean, std, device, fixed_hp=None, compute_test=False):
    import numpy as np
    started_at = time.monotonic()
    pannuke_root = DATASET_ROOTS["pannuke"]
    fold_dirs = {1: PANNUKE_FOLDS["train"], 2: PANNUKE_FOLDS["tune_val"], 3: PANNUKE_FOLDS["test"]}
    def load_fold(idx, kind):
        return np.load(pannuke_root / f"{fold_dirs[idx]}/{kind}/fold{idx}/{kind}.npy", mmap_mode="r")
    def derive_labels(masks):
        labels = np.zeros((masks.shape[0], 256, 256), dtype=np.int64)
        for j in range(PANNUKE_NUM_CLASSES - 1):
            layer = ((j + 1) * np.clip(masks[..., j], 0, 1)).astype(np.int64)
            labels = np.where(layer != 0, layer, labels)
        return labels
    train_images = load_fold(1, "images")
    train_labels = derive_labels(load_fold(1, "masks"))
    fold2_images = load_fold(2, "images")
    fold2_labels = derive_labels(load_fold(2, "masks"))
    perm = np.random.default_rng(PANNUKE_FOLD2_SPLIT_SEED).permutation(len(fold2_images))
    half = len(perm) // 2
    tune_idx, val_idx = np.sort(perm[:half]), np.sort(perm[half:])
    tune_images, tune_labels = np.asarray(fold2_images[tune_idx]), fold2_labels[tune_idx]
    val_images, val_labels = np.asarray(fold2_images[val_idx]), fold2_labels[val_idx]
    j_val, chosen = _seg_head_jaccard(model, mean, std, device, train_images, train_labels,
                                      val_images, val_labels, PANNUKE_NUM_CLASSES,
                                      tune_images=tune_images, tune_labels=tune_labels, fixed_hp=fixed_hp)
    out = {"val_jaccard": j_val, "chosen_lr": chosen[0], "chosen_wd": chosen[1]}
    if compute_test:
        test_images = np.asarray(load_fold(3, "images"))
        test_labels = derive_labels(load_fold(3, "masks"))
        j_test, _ = _seg_head_jaccard(model, mean, std, device, train_images, train_labels,
                                      test_images, test_labels, PANNUKE_NUM_CLASSES,
                                      tune_images=tune_images, tune_labels=tune_labels, fixed_hp=chosen)
        out["test_jaccard"] = j_test
    return out, time.monotonic() - started_at


# MoNuSAC: 4-way split listed in probe_data_splits/monusac.json. train/tune/val carved from the
# 209-ROI train directory (random stratified 70/15/15 per dataset gen-time). test = the official
# 85-ROI test directory (sealed; only run with compute_test=True). All ROIs resized to 256x256.
# Class 5 ("Ambiguous") only appears in the test set in the v2 release; we clip to [0, 4].
# Note: every MoNuSAC slide ID is TCGA-derived, so per-WSI biology overlaps with our pretraining
# universe. We accept this for parity with PanNuke since dense pixel-level segmentation is much
# harder to leak through self-supervised pretraining than k-NN.
def inline_monusac_jaccard(model, mean, std, device, fixed_hp=None, compute_test=False):
    import numpy as np
    from PIL import Image
    started_at = time.monotonic()
    monusac_root = DATASET_ROOTS["monusac"]
    splits = json.loads((PROBE_DATA_SPLITS / "monusac.json").read_text())
    def load_paths(rel_paths, root):
        imgs, labels = [], []
        for rel in rel_paths:
            tif = root / rel
            imgs.append(np.asarray(Image.open(tif).convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.uint8))
            lbl = np.asarray(Image.fromarray(np.load(tif.with_suffix(".npy"), allow_pickle=True).astype(np.uint8)).resize((256, 256), Image.NEAREST), dtype=np.int64)
            labels.append(np.clip(lbl, 0, MONUSAC_NUM_CLASSES - 1))
        return np.stack(imgs), np.stack(labels)
    tr_root = monusac_root / "MoNuSAC_images_and_annotations"
    te_root = monusac_root / "MoNuSAC Testing Data and Annotations"
    train_images, train_labels = load_paths(splits["train"], tr_root)
    tune_images, tune_labels = load_paths(splits["tune"], tr_root)
    val_images, val_labels = load_paths(splits["val"], tr_root)
    j_val, chosen = _seg_head_jaccard(model, mean, std, device, train_images, train_labels,
                                      val_images, val_labels, MONUSAC_NUM_CLASSES,
                                      tune_images=tune_images, tune_labels=tune_labels, fixed_hp=fixed_hp)
    out = {"val_jaccard": j_val, "chosen_lr": chosen[0], "chosen_wd": chosen[1]}
    if compute_test:
        test_images, test_labels = load_paths(splits["test"], te_root)
        j_test, _ = _seg_head_jaccard(model, mean, std, device, train_images, train_labels,
                                      test_images, test_labels, MONUSAC_NUM_CLASSES,
                                      tune_images=tune_images, tune_labels=tune_labels, fixed_hp=chosen)
        out["test_jaccard"] = j_test
    return out, time.monotonic() - started_at


# CoNSeP (Graham et al. 2019, UHCW colorectal — non-TCGA): 27 train + 14 test 1000x1000 H&E
# images. Splits live in probe_data_splits/consep.json: 22 train, 5 val carved from the train
# directory; 14 test in the Test directory (sealed). NO tune split — too small to sweep on,
# so the head reuses MoNuSAC's selected (lr, wd) via the fixed_hp argument.
def inline_consep_jaccard(model, mean, std, device, fixed_hp=None, compute_test=False):
    import numpy as np
    import scipy.io as sio
    from PIL import Image
    started_at = time.monotonic()
    consep_root = DATASET_ROOTS["consep"]
    remap = np.array(CONSEP_REMAP, dtype=np.int64)
    splits = json.loads((PROBE_DATA_SPLITS / "consep.json").read_text())
    def load_pngs(names, split_dir):
        imgs, labels = [], []
        for name in names:
            png = split_dir / "Images" / name
            mat = sio.loadmat(split_dir / "Labels" / (Path(name).stem + ".mat"))
            imgs.append(np.asarray(Image.open(png).convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.uint8))
            labels.append(remap[np.asarray(Image.fromarray(mat["type_map"].astype(np.uint8)).resize((256, 256), Image.NEAREST), dtype=np.int64)])
        return np.stack(imgs), np.stack(labels)
    train_images, train_labels = load_pngs(splits["train"], consep_root / "Train")
    val_images, val_labels = load_pngs(splits["val"], consep_root / "Train")
    j_val, chosen = _seg_head_jaccard(model, mean, std, device, train_images, train_labels,
                                      val_images, val_labels, CONSEP_NUM_CLASSES, fixed_hp=fixed_hp)
    out = {"val_jaccard": j_val, "chosen_lr": chosen[0], "chosen_wd": chosen[1]}
    if compute_test:
        test_images, test_labels = load_pngs(splits["test"], consep_root / "Test")
        j_test, _ = _seg_head_jaccard(model, mean, std, device, train_images, train_labels,
                                      test_images, test_labels, CONSEP_NUM_CLASSES, fixed_hp=chosen)
        out["test_jaccard"] = j_test
    return out, time.monotonic() - started_at


# PathoROB robustness index (Kömen et al. 2025, https://arxiv.org/abs/2507.17845).
# Embed each PathoROB dataset once with cls + mean(patch tokens), L2-normalize, then
# compute robustness_index = SO / (SO + OS) at the per-dataset median k_opt from the
# preprint, where SO = same biological / other medical center and OS = other biological /
# same medical center counted over cosine k-NN neighbours. Same-slide neighbours are
# filtered out (patches from one slide are near-duplicates) before taking the top k_opt.
# We exclude the benchmark's TCGA dataset (tcga_2x2 paired-evaluation cell) because TCGA
# tiles are in our pretraining universe — both remaining datasets (camelyon, tolkach_esca)
# are unpaired single-group evaluations. cls + mean is the benchmark's standard `_clsmean`
# pooling — what every model on the leaderboard uses.
def inline_pathorob_robustness_index(model, mean, std, device, transform, datasets):
    import io, numpy as np, pyarrow as pa, pyarrow.parquet as pq
    from PIL import Image

    started_at = time.monotonic()
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    pathorob_root = DATASET_ROOTS["pathorob"]

    # In-memory image dataset over a pa.Table's "image" column (each row is {bytes, path}).
    # to_pylist() materializes the bytes once so DataLoader workers can pickle them cheaply.
    class _PathoRobDataset(torch.utils.data.Dataset):
        def __init__(self, table):
            self.bytes_list = [r["bytes"] for r in table.column("image").to_pylist()]
        def __len__(self): return len(self.bytes_list)
        def __getitem__(self, i):
            return transform(Image.open(io.BytesIO(self.bytes_list[i])).convert("RGB"))

    out = {}
    for name in datasets:
        files = sorted((pathorob_root / name).glob("data/*.parquet"))
        tbl = pq.read_table(files[0]) if len(files) == 1 else pa.concat_tables([pq.read_table(f) for f in files])
        meta = tbl.select(["slide_id", "biological_class", "medical_center"]).to_pandas()
        n = len(meta)
        loader = torch.utils.data.DataLoader(_PathoRobDataset(tbl), batch_size=EMBED_BATCH_SIZE,
                                             num_workers=EMBED_NUM_WORKERS, pin_memory=True, shuffle=False)
        embs = []
        with torch.no_grad():
            for batch in loader:
                x = batch.to(device, non_blocking=True)
                with autocast:
                    o = model((x - mean) / std)
                    feat = torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(dim=1)], dim=-1)
                embs.append(feat.float().cpu().numpy())
        embs = np.concatenate(embs).astype(np.float32)
        embs /= np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
        embs_t = torch.from_numpy(embs).to(device)
        # to_numpy(dtype=object) so 2D fancy indexing (bi[topk], etc.) works downstream;
        # pandas/Arrow string-backed arrays only support 1D take.
        sl = meta.slide_id.to_numpy(dtype=object)
        bi = meta.biological_class.to_numpy(dtype=object)
        ce = meta.medical_center.to_numpy(dtype=object)
        k_target = ROBUSTNESS_K[name]
        # Pull k_target + margin neighbours so same-slide filtering still leaves k_target.
        margin = int(np.unique(sl, return_counts=True)[1].max())
        k = min(k_target + margin, n - 1)
        SO = OS = 0
        for s in range(0, n, KNN_CHUNK_SIZE):
            e = min(s + KNN_CHUNK_SIZE, n)
            sim = embs_t[s:e] @ embs_t.T
            # Exclude the query from its own neighbours.
            sim[torch.arange(e - s, device=device), torch.arange(s, e, device=device)] = -float("inf")
            topk = torch.topk(sim, k, dim=1).indices.cpu().numpy()
            qi = np.arange(s, e)
            bm = bi[topk] == bi[qi][:, None]
            cm = ce[topk] == ce[qi][:, None]
            # keep = first k_target neighbours per row whose slide differs from the query.
            ns = sl[topk] != sl[qi][:, None]
            keep = ns & (np.cumsum(ns, axis=1) <= k_target)
            SO += int(((bm & ~cm) & keep).sum())
            OS += int(((~bm & cm) & keep).sum())
        out[name] = SO / max(1, SO + OS)
    return out, time.monotonic() - started_at


# Slide-level LR probe shared by chimera and surgen. Embeds every tile, mean-pools per slide,
# then runs 5-fold StratifiedKFold over (train + tune + val) cases for a CV-mean AUROC report.
# Each outer fold's "rest" is split 80/20 into inner-train + inner-tune for C selection at that
# fold; final per-fold model refits on the entire rest at the chosen C and scores the held-out
# fold once. Reports the mean and std across the 5 fold AUCs (lower variance than the prior
# single-split design where each report was a single n=27 [chimera] or n=60 [surgen] AUC).
# Test stays sealed and is only scored when compute_test=true: a final LR is fit on the full
# pool at the modal per-fold C and scored once on the test split.
SLIDE_CV_FOLDS = 5
SLIDE_CV_SEED = 1337
SLIDE_INNER_TUNE_FRACTION = 0.20


def _slide_cv_auc(model, mean, std, device, transform, parquet_root, slide_id_col, label_col, splits, dataset_name, compute_test):
    import io, numpy as np, pyarrow as pa, pyarrow.parquet as pq
    from collections import Counter
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, train_test_split

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    files = sorted((parquet_root / "data").glob(f"{dataset_name}-*.parquet"))
    tbl = pq.read_table(files[0]) if len(files) == 1 else pa.concat_tables([pq.read_table(f) for f in files])
    meta = tbl.select([slide_id_col, label_col]).to_pandas()

    class _SlideDataset(torch.utils.data.Dataset):
        def __init__(self, table):
            self.bytes_list = table.column("jpeg").to_pylist()
        def __len__(self): return len(self.bytes_list)
        def __getitem__(self, i):
            return transform(Image.open(io.BytesIO(self.bytes_list[i])).convert("RGB"))
    loader = torch.utils.data.DataLoader(_SlideDataset(tbl), batch_size=EMBED_BATCH_SIZE,
                                         num_workers=EMBED_NUM_WORKERS, pin_memory=True, shuffle=False)
    embs = []
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device, non_blocking=True)
            with autocast:
                o = model((x - mean) / std)
                feat = torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(dim=1)], dim=-1)
            embs.append(feat.float().cpu().numpy())
    embs = np.concatenate(embs).astype(np.float32)
    slide_ids = meta[slide_id_col].to_numpy(dtype=object)
    labels = meta[label_col].to_numpy()
    pool = {}
    for sid in np.unique(slide_ids):
        m = slide_ids == sid
        pool[sid] = (embs[m].mean(axis=0), int(labels[m][0]))

    def stack(ids):
        X = np.stack([pool[s][0] for s in ids])
        y = np.asarray([pool[s][1] for s in ids])
        return X, y

    pool_ids = np.asarray(splits["train"] + splits["tune"] + splits["val"], dtype=object)
    X_pool, y_pool = stack(pool_ids)
    skf = StratifiedKFold(n_splits=SLIDE_CV_FOLDS, shuffle=True, random_state=SLIDE_CV_SEED)
    fold_aucs, fold_Cs = [], []
    for rest_idx, val_idx in skf.split(X_pool, y_pool):
        X_rest, y_rest = X_pool[rest_idx], y_pool[rest_idx]
        # Inner 80/20 split for C selection on tune-AUC; final fit refits on the whole rest.
        X_in_tr, X_in_tu, y_in_tr, y_in_tu = train_test_split(
            X_rest, y_rest, test_size=SLIDE_INNER_TUNE_FRACTION, random_state=SLIDE_CV_SEED, stratify=y_rest)
        best_C, best_tune = None, -1.0
        for C in SLIDE_LR_CS:
            clf = LogisticRegression(C=C, max_iter=SLIDE_LR_MAX_ITER).fit(X_in_tr, y_in_tr)
            tune_auc = float(roc_auc_score(y_in_tu, clf.predict_proba(X_in_tu)[:, 1]))
            if tune_auc > best_tune:
                best_tune, best_C = tune_auc, C
        clf = LogisticRegression(C=best_C, max_iter=SLIDE_LR_MAX_ITER).fit(X_rest, y_rest)
        fold_aucs.append(float(roc_auc_score(y_pool[val_idx], clf.predict_proba(X_pool[val_idx])[:, 1])))
        fold_Cs.append(best_C)
    out = {
        "val_auc": float(np.mean(fold_aucs)),
        "val_auc_std": float(np.std(fold_aucs, ddof=1)),
        "fold_aucs": [float(a) for a in fold_aucs],
        "chosen_Cs": fold_Cs,
    }
    if compute_test:
        from collections import Counter
        final_C = Counter(fold_Cs).most_common(1)[0][0]
        clf = LogisticRegression(C=final_C, max_iter=SLIDE_LR_MAX_ITER).fit(X_pool, y_pool)
        Xte, yte = stack(splits["test"])
        out["test_auc"] = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
        out["test_C"] = final_C
    return out


# Chimera task-3 progression slide-level probe. The challenge's actual test cohort is hidden by
# the organizers (3A + 3B are both training data per DIAGNijmegen's official baseline). We carve
# a small sealed test (~10%) ourselves so contributors can iterate without touching it; routine
# reporting uses 5-fold CV over the rest of the cohort.
def inline_chimera_progression_auc(model, mean, std, device, transform, compute_test=False):
    started_at = time.monotonic()
    splits = json.loads((PROBE_DATA_SPLITS / "chimera.json").read_text())
    out = _slide_cv_auc(model, mean, std, device, transform, DATASET_ROOTS["chimera_tiles"],
                       "slide_id", "progression", splits, "chimera", compute_test)
    return {"chimera_progression": out}, time.monotonic() - started_at


# SurGen extended-RAS slide-level probe (KRAS or NRAS mutation, SR1482 cohort). Same 5-fold CV
# shape as chimera; pooled by case_id since our cohort has one canonical slide per case.
def inline_surgen_extras_auc(model, mean, std, device, transform, compute_test=False):
    started_at = time.monotonic()
    splits = json.loads((PROBE_DATA_SPLITS / "surgen.json").read_text())
    out = _slide_cv_auc(model, mean, std, device, transform, DATASET_ROOTS["surgen_tiles"],
                       "case_id", "ras", splits, "surgen", compute_test)
    return {"surgen_extras": out}, time.monotonic() - started_at


# KNN probe: cosine kNN is implemented with normalized dot products in chunks to cap memory.
# Sweep k on tune to pick best_k, then report val F1 at best_k once. Returns the per-k tune
# F1 dict for diagnostics alongside the single val F1 at the chosen k.
def inline_knn_val_f1(train_embs, train_labels, tune_embs, tune_labels, val_embs, val_labels, k_vals):
    import numpy as np
    from sklearn.metrics import f1_score

    train_n = train_embs.astype(np.float32, copy=False) / np.linalg.norm(train_embs, axis=1, keepdims=True)
    def predict(target_embs, ks):
        target_n = target_embs.astype(np.float32, copy=False) / np.linalg.norm(target_embs, axis=1, keepdims=True)
        preds_per_k = {k: [] for k in ks}
        for start in range(0, len(target_n), KNN_CHUNK_SIZE):
            chunk = target_n[start : start + KNN_CHUNK_SIZE]
            sim = chunk @ train_n.T
            order = np.argsort(-sim, axis=1)
            for i in range(len(chunk)):
                row = train_labels[order[i]]
                for k in ks:
                    preds_per_k[k].append(int(np.bincount(row[:k]).argmax()))
        return preds_per_k
    tune_preds = predict(tune_embs, k_vals)
    tune_f1_per_k = {k: float(f1_score(tune_labels, tune_preds[k], average="macro")) for k in k_vals}
    best_k = max(tune_f1_per_k, key=lambda k: tune_f1_per_k[k])
    val_pred = predict(val_embs, [best_k])[best_k]
    val_f1 = float(f1_score(val_labels, val_pred, average="macro"))
    return best_k, val_f1, tune_f1_per_k


# SimpleShot-style few-shot probe: class prototypes from random support sets, voted over trials.
def inline_fewshot_val_f1(train_embs, train_labels, val_embs, val_labels, shots, trials, seed):
    import numpy as np
    from sklearn.metrics import f1_score

    train_embs = train_embs.astype(np.float32, copy=False)
    val_embs = val_embs.astype(np.float32, copy=False)
    label_to_idx = defaultdict(list)
    for i, label in enumerate(train_labels):
        label_to_idx[int(label)].append(i)
    sorted_labels = sorted(label_to_idx)
    rng = np.random.default_rng(seed)
    f1_per_shot = {}
    for shot in shots:
        # Many trials reduce support-set noise; final prediction is a per-example majority vote.
        trial_preds = np.zeros((trials, len(val_labels)), dtype=np.int64)
        for trial in range(trials):
            support_idx = []
            support_lbl = []
            for label in sorted_labels:
                picks = rng.choice(label_to_idx[label], size=shot, replace=False)
                support_idx.extend(picks.tolist())
                support_lbl.extend([label] * shot)
            support = train_embs[support_idx]
            support_lbl_arr = np.asarray(support_lbl)
            mean = support.mean(axis=0)
            support_centered = support - mean
            cls_embs = np.stack([support_centered[support_lbl_arr == l].mean(axis=0) for l in sorted_labels])
            cls_n = cls_embs / np.linalg.norm(cls_embs, axis=1, keepdims=True)
            val_centered = val_embs - mean
            val_n = val_centered / np.linalg.norm(val_centered, axis=1, keepdims=True)
            sim = val_n @ cls_n.T
            trial_preds[trial] = np.asarray(sorted_labels)[sim.argmax(axis=1)]
        final = np.array([np.bincount(trial_preds[:, i]).argmax() for i in range(trial_preds.shape[1])])
        f1_per_shot[shot] = float(f1_score(val_labels, final, average="macro"))
    return f1_per_shot


# Linear probe: train a head on train, sweep (lr, wd, epoch) selecting on tune F1, then report
# val F1 once at the chosen state. The (lr, wd) grid is LINEAR_PROBE_LRS × LINEAR_PROBE_WEIGHT_DECAYS;
# epoch is implicitly swept by checking tune F1 every epoch and saving the best state.
def inline_linear_val_f1(train_embs, train_labels, tune_embs, tune_labels, val_embs, val_labels):
    import numpy as np
    from sklearn.metrics import f1_score
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(np.max(train_labels)) + 1
    train_embs_t = torch.from_numpy(train_embs).to(device)
    train_labels_t = torch.from_numpy(train_labels).long().to(device)
    tune_embs_t = torch.from_numpy(tune_embs).to(device)
    val_embs_t = torch.from_numpy(val_embs).to(device)
    n = len(train_embs_t)
    best_tune_f1, best_state = -1.0, None
    for lr in LINEAR_PROBE_LRS:
        for wd in LINEAR_PROBE_WEIGHT_DECAYS:
            head = nn.Linear(train_embs.shape[1], num_classes).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
            for _ in range(LINEAR_PROBE_EPOCHS):
                perm = torch.randperm(n, device=device)
                for i in range(0, n, LINEAR_PROBE_BATCH_SIZE):
                    idx = perm[i : i + LINEAR_PROBE_BATCH_SIZE]
                    loss = F.cross_entropy(head(train_embs_t[idx]), train_labels_t[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    preds = head(tune_embs_t).argmax(-1).cpu().numpy()
                tune_f1 = float(f1_score(tune_labels, preds, average="macro"))
                if tune_f1 > best_tune_f1:
                    best_tune_f1 = tune_f1
                    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head = nn.Linear(train_embs.shape[1], num_classes).to(device)
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        val_preds = head(val_embs_t).argmax(-1).cpu().numpy()
    return float(f1_score(val_labels, val_preds, average="macro"))


# Worker entry point launched by queue_probe_job(); owns model loading and probe aggregation.
def run_probe_job(request_path):
    from torchvision import transforms
    from model import DinoV2ViT

    probe_started_at = time.monotonic()
    request = json.loads(Path(request_path).read_text())
    classification = list(request["classification_datasets"])
    segmentation = list(request["segmentation_datasets"])
    robustness = list(request.get("robustness_datasets", []))
    slide = list(request.get("slide_datasets", []))
    compute_test = bool(request.get("compute_test", False))
    print(
        f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
        f"start: {request['job_id']}  checkpoint: {request['checkpoint_path']}",
        flush=True,
    )
    checkpoint = torch.load(request["checkpoint_path"], map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    DATASET_ROOTS.clear()
    DATASET_ROOTS.update({k: Path(v) for k, v in cfg["probe"]["dataset_roots"].items()})
    # Recipes can compare live model weights or EMA weights without changing probe code.
    state_key = {"ema": "model_ema", "model": "model"}[str(cfg["probe"]["model_weights"])]
    model_state = checkpoint[state_key]
    del checkpoint
    device = torch.device("cuda")
    model = DinoV2ViT(variant=cfg["model"]["type"]).to(device).eval()
    model.load_state_dict(model_state, strict=True)
    for param in model.parameters():
        param.requires_grad = False
    mean = torch.tensor(cfg["data"]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["data"]["std"], device=device).view(1, 3, 1, 1)
    transform = transforms.Compose([transforms.Resize(256, antialias=True), transforms.CenterCrop(224), transforms.ToTensor()])

    # Segmentation overlaps with the classification loop on the same GPU and the
    # same loaded DinoV2ViT. Both paths only read the backbone (no .train()/.eval()
    # flips, all backbone forwards are no_grad), and PanNuke trains its own
    # MaskTransformer head with its own optimizer, so sharing the module is safe.
    # CUDA kernels still serialize on the default stream; the win comes from
    # overlapping CPU-side work (PIL decode, NumPy SimpleShot, sklearn F1) with
    # PanNuke's GPU work, plus segmentation's mmap'd PanNuke loads.
    # Segmentation runs in a background thread alongside the classification loop. monusac runs
    # before consep so consep can reuse monusac's selected (lr, wd) — too few CoNSeP ROIs to
    # support its own sweep. PanNuke also sweeps independently.
    seg_results = {}
    def run_segmentation():
        chosen_for_consep = None
        for dataset in segmentation:
            print(f"{console_prefix()} ProbeWorker  [{request['train_step']}]  inline_seg_start: {dataset}", flush=True)
            if dataset == "consep":
                runner = inline_consep_jaccard
                fixed_hp = chosen_for_consep
            elif dataset == "monusac":
                runner = inline_monusac_jaccard
                fixed_hp = None
            elif dataset == "pannuke":
                runner = inline_pannuke_jaccard
                fixed_hp = None
            else:
                raise ValueError(f"unsupported seg dataset: {dataset}")
            res, seg_wall = runner(model, mean, std, device, fixed_hp=fixed_hp, compute_test=compute_test)
            seg_results[dataset] = res
            if dataset == "monusac":
                chosen_for_consep = (res["chosen_lr"], res["chosen_wd"])
            print(
                f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
                f"inline_seg_done: {dataset}  jaccard={res['val_jaccard']:.4f}  "
                f"chosen_lr={res['chosen_lr']:.0e}  chosen_wd={res['chosen_wd']}  wall={seg_wall:.2f}s",
                flush=True,
            )
    seg_executor = ThreadPoolExecutor(max_workers=1) if segmentation else None
    seg_future = seg_executor.submit(run_segmentation) if seg_executor is not None else None

    inline_metrics = {}
    for dataset in classification:
        # Classification probes share embeddings, then evaluate KNN, SimpleShot, and linear heads.
        embed_started = time.monotonic()
        train_embs, train_labels = embed_classification_dataset(model, mean, std, dataset, "train", device, transform)
        tune_embs, tune_labels = embed_classification_dataset(model, mean, std, dataset, "tune", device, transform)
        val_embs, val_labels = embed_classification_dataset(model, mean, std, dataset, "val", device, transform)
        knn_best_k, knn_best_f1, knn_tune_all = inline_knn_val_f1(train_embs, train_labels, tune_embs, tune_labels, val_embs, val_labels, KNN_K_VALS)
        fewshot_per_shot = inline_fewshot_val_f1(train_embs, train_labels, val_embs, val_labels, FEWSHOT_SHOTS, FEWSHOT_TRIALS, FEWSHOT_SEED + CLASSIFICATION_DATASETS.index(dataset))
        linear_f1 = inline_linear_val_f1(train_embs, train_labels, tune_embs, tune_labels, val_embs, val_labels)
        inline_metrics[dataset] = {
            "linear_val_f1": linear_f1,
            "knn_best_k": knn_best_k,
            "knn_val_f1": knn_best_f1,
            "knn_tune_f1_per_k": {int(k): float(v) for k, v in knn_tune_all.items()},
            "fewshot_val_f1": float(sum(fewshot_per_shot.values()) / len(fewshot_per_shot)),
            "fewshot_val_f1_per_shot": {int(s): float(v) for s, v in fewshot_per_shot.items()},
        }
        print(
            f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
            f"inline_done: {dataset}  linear_f1={linear_f1:.4f}  knn_f1={knn_best_f1:.4f}  "
            f"fewshot_f1={inline_metrics[dataset]['fewshot_val_f1']:.4f}  wall={time.monotonic()-embed_started:.2f}s",
            flush=True,
        )

    # Robustness runs in the main thread after classification — it's GPU-bound, so chaining
    # it after classification keeps it overlapped with the segmentation background thread.
    rob_indices = {}
    if robustness:
        rob_started = time.monotonic()
        rob_indices = inline_pathorob_robustness_index(model, mean, std, device, transform, robustness)[0]
        for d, v in rob_indices.items():
            print(
                f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
                f"inline_robustness_done: {d}  robustness_index={v:.4f}",
                flush=True,
            )
        print(f"{console_prefix()} ProbeWorker  [{request['train_step']}]  inline_robustness_total_wall={time.monotonic()-rob_started:.2f}s", flush=True)

    # Slide-level outcome probes — GPU-bound, run after PathoROB. Each returns {dataset_name:
    # {val_auc (CV mean), val_auc_std (across folds), fold_aucs, chosen_Cs, [test_auc, test_C]}}.
    slide_runners = {"chimera_progression": inline_chimera_progression_auc, "surgen_extras": inline_surgen_extras_auc}
    slide_results = {}
    for dataset in slide:
        result, slide_wall = slide_runners[dataset](model, mean, std, device, transform, compute_test=compute_test)
        slide_results.update(result)
        for d, r in result.items():
            print(
                f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
                f"inline_slide_done: {d}  val_auc={r['val_auc']:.4f}±{r['val_auc_std']:.4f}  chosen_Cs={r['chosen_Cs']}  wall={slide_wall:.2f}s",
                flush=True,
            )

    if seg_future is not None:
        # .result() re-raises any exception from the segmentation thread so the probe job fails loudly.
        seg_future.result()
        seg_executor.shutdown()

    # Aggregate per-dataset metrics into the result file consumed by train.py. `_val_*` keys
    # are the routine probe metrics that feed mean_probe_score; `_test_*` keys (when present)
    # are the sealed-test scores produced only when cfg.probe.compute_test=true.
    metrics = {}
    results = {}
    for dataset in classification:
        metrics[f"probe_{dataset}_linear_val_f1"] = inline_metrics[dataset]["linear_val_f1"]
        metrics[f"probe_{dataset}_knn_val_f1"] = inline_metrics[dataset]["knn_val_f1"]
        metrics[f"probe_{dataset}_fewshot_val_f1"] = inline_metrics[dataset]["fewshot_val_f1"]
        results[dataset] = inline_metrics[dataset]
    for dataset in segmentation:
        r = seg_results[dataset]
        metrics[f"probe_{dataset}_seg_val_jaccard"] = r["val_jaccard"]
        if "test_jaccard" in r:
            metrics[f"probe_{dataset}_seg_test_jaccard"] = r["test_jaccard"]
        results[dataset] = r
    for dataset in robustness:
        metrics[f"probe_{dataset}_robustness_index"] = rob_indices[dataset]
        results[dataset] = {"robustness_index": rob_indices[dataset]}
    for dataset in slide:
        r = slide_results[dataset]
        metrics[f"probe_{dataset}_val_auc"] = r["val_auc"]
        if "test_auc" in r:
            metrics[f"probe_{dataset}_test_auc"] = r["test_auc"]
        results[dataset] = r

    # Mean probe score is the main model-selection signal across classification, segmentation,
    # robustness, and slide-level tasks. Each task category contributes one mean to the unweighted average.
    if len(classification) > 0:
        metrics["linear_mean_f1"] = sum(metrics[f"probe_{d}_linear_val_f1"] for d in classification) / len(classification)
        metrics["knn_mean_f1"] = sum(metrics[f"probe_{d}_knn_val_f1"] for d in classification) / len(classification)
        metrics["fewshot_mean_f1"] = sum(metrics[f"probe_{d}_fewshot_val_f1"] for d in classification) / len(classification)
    if len(segmentation) > 0:
        metrics["seg_mean_jaccard"] = sum(metrics[f"probe_{d}_seg_val_jaccard"] for d in segmentation) / len(segmentation)
    if len(robustness) > 0:
        metrics["robustness_mean"] = sum(metrics[f"probe_{d}_robustness_index"] for d in robustness) / len(robustness)
    if len(slide) > 0:
        metrics["slide_mean_auc"] = sum(metrics[f"probe_{d}_val_auc"] for d in slide) / len(slide)

    task_means = [metrics[k] for k in ("linear_mean_f1", "knn_mean_f1", "fewshot_mean_f1", "seg_mean_jaccard", "robustness_mean", "slide_mean_auc") if k in metrics]
    metrics["mean_probe_score"] = sum(task_means) / len(task_means)

    print(
        f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
        f"result: mean_probe_score={metrics['mean_probe_score']:.6f}  "
        f"linear={metrics.get('linear_mean_f1')}  knn={metrics.get('knn_mean_f1')}  "
        f"fewshot={metrics.get('fewshot_mean_f1')}  seg={metrics.get('seg_mean_jaccard')}  "
        f"robustness={metrics.get('robustness_mean')}  "
        f"slide={metrics.get('slide_mean_auc')}  "
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
                "classification_datasets": classification,
                "segmentation_datasets": segmentation,
                "robustness_datasets": robustness,
                "slide_datasets": slide,
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
            f"log_result: mean_probe_score={metrics.get('mean_probe_score')}  "
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


# Flatten the latest successful probe result into summary.json final_probe_* keys.
def completed_probe_summary(output_dir):
    summary = {}
    final_result = None
    for result_path in sorted(probe_paths(output_dir)["results_dir"].glob("step_*.json")):
        result = json.loads(result_path.read_text())
        if "mean_probe_score" not in result["metrics"]:
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
        flat = "score" if key == "mean_probe_score" else key.removeprefix("probe_")
        summary[f"final_probe_{flat}"] = float(value)
    return summary


# CLI entry point for probe subprocesses.
def main():
    if len(sys.argv) != 2:
        raise ValueError("usage: python probe.py <request.json>")
    run_probe_job(sys.argv[1])


if __name__ == "__main__":
    main()
