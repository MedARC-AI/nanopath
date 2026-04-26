# Inline downstream evaluation. This is the main comparison signal between
# pretraining recipes: mean F1 across five classification datasets
# (bach, bracs, break_his, mhist, pcam — KNN + SimpleShot few-shot + linear)
# plus pannuke segmentation (MaskTransformer head from seg_head.py,
# trained with multiclass dice loss, scored by per-image macro Jaccard).
#
# train.py rank-0 calls queue_probe_job at FLOP milestones; that snapshots the
# checkpoint and re-execs this file as a subprocess (`python probe.py req.json`)
# so the probe gets a clean GPU. The subprocess writes a result JSON, then
# collect_probe_results ingests it back into wandb + metrics.jsonl.
#
# Classification splits come from `thunder generate-data-splits` cached under
# /data/nanopath/cache/data_splits; pcam is subsampled inline; pannuke images
# come straight from the npy folds under /block/thunder-data/pannuke.

import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


DATA_SPLITS_CACHE = Path(__file__).resolve().parent / "data_splits"
EMBED_BATCH_SIZE = 128
EMBED_NUM_WORKERS = 4
SEGMENTATION_EPOCHS = 30
SEGMENTATION_LR = 1e-3
SEGMENTATION_WEIGHT_DECAY = 1e-4
SEGMENTATION_BATCH_SIZE = 64
PANNUKE_NUM_CLASSES = 6
PANNUKE_FOLDS = {"train": "Fold1/{kind}/fold1/{kind}.npy", "val": "Fold2/{kind}/fold2/{kind}.npy"}
LINEAR_PROBE_LRS = (1e-3, 1e-4, 1e-5)
LINEAR_PROBE_WEIGHT_DECAY = 1e-4
LINEAR_PROBE_EPOCHS = 200
LINEAR_PROBE_BATCH_SIZE = 64
PCAM_SUBSET_SEED = 1337
PCAM_SUBSET_SIZES = {"train": 3072, "val": 768}
FEWSHOT_SHOTS = [1, 2, 4, 8, 16]
FEWSHOT_TRIALS = 1000
FEWSHOT_SEED = 1337
KNN_K_VALS = [1, 3, 5, 10, 20, 30, 40, 50]
KNN_CHUNK_SIZE = 4096
CLASSIFICATION_DATASETS = ["bach", "bracs", "break_his", "mhist", "pcam"]
SEGMENTATION_DATASETS = ["pannuke"]
DATASET_ROOTS = {
    "bach": Path("/block/eva-data/bach"),
    "bracs": Path("/block/eva-data/bracs"),
    "break_his": Path("/block/eva-data/breakhis"),
    "mhist": Path("/block/eva-data/mhist"),
    "pcam": Path("/block/eva-data/patch_camelyon"),
    "pannuke": Path("/block/thunder-data/pannuke"),
}


def console_prefix():
    return f"{time.strftime('%H:%M:%S')} {os.environ.get('SLURM_JOB_ID', str(os.getpid()))}"


def probe_paths(output_dir):
    probe_dir = Path(output_dir) / "thunder"
    return {
        "probe_dir": probe_dir,
        "state_path": probe_dir / "state.json",
        "results_dir": probe_dir / "results",
    }


def probe_enabled(cfg):
    return bool(cfg["probe"]["enabled"]) and (len(cfg["probe"]["datasets"]) + len(cfg["probe"].get("segmentation_datasets", []))) > 0


def write_probe_state(state):
    state["paths"]["state_path"].write_text(json.dumps(state["data"], indent=2) + "\n")


def prepare_probe_state(cfg, output_dir):
    paths = probe_paths(output_dir)
    for path in [paths["probe_dir"], paths["results_dir"]]:
        path.mkdir(parents=True, exist_ok=True)
    classification = [str(x) for x in cfg["probe"]["datasets"]]
    segmentation = [str(x) for x in cfg["probe"].get("segmentation_datasets", [])]
    data = {
        "version": 8,
        "family": str(cfg["project"]["family"]),
        "classification_datasets": classification,
        "segmentation_datasets": segmentation,
        "count": int(cfg["probe"]["count"]),
        "logged_results": [],
    }
    if paths["state_path"].exists():
        previous = json.loads(paths["state_path"].read_text())
        if previous["version"] != 8:
            raise ValueError(f"unsupported probe state version: {previous['version']}")
        if previous["family"] != data["family"]:
            raise ValueError(f"probe family changed from {previous['family']} to {data['family']}")
        if previous["classification_datasets"] != data["classification_datasets"]:
            raise ValueError(f"classification datasets changed from {previous['classification_datasets']} to {data['classification_datasets']}")
        if previous["segmentation_datasets"] != data["segmentation_datasets"]:
            raise ValueError(f"segmentation datasets changed from {previous['segmentation_datasets']} to {data['segmentation_datasets']}")
        if previous["count"] != data["count"]:
            raise ValueError(f"probe count changed from {previous['count']} to {data['count']}")
        data["logged_results"] = previous["logged_results"]
    for dataset in classification:
        if dataset not in CLASSIFICATION_DATASETS:
            raise ValueError(f"unsupported classification dataset: {dataset}")
    for dataset in segmentation:
        if dataset not in SEGMENTATION_DATASETS:
            raise ValueError(f"unsupported segmentation dataset: {dataset}")
    state = {"paths": paths, "data": data}
    write_probe_state(state)
    return state


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
        "job_id": f"{slurm_id}-{checkpoint_step:07d}",
    }
    for dataset in request["classification_datasets"] + request["segmentation_datasets"]:
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
        f"segmentation: {','.join(request['segmentation_datasets']) or '-'}",
        flush=True,
    )
    subprocess.run([sys.executable, str(Path(__file__).resolve()), request["request_path"]], env=env, check=True)
    print(
        f"{console_prefix()} Probe  [{checkpoint_step}]  "
        f"finished: {request['job_id']}  result: {request['result_path']}",
        flush=True,
    )
    return state


class ClassificationDataset(torch.utils.data.Dataset):
    # Loads images for the classification probes. For pcam we subsample with a fixed seed
    # to match Thunder's PATCH_CAMELYON_SUBSET behaviour; the other four datasets use the
    # cached data_splits JSON (one-time output of `thunder generate-data-splits`).
    def __init__(self, dataset, split, transform):
        import h5py
        import numpy as np

        self.transform = transform
        self.dataset = dataset
        if dataset == "pcam":
            pcam_split = "train" if split == "train" else "valid"
            with h5py.File(DATASET_ROOTS["pcam"] / f"camelyonpatch_level_2_split_{pcam_split}_x.h5", "r") as fx:
                key_x = next(iter(fx.keys()))
                idx = np.sort(np.random.default_rng(PCAM_SUBSET_SEED + (0 if split == "train" else 1)).choice(fx[key_x].shape[0], size=PCAM_SUBSET_SIZES[split], replace=False))
                self.images = np.array(fx[key_x][idx])
            with h5py.File(DATASET_ROOTS["pcam"] / f"camelyonpatch_level_2_split_{pcam_split}_y.h5", "r") as fy:
                self.labels = [int(v) for v in np.array(fy[next(iter(fy.keys()))][idx]).reshape(-1)]
        else:
            splits = json.loads((DATA_SPLITS_CACHE / f"{dataset}.json").read_text())[split]
            self.images = splits["images"]
            self.labels = [int(v) for v in splits["labels"]]
            self.root = DATASET_ROOTS[dataset]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        from PIL import Image
        if self.dataset == "pcam":
            img = Image.fromarray(self.images[idx])
        else:
            img = Image.open(self.root / self.images[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


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
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with autocast:
                e = model.probe_features((x - mean) / std)
            embs.append(e.float().cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(embs, axis=0).astype(np.float32), np.concatenate(labels, axis=0).astype(np.int64)


def inline_pannuke_jaccard(model, mean, std, device):
    # Precompute spatial token features once with the supplied backbone, then train
    # MaskTransformer with multiclass_dice_loss for SEGMENTATION_EPOCHS, select best
    # epoch by val loss, and report per-image macro jaccard with Thunder's bg-only
    # weighting (no_bg_only_weight_test=16).
    import numpy as np
    from sklearn.metrics import jaccard_score
    from seg_head import MaskTransformer, multiclass_dice_loss

    started_at = time.monotonic()

    @torch.no_grad()
    def extract(images_np):
        feats = []
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        for i in range(0, len(images_np), SEGMENTATION_BATCH_SIZE):
            batch = torch.from_numpy(np.ascontiguousarray(images_np[i : i + SEGMENTATION_BATCH_SIZE, 16:240, 16:240, :])).permute(0, 3, 1, 2).float().to(device) / 255.0
            with autocast:
                feats.append(model.encode_image((batch - mean) / std)[:, model.registers :].float().cpu())
        return torch.cat(feats, dim=0)

    pannuke_root = DATASET_ROOTS["pannuke"]

    def derive_labels(masks):
        labels = np.zeros((masks.shape[0], 256, 256), dtype=np.int64)
        for j in range(5):
            layer = ((j + 1) * np.clip(masks[..., j], 0, 1)).astype(np.int64)
            labels = np.where(layer != 0, layer, labels)
        return labels

    train_images = np.load(pannuke_root / PANNUKE_FOLDS["train"].format(kind="images"), mmap_mode="r")
    val_images = np.load(pannuke_root / PANNUKE_FOLDS["val"].format(kind="images"), mmap_mode="r")
    train_labels = derive_labels(np.load(pannuke_root / PANNUKE_FOLDS["train"].format(kind="masks"), mmap_mode="r"))
    val_labels = derive_labels(np.load(pannuke_root / PANNUKE_FOLDS["val"].format(kind="masks"), mmap_mode="r"))
    train_feats = extract(train_images)
    val_feats = extract(val_images)
    d_encoder = train_feats.shape[-1]
    train_labels_t = torch.from_numpy(train_labels)
    val_labels_t = torch.from_numpy(val_labels)
    head = MaskTransformer(n_cls=PANNUKE_NUM_CLASSES, d_encoder=d_encoder, n_layers=2, n_heads=8, d_model=768, d_ff=3072).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=SEGMENTATION_LR, weight_decay=SEGMENTATION_WEIGHT_DECAY)
    n = len(train_feats)
    best_val_loss = float("inf")
    best_state = None
    for _ in range(SEGMENTATION_EPOCHS):
        head.train()
        perm = torch.randperm(n)
        for i in range(0, n, SEGMENTATION_BATCH_SIZE):
            idx = perm[i : i + SEGMENTATION_BATCH_SIZE]
            labels = train_labels_t[idx].to(device)
            logits = torch.nn.functional.interpolate(head(train_feats[idx].to(device)), (256, 256), mode="bilinear")
            loss = multiclass_dice_loss(logits, labels, torch.ones_like(labels, dtype=torch.bool))
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_feats), SEGMENTATION_BATCH_SIZE):
                labels = val_labels_t[i : i + SEGMENTATION_BATCH_SIZE].to(device)
                logits = torch.nn.functional.interpolate(head(val_feats[i : i + SEGMENTATION_BATCH_SIZE].to(device)), (256, 256), mode="bilinear")
                val_loss_sum += multiclass_dice_loss(logits, labels, torch.ones_like(labels, dtype=torch.bool)).item()
                val_batches += 1
        val_loss = val_loss_sum / max(1, val_batches)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()
    per_image_j, per_image_bg_only = [], []
    with torch.no_grad():
        for i in range(0, len(val_feats), SEGMENTATION_BATCH_SIZE):
            preds = torch.nn.functional.interpolate(head(val_feats[i : i + SEGMENTATION_BATCH_SIZE].to(device)), (256, 256), mode="bilinear").argmax(dim=1).cpu().numpy()
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
    return float(np.average(per_image_j, weights=weights)), time.monotonic() - started_at


def inline_knn_val_f1(train_embs, train_labels, val_embs, val_labels, k_vals):
    import numpy as np
    from sklearn.metrics import f1_score

    train_f = train_embs.astype(np.float32, copy=False)
    val_f = val_embs.astype(np.float32, copy=False)
    train_n = train_f / np.linalg.norm(train_f, axis=1, keepdims=True)
    val_n = val_f / np.linalg.norm(val_f, axis=1, keepdims=True)
    preds_per_k = {k: [] for k in k_vals}
    for start in range(0, len(val_n), KNN_CHUNK_SIZE):
        chunk = val_n[start : start + KNN_CHUNK_SIZE]
        sim = chunk @ train_n.T
        order = np.argsort(-sim, axis=1)
        for i in range(len(chunk)):
            row = train_labels[order[i]]
            for k in k_vals:
                preds_per_k[k].append(int(np.bincount(row[:k]).argmax()))
    f1_per_k = {k: float(f1_score(val_labels, preds_per_k[k], average="macro")) for k in k_vals}
    best_k = max(f1_per_k, key=lambda k: f1_per_k[k])
    return best_k, f1_per_k[best_k], f1_per_k


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


def inline_linear_val_f1(train_embs, train_labels, val_embs, val_labels):
    import numpy as np
    from sklearn.metrics import f1_score
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(np.max(train_labels)) + 1
    train_embs_t = torch.from_numpy(train_embs).to(device)
    train_labels_t = torch.from_numpy(train_labels).long().to(device)
    val_embs_t = torch.from_numpy(val_embs).to(device)
    n = len(train_embs_t)
    best_f1 = 0.0
    for lr in LINEAR_PROBE_LRS:
        head = torch.nn.Linear(train_embs.shape[1], num_classes).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=LINEAR_PROBE_WEIGHT_DECAY)
        for _ in range(LINEAR_PROBE_EPOCHS):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, LINEAR_PROBE_BATCH_SIZE):
                idx = perm[i : i + LINEAR_PROBE_BATCH_SIZE]
                loss = torch.nn.functional.cross_entropy(head(train_embs_t[idx]), train_labels_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                preds = head(val_embs_t).argmax(-1).cpu().numpy()
            best_f1 = max(best_f1, float(f1_score(val_labels, preds, average="macro")))
    return best_f1


def run_probe_job(request_path):
    from torchvision import transforms
    from model import NanoPathFM

    probe_started_at = time.monotonic()
    request = json.loads(Path(request_path).read_text())
    classification = list(request["classification_datasets"])
    segmentation = list(request["segmentation_datasets"])
    print(
        f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
        f"start: {request['job_id']}  checkpoint: {request['checkpoint_path']}",
        flush=True,
    )
    checkpoint = torch.load(request["checkpoint_path"], map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    state_key = "model_ema" if str(cfg["probe"]["model_weights"]) == "ema" else "model"
    model_state = checkpoint[state_key]
    del checkpoint
    device = torch.device("cuda")
    model = NanoPathFM(cfg).to(device).eval()
    model.load_state_dict(model_state)
    for param in model.parameters():
        param.requires_grad = False
    mean = torch.tensor(cfg["data"]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["data"]["std"], device=device).view(1, 3, 1, 1)
    transform = transforms.Compose([transforms.Resize(256, antialias=True), transforms.CenterCrop(224), transforms.ToTensor()])

    inline_metrics = {}
    for dataset in classification:
        embed_started = time.monotonic()
        train_embs, train_labels = embed_classification_dataset(model, mean, std, dataset, "train", device, transform)
        val_embs, val_labels = embed_classification_dataset(model, mean, std, dataset, "val", device, transform)
        knn_best_k, knn_best_f1, knn_all = inline_knn_val_f1(train_embs, train_labels, val_embs, val_labels, KNN_K_VALS)
        fewshot_per_shot = inline_fewshot_val_f1(train_embs, train_labels, val_embs, val_labels, FEWSHOT_SHOTS, FEWSHOT_TRIALS, FEWSHOT_SEED + CLASSIFICATION_DATASETS.index(dataset))
        linear_f1 = inline_linear_val_f1(train_embs, train_labels, val_embs, val_labels)
        inline_metrics[dataset] = {
            "linear_val_f1": linear_f1,
            "knn_best_k": knn_best_k,
            "knn_val_f1": knn_best_f1,
            "knn_val_f1_per_k": {int(k): float(v) for k, v in knn_all.items()},
            "fewshot_val_f1": float(sum(fewshot_per_shot.values()) / len(fewshot_per_shot)),
            "fewshot_val_f1_per_shot": {int(s): float(v) for s, v in fewshot_per_shot.items()},
        }
        print(
            f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
            f"inline_done: {dataset}  linear_f1={linear_f1:.4f}  knn_f1={knn_best_f1:.4f}  "
            f"fewshot_f1={inline_metrics[dataset]['fewshot_val_f1']:.4f}  wall={time.monotonic()-embed_started:.2f}s",
            flush=True,
        )

    seg_jaccards = {}
    for dataset in segmentation:
        print(f"{console_prefix()} ProbeWorker  [{request['train_step']}]  inline_seg_start: {dataset}", flush=True)
        jaccard, seg_wall = inline_pannuke_jaccard(model, mean, std, device)
        seg_jaccards[dataset] = jaccard
        print(
            f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
            f"inline_seg_done: {dataset}  jaccard={jaccard:.4f}  wall={seg_wall:.2f}s",
            flush=True,
        )

    # Aggregate per-dataset metrics into the result file.
    metrics = {}
    results = {}
    for dataset in classification:
        metrics[f"probe_{dataset}_linear_val_f1"] = inline_metrics[dataset]["linear_val_f1"]
        metrics[f"probe_{dataset}_knn_val_f1"] = inline_metrics[dataset]["knn_val_f1"]
        metrics[f"probe_{dataset}_fewshot_val_f1"] = inline_metrics[dataset]["fewshot_val_f1"]
        results[dataset] = inline_metrics[dataset]
    for dataset in segmentation:
        metrics[f"probe_{dataset}_seg_val_jaccard"] = seg_jaccards[dataset]
        results[dataset] = {"seg_val_jaccard": seg_jaccards[dataset]}

    # Aggregates
    if len(classification) > 0:
        metrics["linear_mean_f1"] = sum(metrics[f"probe_{d}_linear_val_f1"] for d in classification) / len(classification)
        metrics["knn_mean_f1"] = sum(metrics[f"probe_{d}_knn_val_f1"] for d in classification) / len(classification)
        metrics["fewshot_mean_f1"] = sum(metrics[f"probe_{d}_fewshot_val_f1"] for d in classification) / len(classification)
    if len(segmentation) > 0:
        metrics["seg_mean_jaccard"] = sum(metrics[f"probe_{d}_seg_val_jaccard"] for d in segmentation) / len(segmentation)

    task_means = [metrics[k] for k in ("linear_mean_f1", "knn_mean_f1", "fewshot_mean_f1", "seg_mean_jaccard") if k in metrics]
    metrics["mean_probe_score"] = sum(task_means) / len(task_means)

    print(
        f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
        f"result: mean_probe_score={metrics['mean_probe_score']:.6f}  "
        f"linear={metrics.get('linear_mean_f1')}  knn={metrics.get('knn_mean_f1')}  "
        f"fewshot={metrics.get('fewshot_mean_f1')}  seg={metrics.get('seg_mean_jaccard')}  "
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
                "status": "ok",
                "metrics": metrics,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


def collect_probe_results(state, wandb_run, metrics_path, output_dir, best_val_mean_probe_score, best_val_mean_probe_score_step, best_probe_scores):
    state["data"] = json.loads(state["paths"]["state_path"].read_text())
    logged = set(state["data"]["logged_results"])
    for result_path in sorted(state["paths"]["results_dir"].glob("step_*.json")):
        result_path_str = str(result_path)
        result = json.loads(result_path.read_text())
        metrics = {key: float(value) for key, value in result["metrics"].items()}
        checkpoint_path = Path(result["checkpoint_path"])
        if result["status"] == "ok" and "mean_probe_score" in metrics:
            if metrics["mean_probe_score"] > best_val_mean_probe_score:
                best_val_mean_probe_score = metrics["mean_probe_score"]
                best_val_mean_probe_score_step = int(result["train_step"])
                best_probe_scores = dict(metrics)
                if checkpoint_path.exists():
                    shutil.copy2(checkpoint_path, output_dir / "best_mean_probe_score.pt")
        if result_path_str in logged:
            continue
        event_payload = {
            "event": "probe",
            "step": result["train_step"],
            "status": result["status"],
            "target_flops": result["target_flops"],
            "target_fraction": result["target_fraction"],
            "probe_wall_seconds": float(result["wall_seconds"]),
            **metrics,
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(event_payload) + "\n")
        print(
            f"{console_prefix()} Probe  [{result['train_step']}]  "
            f"log_result: status={result['status']}  mean_probe_score={metrics.get('mean_probe_score')}  "
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
    return best_val_mean_probe_score, best_val_mean_probe_score_step, best_probe_scores


def completed_probe_summary(output_dir):
    summary = {}
    final_result = None
    for result_path in sorted(probe_paths(output_dir)["results_dir"].glob("step_*.json")):
        result = json.loads(result_path.read_text())
        if result["status"] != "ok" or "mean_probe_score" not in result["metrics"]:
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


def main():
    if len(sys.argv) != 2:
        raise ValueError("usage: python probe.py <request.json>")
    run_probe_job(sys.argv[1])


if __name__ == "__main__":
    main()
