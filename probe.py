import json
import math
import os
import shutil
import subprocess
import sys
import fcntl
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import wandb


THUNDER_REPO = Path("/admin/home/paul/thunder")
THUNDER_PYTHON = THUNDER_REPO / ".venv" / "bin" / "python"
THUNDER_BIN = THUNDER_REPO / ".venv" / "bin" / "thunder"
THUNDER_MODEL = Path(__file__).resolve().with_name("thunder_adapter.py")
PRECOMPUTE_EMBEDDING_BATCH_SIZE = 512
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
PATCH_CAMELYON_SUBSET_SEED = 1337
PATCH_CAMELYON_SUBSET_SIZES = {"train": 3072, "valid": 768}
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
THUNDER_DATASET_NAMES = {
    "bach": "bach",
    "bracs": "bracs",
    "break_his": "break_his",
    "mhist": "mhist",
    "pcam": "patch_camelyon",
    "pannuke": "pannuke",
}


def configure_probe_wandb_metrics(wandb_run):
    for key in ("probe/target_flops", "probe/wall_seconds"):
        wandb_run.define_metric(key, hidden=True, overwrite=True)


def console_prefix():
    return f"{time.strftime('%H:%M:%S')} {os.environ.get('SLURM_JOB_ID', str(os.getpid()))}"


def probe_paths(output_dir):
    output_dir = Path(output_dir)
    probe_dir = output_dir / "thunder"
    return {
        "probe_dir": probe_dir,
        "state_path": probe_dir / "state.json",
        "lock_path": probe_dir / "state.lock",
        "results_dir": probe_dir / "results",
        "scratch_root": Path("/tmp/nanopath-thunder") / output_dir.name,
    }


def probe_enabled(cfg):
    return bool(cfg["probe"]["enabled"]) and (len(cfg["probe"]["datasets"]) + len(cfg["probe"].get("segmentation_datasets", []))) > 0


def write_probe_state(state):
    state["paths"]["state_path"].write_text(json.dumps(state["data"], indent=2) + "\n")


def prepare_probe_state(cfg, output_dir):
    paths = probe_paths(output_dir)
    for path in [paths["probe_dir"], paths["results_dir"], paths["scratch_root"]]:
        path.mkdir(parents=True, exist_ok=True)
    classification = [str(x) for x in cfg["probe"]["datasets"]]
    segmentation = [str(x) for x in cfg["probe"].get("segmentation_datasets", [])]
    data = {
        "version": 7,
        "family": str(cfg["project"]["family"]),
        "classification_datasets": classification,
        "segmentation_datasets": segmentation,
        "count": int(cfg["probe"]["count"]),
        "active": None,
        "logged_results": [],
    }
    if paths["state_path"].exists():
        previous = json.loads(paths["state_path"].read_text())
        if previous["version"] != 7:
            raise ValueError(f"unsupported Thunder probe state version: {previous['version']}")
        if previous["family"] != data["family"]:
            raise ValueError(f"Thunder probe family changed from {previous['family']} to {data['family']}")
        if previous["classification_datasets"] != data["classification_datasets"]:
            raise ValueError(f"Thunder classification datasets changed from {previous['classification_datasets']} to {data['classification_datasets']}")
        if previous["segmentation_datasets"] != data["segmentation_datasets"]:
            raise ValueError(f"Thunder segmentation datasets changed from {previous['segmentation_datasets']} to {data['segmentation_datasets']}")
        if previous["count"] != data["count"]:
            raise ValueError(f"Thunder probe count changed from {previous['count']} to {data['count']}")
        data["logged_results"] = previous["logged_results"]
    for dataset in classification:
        if dataset not in CLASSIFICATION_DATASETS:
            raise ValueError(f"unsupported Thunder classification dataset: {dataset}")
    for dataset in segmentation:
        if dataset not in SEGMENTATION_DATASETS:
            raise ValueError(f"unsupported Thunder segmentation dataset: {dataset}")
    state = {"paths": paths, "data": data}
    write_probe_state(state)
    return state


def checkpoint_request(state, checkpoint_step, target_flops, target_fraction):
    step_tag = f"step_{checkpoint_step:07d}"
    return {
        "checkpoint_step": int(checkpoint_step),
        "train_step": int(checkpoint_step),
        "target_flops": int(target_flops),
        "target_fraction": float(target_fraction),
        "checkpoint_path": str(state["paths"]["probe_dir"] / f"{step_tag}.pt"),
        "request_path": str(state["paths"]["probe_dir"] / f"{step_tag}.request.json"),
        "result_path": str(state["paths"]["results_dir"] / f"{step_tag}.json"),
        "model_name": f"{state['data']['family']}_{step_tag}",
        "classification_datasets": list(state["data"]["classification_datasets"]),
        "segmentation_datasets": list(state["data"]["segmentation_datasets"]),
        "job_id": None,
        "submitted_at_utc": None,
    }


def queue_probe_job(cfg, state, checkpoint_payload, checkpoint_step, target_flops, target_fraction):
    with state["paths"]["lock_path"].open("w") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        state["data"] = json.loads(state["paths"]["state_path"].read_text())
        request = checkpoint_request(state, checkpoint_step, target_flops, target_fraction)
        torch.save(checkpoint_payload, request["checkpoint_path"])
        if state["data"]["active"] is not None:
            raise RuntimeError(f"Thunder probe already active for step {state['data']['active']['train_step']}")
        for dataset in request["classification_datasets"] + request["segmentation_datasets"]:
            if not DATASET_ROOTS[dataset].exists():
                raise FileNotFoundError(f"missing Thunder dataset root for {dataset}: {DATASET_ROOTS[dataset]}")
        slurm_id = os.environ.get("SLURM_JOB_ID", f"local-{os.getpid()}")
        request["job_id"] = f"{slurm_id}-{request['checkpoint_step']:07d}"
        request["submitted_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        Path(request["request_path"]).write_text(json.dumps(request, indent=2) + "\n")
        state["data"]["active"] = request
        write_probe_state(state)
    torch.cuda.empty_cache()
    runtime_dir = state["paths"]["scratch_root"] / f"step_{checkpoint_step:07d}-{request['job_id']}"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("WANDB_SERVICE", None)
    env["NANOPATH_THUNDER_RUNTIME_DIR"] = str(runtime_dir)
    print(
        f"{console_prefix()} Probe  [{checkpoint_step}]  "
        f"start: {request['job_id']}  target_fraction: {target_fraction:.4f}  "
        f"classification: {','.join(request['classification_datasets']) or '-'}  "
        f"segmentation: {','.join(request['segmentation_datasets']) or '-'}",
        flush=True,
    )
    subprocess.run([str(THUNDER_PYTHON), str(Path(__file__).resolve()), request["request_path"]], env=env, check=True)
    shutil.rmtree(runtime_dir)
    print(
        f"{console_prefix()} Probe  [{checkpoint_step}]  "
        f"finished: {request['job_id']}  result: {request['result_path']}",
        flush=True,
    )
    return state


def prepare_patch_camelyon(runtime_dir):
    import h5py
    import numpy as np
    import yaml

    dst_root = runtime_dir / "datasets" / "patch_camelyon"
    dst_root.mkdir(parents=True, exist_ok=True)
    custom_dir = runtime_dir / "custom_datasets"
    custom_dir.mkdir(parents=True, exist_ok=True)
    for split_idx, split in enumerate(["train", "valid"]):
        src_x = DATASET_ROOTS["pcam"] / f"camelyonpatch_level_2_split_{split}_x.h5"
        src_y = DATASET_ROOTS["pcam"] / f"camelyonpatch_level_2_split_{split}_y.h5"
        dst_x = dst_root / src_x.name
        dst_y = dst_root / src_y.name
        with h5py.File(src_x, "r") as fx, h5py.File(src_y, "r") as fy:
            x_key = next(iter(fx.keys()))
            y_key = next(iter(fy.keys()))
            total = fx[x_key].shape[0]
            take = min(total, int(PATCH_CAMELYON_SUBSET_SIZES[split]))
            indices = np.sort(np.random.default_rng(PATCH_CAMELYON_SUBSET_SEED + split_idx).choice(total, size=take, replace=False))
            with h5py.File(dst_x, "w") as gx, h5py.File(dst_y, "w") as gy:
                gx.create_dataset(x_key, data=fx[x_key][indices])
                gy.create_dataset(y_key, data=fy[y_key][indices])
    data_splits = {
        "train": {
            "images": "camelyonpatch_level_2_split_train_x.h5",
            "labels": "camelyonpatch_level_2_split_train_y.h5",
        },
        "val": {
            "images": "camelyonpatch_level_2_split_valid_x.h5",
            "labels": "camelyonpatch_level_2_split_valid_y.h5",
        },
    }
    json_path = custom_dir / "patch_camelyon.json"
    yaml_path = custom_dir / "patch_camelyon.yaml"
    json_path.write_text(json.dumps(data_splits, indent=2) + "\n")
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "dataset_name": "patch_camelyon",
                "nb_classes": 2,
                "base_data_folder": str(runtime_dir / "datasets"),
                "compatible_tasks": [
                    "knn",
                    "linear_probing",
                    "pre_computing_embeddings",
                    "simple_shot",
                ],
                "nb_train_samples": PATCH_CAMELYON_SUBSET_SIZES["train"],
                "nb_val_samples": PATCH_CAMELYON_SUBSET_SIZES["valid"],
                "nb_test_samples": 0,
                "image_sizes": [[96, 96]],
                "mpp": 1.0,
                "cancer_type": "breast",
                "h5_format": True,
                "data_splits": str(json_path),
                "classes": ["no-metastatic-tissue", "metastatic-tissue"],
                "class_to_id": {"no-metastatic-tissue": 0, "metastatic-tissue": 1},
                "id_to_class": {0: "no-metastatic-tissue", 1: "metastatic-tissue"},
                "id_to_classname": {
                    0: "lymph node",
                    1: "lymph node containing metastatic tumor tissue",
                },
            },
            sort_keys=False,
        )
    )
    return f"custom:{yaml_path}"


def load_pannuke_split(split):
    import numpy as np

    root = DATASET_ROOTS["pannuke"]
    images = np.load(root / PANNUKE_FOLDS[split].format(kind="images"), mmap_mode="r")
    masks = np.load(root / PANNUKE_FOLDS[split].format(kind="masks"), mmap_mode="r")
    labels = np.zeros((masks.shape[0], 256, 256), dtype=np.int64)
    for j in range(5):
        layer = ((j + 1) * np.clip(masks[..., j], 0, 1)).astype(np.int64)
        labels = np.where(layer != 0, layer, labels)
    return images, labels


def inline_pannuke_jaccard(cfg, model_state):
    # Reuse the just-pretrained backbone (already-loaded EMA state). Precompute spatial
    # token features once, then train Thunder's MaskTransformer with multiclass_dice_loss
    # for SEGMENTATION_EPOCHS, select best epoch by val loss, and report per-image macro
    # jaccard with Thunder's bg-only weighting (no_bg_only_weight_test=16).
    import numpy as np
    from sklearn.metrics import jaccard_score
    from thunder.models.task_specific_models import MaskTransformer
    from thunder.utils.dice_loss import multiclass_dice_loss
    from model import NanoPathFM

    started_at = time.monotonic()
    device = torch.device("cuda")
    model = NanoPathFM(cfg).to(device).eval()
    model.load_state_dict(model_state)
    for param in model.parameters():
        param.requires_grad = False
    mean = torch.tensor(cfg["data"]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["data"]["std"], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def extract(images_np):
        feats = []
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        for i in range(0, len(images_np), SEGMENTATION_BATCH_SIZE):
            batch = torch.from_numpy(np.ascontiguousarray(images_np[i : i + SEGMENTATION_BATCH_SIZE, 16:240, 16:240, :])).permute(0, 3, 1, 2).float().to(device) / 255.0
            with autocast:
                feats.append(model.encode_image((batch - mean) / std)[:, model.registers :].float().cpu())
        return torch.cat(feats, dim=0)

    train_images, train_labels = load_pannuke_split("train")
    val_images, val_labels = load_pannuke_split("val")
    train_feats = extract(train_images)
    val_feats = extract(val_images)
    d_encoder = train_feats.shape[-1]
    del model
    torch.cuda.empty_cache()
    train_labels_t = torch.from_numpy(train_labels)
    val_labels_t = torch.from_numpy(val_labels)
    head = MaskTransformer(n_cls=PANNUKE_NUM_CLASSES, d_encoder=d_encoder, n_layers=2, n_heads=8, d_model=768, d_ff=3072, drop_path_rate=0.0, dropout=0.0).to(device)
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


def _normalize_rows(x):
    import numpy as np
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def inline_knn_val_f1(train_embs, train_labels, val_embs, val_labels, k_vals):
    import numpy as np
    from sklearn.metrics import f1_score

    train_n = _normalize_rows(train_embs.astype(np.float32, copy=False))
    val_n = _normalize_rows(val_embs.astype(np.float32, copy=False))
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
            cls_n = _normalize_rows(cls_embs)
            val_n = _normalize_rows(val_embs - mean)
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


def load_precomputed_embeddings(runtime_dir, thunder_name, model_name):
    import h5py
    import numpy as np

    embs_dir = runtime_dir / "embeddings" / thunder_name / model_name
    out = {}
    for split in ("train", "val"):
        with h5py.File(embs_dir / split / "embeddings.h5", "r") as f_e, h5py.File(embs_dir / split / "labels.h5", "r") as f_l:
            keys = sorted(f_e.keys(), key=int)
            out[f"{split}_embs"] = np.stack([np.asarray(f_e[k]) for k in keys]).astype(np.float32)
            out[f"{split}_labels"] = np.asarray([int(np.asarray(f_l[k])) for k in keys], dtype=np.int64)
    return out


def run_probe_job(request_path):
    probe_started_at = time.monotonic()
    request = json.loads(Path(request_path).read_text())
    classification = list(request["classification_datasets"])
    segmentation = list(request["segmentation_datasets"])
    print(
        f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
        f"start: {request['job_id']}  checkpoint: {request['checkpoint_path']}",
        flush=True,
    )
    runtime_dir = Path(os.environ["NANOPATH_THUNDER_RUNTIME_DIR"])
    datasets_dir = runtime_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(request["checkpoint_path"], map_location="cpu", weights_only=False)
    checkpoint_cfg = checkpoint["config"]
    seg_model_state = checkpoint["model_ema" if str(checkpoint_cfg["probe"]["model_weights"]) == "ema" else "model"]
    del checkpoint
    env = os.environ.copy()
    env.update(
        {
            "THUNDER_BASE_DATA_FOLDER": str(runtime_dir),
            "NANOPATH_THUNDER_CKPT": request["checkpoint_path"],
            "NANOPATH_THUNDER_MODEL_NAME": request["model_name"],
            "PYTHONPATH": str(Path(__file__).resolve().parent),
            "THUNDER_WANDB_MODE": "disabled",
            "WANDB_MODE": "disabled",
            "WANDB_SILENT": "true",
        }
    )
    os.environ.update(env)

    # Set up classification datasets (symlink or custom yaml) and collect builtin names
    classification_targets = {}
    classification_builtin = []
    for dataset in classification:
        thunder_name = THUNDER_DATASET_NAMES[dataset]
        if dataset == "pcam":
            classification_targets[dataset] = prepare_patch_camelyon(runtime_dir)
        else:
            (datasets_dir / thunder_name).symlink_to(DATASET_ROOTS[dataset])
            classification_builtin.append(thunder_name)
            classification_targets[dataset] = thunder_name

    # Generate Thunder data splits for the classification datasets (pannuke is loaded inline below).
    if len(classification_builtin) > 0:
        print(
            f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
            f"generate_splits: {','.join(classification_builtin)}",
            flush=True,
        )
        subprocess.run([str(THUNDER_BIN), "generate-data-splits", *classification_builtin], cwd=THUNDER_REPO, env=env, check=True)

    # Launch classification precompute subprocesses; while they run, do inline pannuke segmentation on the same GPU.
    precompute_procs = []
    for dataset in classification:
        args = [str(THUNDER_BIN), "benchmark", f"custom:{THUNDER_MODEL}", classification_targets[dataset], "pre_computing_embeddings"]
        if dataset != "bracs":
            args.extend(["--task.pre_comp_emb_batch_size", str(PRECOMPUTE_EMBEDDING_BATCH_SIZE)])
        precompute_procs.append((dataset, subprocess.Popen(args, cwd=THUNDER_REPO, env=env)))
    print(
        f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
        f"precompute_start: {','.join(classification) or '-'}",
        flush=True,
    )
    seg_jaccards = {}
    for dataset in segmentation:
        if dataset != "pannuke":
            raise NotImplementedError(f"inline segmentation not implemented for {dataset}")
        print(
            f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
            f"inline_seg_start: {dataset}",
            flush=True,
        )
        jaccard, seg_wall = inline_pannuke_jaccard(checkpoint_cfg, seg_model_state)
        seg_jaccards[dataset] = jaccard
        print(
            f"{console_prefix()} ProbeWorker  [{request['train_step']}]  "
            f"inline_seg_done: {dataset}  jaccard={jaccard:.4f}  wall={seg_wall:.2f}s",
            flush=True,
        )
    for dataset, proc in precompute_procs:
        if proc.wait() != 0:
            raise RuntimeError(f"Thunder embedding precompute failed for {dataset}")

    # Once precompute subprocesses finish, run inline KNN + SimpleShot + linear probe on the embedding files.
    inline_metrics = {}
    for dataset in classification:
        thunder_name = THUNDER_DATASET_NAMES[dataset]
        loaded = load_precomputed_embeddings(runtime_dir, thunder_name, request["model_name"])
        knn_best_k, knn_best_f1, knn_all = inline_knn_val_f1(
            loaded["train_embs"], loaded["train_labels"], loaded["val_embs"], loaded["val_labels"], KNN_K_VALS,
        )
        fewshot_per_shot = inline_fewshot_val_f1(
            loaded["train_embs"], loaded["train_labels"], loaded["val_embs"], loaded["val_labels"],
            FEWSHOT_SHOTS, FEWSHOT_TRIALS, FEWSHOT_SEED + hash(dataset) % (2**31),
        )
        linear_f1 = inline_linear_val_f1(
            loaded["train_embs"], loaded["train_labels"], loaded["val_embs"], loaded["val_labels"],
        )
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
            f"fewshot_f1={inline_metrics[dataset]['fewshot_val_f1']:.4f}",
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
                "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "wall_seconds": time.monotonic() - probe_started_at,
                "runner_id": request["job_id"],
                "checkpoint_step": request["checkpoint_step"],
                "train_step": request["train_step"],
                "target_flops": request["target_flops"],
                "target_fraction": request["target_fraction"],
                "checkpoint_path": request["checkpoint_path"],
                "model_name": request["model_name"],
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
    paths = probe_paths(Path(request["checkpoint_path"]).parents[1])
    with paths["lock_path"].open("w") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        state = {"paths": paths, "data": json.loads(paths["state_path"].read_text())}
        if state["data"]["active"] is not None and str(state["data"]["active"]["job_id"]) == str(request["job_id"]):
            state["data"]["active"] = None
        write_probe_state(state)


def collect_probe_results(state, wandb_run, metrics_path, output_dir, best_val_mean_probe_score, best_val_mean_probe_score_step, best_probe_scores):
    if wandb_run is None:
        return best_val_mean_probe_score, best_val_mean_probe_score_step, best_probe_scores
    with state["paths"]["lock_path"].open("w") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
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
                "event": "thunder_probe",
                "step": result["train_step"],
                "status": result["status"],
                "target_flops": result["target_flops"],
                "target_fraction": result["target_fraction"],
                **metrics,
            }
            if "wall_seconds" in result:
                event_payload["probe_wall_seconds"] = float(result["wall_seconds"])
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(event_payload) + "\n")
            print(
                f"{console_prefix()} Probe  [{result['train_step']}]  "
                f"log_result: status={result['status']}  mean_probe_score={metrics.get('mean_probe_score')}  "
                f"wall={result.get('wall_seconds')}",
                flush=True,
            )
            if len(metrics) > 0:
                wandb_payload = {"probe/target_flops": int(result["target_flops"])}
                for key, value in metrics.items():
                    metric_name = key.removeprefix("probe_") if key.startswith("probe_") else key
                    wandb_payload[f"probe/{metric_name}"] = value
                if "wall_seconds" in result:
                    wandb_payload["probe/wall_seconds"] = float(result["wall_seconds"])
                wandb_run.log(wandb_payload, step=int(result["train_step"]))
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            logged.add(result_path_str)
        state["data"]["logged_results"] = sorted(logged)
        active = state["data"]["active"]
        if active is not None and Path(active["result_path"]).exists():
            state["data"]["active"] = None
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
    if "wall_seconds" in final_result:
        summary["final_probe_wall_seconds"] = float(final_result["wall_seconds"])
    for key, value in final_result["metrics"].items():
        flat = "score" if key == "mean_probe_score" else key.removeprefix("probe_")
        summary[f"final_probe_{flat}"] = float(value)
    return summary


def collect_finished_probe_results(output_dir):
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return
    state = {"paths": probe_paths(output_dir), "data": json.loads((output_dir / "thunder" / "state.json").read_text())}
    latest_checkpoint_path = output_dir / "latest.pt"
    if not latest_checkpoint_path.exists():
        raise FileNotFoundError(f"missing latest training checkpoint in {output_dir}")
    checkpoint = torch.load(latest_checkpoint_path, map_location="cpu", weights_only=False)
    wandb_meta = checkpoint["wandb"]
    if wandb_meta is None:
        raise ValueError(f"missing wandb metadata in {latest_checkpoint_path}")
    os.environ.pop("WANDB_SERVICE", None)
    wandb_run = wandb.init(
        project=wandb_meta["project"],
        name=wandb_meta["name"],
        id=wandb_meta["id"],
        resume="must",
        dir="/data/nanopath/wandb",
        config=checkpoint["config"],
        settings=wandb.Settings(
            console="wrap",
            x_file_stream_transmit_interval=5,
        ),
    )
    configure_probe_wandb_metrics(wandb_run)
    summary = json.loads(summary_path.read_text())
    best_val_mean_probe_score = float("-inf") if summary["best_val_mean_probe_score"] is None else float(summary["best_val_mean_probe_score"])
    best_val_mean_probe_score_step = int(summary["best_val_mean_probe_score_step"])
    best_probe_scores = {}
    for key, value in summary.items():
        if key == "mean_probe_score" or key.endswith("_mean_f1") or key.endswith("_mean_jaccard") or (key.startswith("probe_") and (key.endswith("_val_f1") or key.endswith("_val_jaccard"))):
            best_probe_scores[key] = float(value)
    best_val_mean_probe_score, best_val_mean_probe_score_step, best_probe_scores = collect_probe_results(
        state,
        wandb_run,
        output_dir / "metrics.jsonl",
        output_dir,
        best_val_mean_probe_score,
        best_val_mean_probe_score_step,
        best_probe_scores,
    )
    state["data"] = json.loads(state["paths"]["state_path"].read_text())
    summary["best_val_mean_probe_score"] = None if not math.isfinite(best_val_mean_probe_score) else best_val_mean_probe_score
    summary["best_val_mean_probe_score_step"] = best_val_mean_probe_score_step
    summary["thunder_probe_active_job_id"] = None if state["data"]["active"] is None else state["data"]["active"]["job_id"]
    summary["thunder_probe_active_step"] = None if state["data"]["active"] is None else state["data"]["active"]["train_step"]
    summary.update(best_probe_scores)
    summary.update(completed_probe_summary(output_dir))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    for key, value in summary.items():
        wandb_run.summary[key] = value
    wandb_run.finish()


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "collect":
        collect_finished_probe_results(sys.argv[2])
        return
    if len(sys.argv) != 2:
        raise ValueError("usage: python probe.py <request.json> | python probe.py collect <output_dir>")
    run_probe_job(sys.argv[1])


if __name__ == "__main__":
    main()
