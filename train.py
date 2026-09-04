# DINO pretraining on TCGA tiles (single-GPU), initialized from DINOv2. The losses are:
# DINO CLS self-distillation (Sinkhorn-Knopp centred teacher targets),
# iBOT or I-JEPA masked-patch learning, optional FINO metadata guidance, and KDE uniformity on
# L2-normalised CLS tokens. YAML drives the tunable knobs (backbone variant,
# LR + LR scheduler, drop path, layerwise decay, KDE weight + concentration,
# FLOP/sample budgets, batch size); other objective hyperparameters are hardcoded
# inline at their use sites.

import atexit
import contextlib
import fnmatch
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from torchvision import transforms
from torchvision.transforms import functional as TF

from dataloader import TCGATileDataset, TILE_SIZE
from model import DINOHead, GradScale, JEPAPredictor, ViT, load_pretrained
from probe import (
    completed_probe_summary,
    collect_probe_results,
    prepare_probe_state,
    probe_enabled,
    queue_probe_job,
)


# Prefix every console line with wall time and job/process id so SLURM logs are easy to scan.
def console_prefix(): return f"{time.strftime('%H:%M:%S')} {os.environ.get('SLURM_JOB_ID', str(os.getpid()))}"


# Read the YAML recipe and fail before any GPU work if the parquet tile dataset is absent.
# expandvars is necessary to resolve `$USER` for checked-in configs.
def load_config():
    if len(sys.argv) < 2:
        raise ValueError("usage: python train.py <config.yaml> [output_dir=<path>] [seed=<int>]")
    cfg = yaml.safe_load(os.path.expandvars(Path(sys.argv[1]).read_text()))
    cfg["config_path"] = str(Path(sys.argv[1]).resolve())
    # Run identity and confirmation seed are the only CLI overrides; recipes stay in YAML.
    for arg in sys.argv[2:]:
        key, _, value = arg.partition("=")
        if key == "output_dir":
            cfg["project"]["output_dir"] = os.path.expandvars(value)
        elif key == "seed":
            cfg["train"]["seed"] = int(value)
        else:
            raise ValueError(f"unsupported override {arg!r}; use output_dir=<path> or seed=<int>")
    dataset_dir = Path(cfg["data"]["dataset_dir"])
    if not any(dataset_dir.glob("shard-*.parquet")):
        raise FileNotFoundError(
            f"No parquet shards (shard-*.parquet) under {dataset_dir}. Pull the 4M-tile "
            f"parquet dataset from medarc/nanopath on HF by running "
            f"`python prepare.py {cfg['config_path']} download=True`. Follow the data setup in "
            f"README.md before launching train.py."
        )
    return cfg


# Arm Labless before any GPU work so direct `python train.py ...` gets the same
# no-scope GitHub device login path as the SLURM launcher. Noninteractive runs
# train locally unless the launcher passed a preauthorized token file.
def maybe_arm_labless_autosubmit(cfg, repo_dir):
    token_path = os.environ.get("LABLESS_AUTOSUBMIT_FILE", "")
    eligible = (
        bool(cfg["probe"]["enabled"])
        and int(cfg["probe"]["count"]) > 0
        and int(cfg["train"]["max_train_samples"]) == 1_000_000
        and int(cfg["train"]["max_train_flops"]) == 1_000_000_000_000_000_000
    )
    if token_path:
        atexit.register(lambda p=Path(token_path): p.unlink(missing_ok=True))
        return token_path
    if not eligible:
        return ""
    if not sys.stdin.isatty():
        if not os.environ.get("SLURM_JOB_ID"):
            print(f"{console_prefix()} Labless  no interactive stdin; training will run without auto-submit.", flush=True)
        return ""
    print("This looks like a full Labless-eligible run. Leave the run name blank to train without auto-submit.", flush=True)
    run_name = input("Labless run name (<=20 chars): ").strip()
    notes = input("Labless experiment note (unique change + why): ").strip()
    if not run_name or len(run_name) > 20:
        print("Labless auto-submit skipped; run name is required and must be <=20 chars.", flush=True)
        return ""
    token_path = str(Path(str(Path(cfg["project"]["output_dir"]).expanduser().resolve()) + ".labless_autosubmit.json"))
    status = subprocess.run(
        [sys.executable, str(repo_dir / "labless" / "submit_to_labless.py"), "login_only=true", f"token_output={token_path}", f"run_name={run_name}", f"notes={notes}"],
        cwd=repo_dir,
        check=False,
    ).returncode
    if status != 0:
        print("Labless login did not complete; training will run without auto-submit.", flush=True)
        Path(token_path).unlink(missing_ok=True)
        return ""
    os.environ["LABLESS_AUTOSUBMIT_FILE"] = token_path
    atexit.register(lambda p=Path(token_path): p.unlink(missing_ok=True))
    return token_path


def finish_labless_autosubmit(token_path, output_dir, repo_dir):
    token_file = Path(token_path) if token_path else None
    if token_file is None or not token_file.exists():
        return
    token = json.loads(token_file.read_text())
    status = subprocess.run(
        [
            sys.executable,
            str(repo_dir / "labless" / "submit_to_labless.py"),
            f"output_dir={output_dir.resolve()}",
            f"run_name={token['run_name']}",
            f"notes={token['notes']}",
            f"github_token_file={token_file}",
        ],
        cwd=repo_dir,
        check=False,
    ).returncode
    token_file.unlink(missing_ok=True)
    if status == 2:
        print(f"{console_prefix()} Labless  auto-submit skipped because the completed run did not satisfy submission restrictions.", flush=True)
    elif status != 0:
        raise SystemExit(status)


# Cosine schedule from `start` to `end` over fractional progress in [0, 1].
def cosine_schedule(start, end, frac):
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * min(1.0, max(0.0, frac))))


# Sinkhorn-Knopp centring across this batch, used as DINO/iBOT teacher targets.
def sinkhorn(x, temp):
    q = torch.exp(x.float() / temp).t()
    b = q.shape[1]
    k = q.shape[0]
    q /= q.sum()
    for _ in range(3):
        q /= q.sum(1, keepdim=True) * k
        q /= q.sum(0, keepdim=True) * b
    return (q * b).t()


# Cross-entropy between teacher distribution and softmax(student / 0.1).
def dino_ce(student, teacher):
    return -(teacher * F.log_softmax(student / 0.1, dim=-1)).sum(-1).mean()


# KDE uniformity loss on L2-normalised CLS tokens.
def kde_loss(x, concentration):
    x = F.normalize(x, p=2, dim=-1)
    sim = concentration * (x @ x.T)
    sim.fill_diagonal_(-float("inf"))
    return torch.logsumexp(sim, dim=1).mean() - math.log(max(1, sim.shape[1] - 1))


# Sample iBOT masking pattern: per-image bernoulli on whether to mask, then random patch ratio.
def make_masks(batch, patches, device):
    masks = torch.zeros(batch, patches, dtype=torch.bool, device=device)
    for i in range(batch):
        if random.random() < 0.5:
            masks[i, torch.randperm(patches, device=device)[: int(patches * random.uniform(0.1, 0.45))]] = True
    idx = masks.flatten().nonzero().flatten()
    weights = (1 / masks.sum(-1).clamp(min=1)).unsqueeze(-1).expand_as(masks)[masks]
    return masks, idx, weights


# RobustNorm's I-JEPA recipe masks contiguous square blocks to infer missing tissue context.
def make_block_mask(batch, grid, device, n_blocks, block_scale):
    masks = torch.zeros(batch, grid, grid, dtype=torch.bool, device=device)
    side = max(1, round(grid * block_scale ** 0.5))
    for i in range(batch):
        for _ in range(n_blocks):
            top, left = random.randint(0, grid - side), random.randint(0, grid - side)
            masks[i, top : top + side, left : left + side] = True
    masks = masks.flatten(1)
    idx = masks.flatten().nonzero().flatten()
    weights = (1 / masks.sum(-1).clamp(min=1)).unsqueeze(-1).expand_as(masks)[masks]
    return masks, idx, weights


# AdamW parameter groups with layer-wise LR decay on the backbone:
# block i gets lr * layerwise_decay^(depth - 1 - i); patch_embed gets the deepest decay
# multiplied by patch_embed_lr_mult; biases and norms get no weight decay; the head's
# final weight-norm last_layer parameters get an LR-freeze for the first dino.freeze_last_layer_fraction.
def build_param_groups(student_backbone, student_dino_head, student_patch_head, layerwise_decay, patch_embed_lr_mult):
    depth = len(student_backbone.blocks)
    # Coalesce params that share (lr_mult, wd_mult, last_layer) into a single group each (~30 groups
    # instead of one-per-param), so AdamW's foreach path fuses the step across many tensors rather than
    # launching per-parameter kernels. Per-param lr/wd are unchanged, so the optimization is numerically identical.
    coalesced = {}
    modules = ((student_backbone, "backbone"), (student_dino_head, "dino_head"), (student_patch_head, "patch_head"))
    for module, kind in modules:
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            lr_mult = 1.0
            if kind == "backbone" and name.startswith("blocks."):
                lr_mult = layerwise_decay ** (depth - 1 - int(name.split(".")[1]))
            elif kind == "backbone" and name.startswith("patch_embed."):
                lr_mult = (layerwise_decay ** depth) * patch_embed_lr_mult
            wd_mult = 0.0 if name.endswith("bias") or "norm" in name or p.ndim < 2 else 1.0
            key = (lr_mult, wd_mult, "last_layer" in name)
            coalesced.setdefault(key, {"params": [], "lr_mult": lr_mult, "wd_mult": wd_mult, "last_layer": key[2]})["params"].append(p)
    return list(coalesced.values())


# EMA-update teacher modules from student modules with a single multiplicative decay.
# Params are fused into two _foreach kernels (mul then add) instead of a Python per-tensor loop;
# numerically identical (pt = pt*m + ps*(1-m) per tensor). Called under torch.no_grad() by the caller.
def update_ema(student_module, teacher_module, momentum):
    teacher_params, student_params = list(teacher_module.parameters()), list(student_module.parameters())
    torch._foreach_mul_(teacher_params, momentum)
    torch._foreach_add_(teacher_params, student_params, alpha=1 - momentum)
    for bs, bt in zip(student_module.buffers(), teacher_module.buffers()):
        bt.copy_(bs)


# Orchestrates one pretraining run: setup, train+probe loop, checkpoint, summary.
def main():
    cfg = load_config()
    repo_dir = Path(__file__).resolve().parent
    labless_autosubmit_file = maybe_arm_labless_autosubmit(cfg, repo_dir)
    train_cfg = cfg["train"]
    dino_cfg = cfg["dino"]
    robust_norm = bool(train_cfg.get("robust_norm"))
    patch_name = "jepa" if robust_norm else "ibot"
    fino_cfg = cfg["fino"] if (cfg.get("fino") or {}).get("enabled") else None
    fino_disc = [(factor, float(sign)) for factor, sign in fino_cfg["discrete"]] if fino_cfg else []
    fino_cont = [(factor, float(sign)) for factor, sign in fino_cfg["continuous"]] if fino_cfg else []
    fino_meta = json.loads((Path(cfg["data"]["dataset_dir"]) / "fino_meta.json").read_text()) if fino_cfg else None
    save_every = train_cfg["save_every"]
    save_checkpoints = save_every is not None
    device = torch.device("cuda")
    random.seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])
    torch.manual_seed(train_cfg["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    variant = cfg["model"]["type"]
    student_backbone = load_pretrained(ViT(variant=variant, drop_path_rate=dino_cfg["drop_path_rate"])).to(device)
    teacher_backbone = deepcopy(student_backbone)
    teacher_backbone.train(False)
    for p in teacher_backbone.parameters():
        p.requires_grad = False
    student_dino_head = DINOHead(student_backbone.embed_dim, 131072, dino_cfg["head_hidden_dim"], dino_cfg["head_bottleneck_dim"], 3).to(device)
    student_patch_head = (JEPAPredictor(student_backbone.embed_dim, int(dino_cfg["jepa_pred_depth"]), int(dino_cfg["jepa_pred_width"])) if robust_norm
                          else DINOHead(student_backbone.embed_dim, 131072, dino_cfg["head_hidden_dim"], dino_cfg["head_bottleneck_dim"], 3)).to(device)
    teacher_dino_head = deepcopy(student_dino_head)
    teacher_patch_head = None if robust_norm else deepcopy(student_patch_head)
    for module in (teacher_dino_head,) if robust_norm else (teacher_dino_head, teacher_patch_head):
        for p in module.parameters():
            p.requires_grad = False
    backbone_activated_params = sum(p.numel() for p in student_backbone.parameters() if p.requires_grad)
    predictors = {
        factor: nn.Sequential(nn.Linear(student_backbone.embed_dim, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, fino_meta["cont_dim"].get(factor, 1))).to(device)
        for factor, _ in fino_cont
    }
    # AdamW param groups carry per-parameter LR/WD multipliers (LWD + patch_embed + biases-no-WD).
    param_groups = build_param_groups(student_backbone, student_dino_head, student_patch_head, dino_cfg["layerwise_decay"], dino_cfg["patch_embed_lr_mult"])
    if predictors:
        param_groups.append({"params": [p for model in predictors.values() for p in model.parameters()], "lr_mult": 1.0, "wd_mult": 1.0, "last_layer": False})
    opt = torch.optim.AdamW(param_groups, lr=1.0, betas=(0.9, dino_cfg["adam_beta2"]))
    # One EMA-updated unit-vector bank supplies the FINO target for each discrete factor.
    prototypes = {factor: F.normalize(torch.randn(fino_meta["n"][factor], student_backbone.embed_dim, device=device), dim=-1) for factor, _ in fino_disc}
    step = 0
    batch_size = int(train_cfg["batch_size"])
    max_train_samples = int(train_cfg["max_train_samples"])
    robust_norm_tiles = 6144 if robust_norm else 0
    train_sample_budget = max_train_samples - robust_norm_tiles
    examples_seen = 0
    visible_patch_presentations = 0
    train_flops = 0
    output_dir = Path(cfg["project"]["output_dir"])
    wandb_dir = Path(cfg["project"]["wandb_dir"])
    wandb_name = cfg["project"]["name"]
    if labless_autosubmit_file:
        wandb_name = json.loads(Path(labless_autosubmit_file).read_text()).get("run_name") or wandb_name
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    latest_checkpoint_path = output_dir / "latest.pt"
    # Fresh launches always start from scratch and wipe output_dir.
    resume_path = Path(train_cfg["resume"]) if train_cfg["resume"] else None
    if resume_path is None and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    wandb_meta = None
    if resume_path is not None:
        print(f"{console_prefix()} Resume  loading checkpoint: {resume_path}", flush=True)
        # Resume restores training progress, optimizer state, and wandb identity.
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        student_backbone.load_state_dict(checkpoint["model"])
        teacher_backbone.load_state_dict(checkpoint["model_ema"])
        student_dino_head.load_state_dict(checkpoint["dino_head"])
        teacher_dino_head.load_state_dict(checkpoint["dino_head_ema"])
        student_patch_head.load_state_dict(checkpoint["predictor" if robust_norm else "ibot_head"])
        if teacher_patch_head is not None:
            teacher_patch_head.load_state_dict(checkpoint["ibot_head_ema"])
        opt.load_state_dict(checkpoint["opt"])
        if fino_cfg:
            prototypes = {factor: value.to(device) for factor, value in checkpoint["protos"].items()}
            for factor, predictor in predictors.items():
                predictor.load_state_dict(checkpoint["predictors"][factor])
        step = int(checkpoint["step"])
        examples_seen = int(checkpoint["examples_seen"])
        visible_patch_presentations = int(checkpoint["visible_patch_presentations"])
        train_flops = int(checkpoint["train_flops"])
        wandb_meta = dict(checkpoint["wandb"])
    wandb_init = {
        "project": "nanopath",
        "name": wandb_name,
        "dir": str(wandb_dir),
        "config": cfg,
        "settings": wandb.Settings(
            console="wrap",
            x_file_stream_transmit_interval=5,
        ),
    }
    if wandb_meta is not None:
        wandb_init["id"] = wandb_meta["id"]
        wandb_init["resume"] = "must"
    wandb_run = wandb.init(**wandb_init)
    for key in ("probe/target_flops", "probe/wall_seconds"):
        wandb_run.define_metric(key, hidden=True, overwrite=True)
    print(
        f"{console_prefix()} Run  start: {wandb_name}  "
        f"config: {cfg['config_path']}  batch_size: {batch_size}  max_train_samples: {max_train_samples}  "
        f"seed: {train_cfg['seed']}  "
        f"max_train_flops: {train_cfg['max_train_flops']}  "
        f"probe_count: {cfg['probe']['count']}  warmup_fraction: {dino_cfg['warmup_fraction']}  "
        f"lr: {dino_cfg['lr']}  adam_beta2: {dino_cfg['adam_beta2']}  kde_loss_weight: {dino_cfg['kde_loss_weight']}  "
        f"kde_concentration: {dino_cfg['kde_concentration']}  drop_path: {dino_cfg['drop_path_rate']}  "
        f"layerwise_decay: {dino_cfg['layerwise_decay']}",
        flush=True,
    )
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
    git_remote = subprocess.run(["git", "config", "--get", "remote.origin.url"], cwd=repo_dir, text=True, capture_output=True, check=False).stdout.strip()
    source_id = f"nanopath-source-{wandb_run.id}"
    artifact_ignore = [
        line.strip() for line in (repo_dir / ".gitignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ] + [".git/", "baselines/", "slurm/", "AGENTS.md", "CLAUDE.md"]
    ignored_roots = [output_dir.resolve(), wandb_dir.resolve()]

    def artifact_ignored(path):
        if any(path.resolve().is_relative_to(root) for root in ignored_roots):
            return True
        rel_path = path.relative_to(repo_dir)
        if any(part.startswith(".") for part in rel_path.parts):
            return True
        rel, name = rel_path.as_posix(), path.name
        for pat in artifact_ignore:
            pat = pat.rstrip("/") if pat.endswith("/") else pat
            if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel, pat) or rel == pat or rel.startswith(pat + "/"):
                return True
        return False

    source_files = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = sorted(d for d in dirs if not artifact_ignored(Path(root) / d))
        for name in sorted(files):
            path = Path(root) / name
            if artifact_ignored(path):
                continue
            rel = path.relative_to(repo_dir)
            source_files.append((path, rel))
    source_snapshot_dir = output_dir / "labless_source"
    if source_snapshot_dir.exists():
        shutil.rmtree(source_snapshot_dir)
    for path, rel in source_files:
        target = source_snapshot_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    wandb_meta = {"entity": wandb_run.entity, "project": "nanopath", "id": wandb_run.id, "name": wandb_name, "url": wandb_run.url,
                  "mode": getattr(wandb_run.settings, "mode", ""), "source_artifact": source_id,
                  "source_dir": str(source_snapshot_dir), "git": {"commit": git_commit, "remote": git_remote}}
    train_ds = TCGATileDataset(cfg, is_train=True)
    val_ds = TCGATileDataset(cfg, is_train=False)
    probe_state = prepare_probe_state(cfg, output_dir) if probe_enabled(cfg) else None

    # Train shuffles + drops partials; the loop never starts a batch that would exceed
    # max_train_samples, so every optimizer step keeps the configured batch size.
    loader_kwargs = {
        "batch_size": batch_size,
        "drop_last": True,
        "num_workers": train_cfg["num_workers"],
        "pin_memory": True,
        "prefetch_factor": train_cfg["prefetch_factor"] if train_cfg["num_workers"] > 0 else None,
        "persistent_workers": train_cfg["persistent_workers"] and train_cfg["num_workers"] > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    activation_checkpointing = bool(train_cfg["activation_checkpointing"])
    global_grid = train_cfg["global_size"] // student_backbone.patch_size
    global_patches = global_grid ** 2
    local_patches = (train_cfg["local_size"] // student_backbone.patch_size) ** 2
    last_time = time.time()
    last_examples = examples_seen
    last_visible_patch_presentations = visible_patch_presentations
    last_train_flops = train_flops
    unique_tile_patch_count = (TILE_SIZE // student_backbone.patch_size) ** 2
    seen_ids = {"sample": set(), "slide": set(), "patient": set()}
    pending_ids = {key: set() for key in seen_ids}

    # cpu_state(m) materializes an on-CPU copy of a module's state_dict for torch.save.
    def cpu_state(m): return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    # Full checkpoint (latest.pt) covers explicit train.resume whereas probe checkpoint is a slim
    # weights-only ckpt, given probe.py does not need optimizer or projection heads.
    def checkpoint_payload(next_step, full):
        payload = {"model": cpu_state(student_backbone), "model_ema": cpu_state(teacher_backbone), "step": next_step, "config": cfg}
        if not full:
            return payload
        patch_state = ({"predictor": cpu_state(student_patch_head)} if robust_norm else
                       {"ibot_head": cpu_state(student_patch_head), "ibot_head_ema": cpu_state(teacher_patch_head)})
        return {**payload, "dino_head": cpu_state(student_dino_head), "dino_head_ema": cpu_state(teacher_dino_head), **patch_state,
                "opt": opt.state_dict(), "examples_seen": examples_seen,
                "visible_patch_presentations": visible_patch_presentations, "train_flops": train_flops, "wandb": wandb_meta,
                **({"protos": {factor: value.cpu() for factor, value in prototypes.items()},
                    "predictors": {factor: cpu_state(model) for factor, model in predictors.items()}} if fino_cfg else {})}

    def save_latest_checkpoint(checkpoint_step):
        nonlocal last_saved_step
        print(f"{console_prefix()} Checkpoint  [{checkpoint_step}]  save: latest.pt", flush=True)
        tmp_path = latest_checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint_payload(checkpoint_step, full=True), tmp_path)
        os.replace(tmp_path, latest_checkpoint_path)
        for stale_checkpoint_path in output_dir.glob("step_*.pt"):
            stale_checkpoint_path.unlink()
        last_saved_step = checkpoint_step

    # Count unique tiles/slides/patients for data-coverage diagnostics.
    def flush_unique_counts():
        for key, seen in seen_ids.items():
            seen.update(pending_ids[key])
            pending_ids[key].clear()
        unique_tiles_seen = len(seen_ids["sample"])
        return {
            "unique_slides_seen": len(seen_ids["slide"]),
            "unique_patients_seen": len(seen_ids["patient"]),
            "unique_tiles_seen": unique_tiles_seen,
            "unique_patches_seen": unique_tiles_seen * unique_tile_patch_count,
        }

    # Compute DINO, the configured patch objective, KDE, and optional FINO; validation omits FINO.
    def compute_losses(gf, lf, b, masks, mask_idx, mask_w, t_temp, k_scale, ckpt=False, meta=None):
        with torch.no_grad():
            t = teacher_backbone(gf)
            t_cls = teacher_dino_head(t["cls"]).chunk(train_cfg["global_views"])
            t_prob = sinkhorn(torch.cat((t_cls[1], t_cls[0])), t_temp).view(2, b, -1)
            patch_target = (F.layer_norm(t["patches"].flatten(0, 1), (student_backbone.embed_dim,))[mask_idx] if robust_norm else
                            sinkhorn(teacher_patch_head(t["patches"].flatten(0, 1)[mask_idx]), t_temp))
        sg = student_backbone(gf, masks=masks, checkpoint=ckpt)
        sl = student_backbone(lf, checkpoint=ckpt)
        sg_cls, sl_cls = student_dino_head(sg["cls"]), student_dino_head(sl["cls"])
        L = train_cfg["local_views"]
        local_loss = sum(dino_ce(x, y) for x in sl_cls.chunk(L) for y in t_prob) / (2 * L + 2)
        global_loss = dino_ce(sg_cls, t_prob.flatten(0, 1)) * 2 / (2 * L + 2)
        patch_prediction = (student_patch_head(sg["patches"]).flatten(0, 1)[mask_idx] if robust_norm else
                            student_patch_head(sg["patches"].flatten(0, 1)[mask_idx]))
        if robust_norm:
            patch_loss = F.smooth_l1_loss(patch_prediction, patch_target, reduction="none").mean(-1).mul(mask_w).sum() / max(1, b * 2)
        else:
            patch_loss = -(patch_target * F.log_softmax(patch_prediction / 0.1, dim=-1)).sum(-1).mul(mask_w).sum() / max(1, b * 2)
        kde = dino_cfg["kde_loss_weight"] * k_scale * sum(kde_loss(x, dino_cfg["kde_concentration"]) for x in sg["cls"].chunk(train_cfg["global_views"]))
        meta_loss = None
        if meta is not None:
            # FINO uses signed gradient gates on normalized CLS: prototype CE for discrete metadata,
            # MLP regression for continuous metadata, and EMA teacher features to update prototypes.
            meta_loss = sg["cls"].new_zeros(())
            gamma, discrete, continuous = meta
            student_cls = F.normalize(sg["cls"].float(), dim=-1)
            teacher_cls = F.normalize(t["cls"].float(), dim=-1)
            with torch.autocast(device_type="cuda", enabled=False):
                for j, (factor, sign) in enumerate(fino_disc):
                    labels = discrete[:, j].repeat(train_cfg["global_views"])
                    keep = labels >= 0
                    if keep.any():
                        logits = (GradScale.apply(student_cls[keep], sign * gamma) @ prototypes[factor].T) / 0.023
                        meta_loss = meta_loss + 0.03 * F.cross_entropy(logits, labels[keep])
                        with torch.no_grad():
                            totals = torch.zeros_like(prototypes[factor]).index_add_(0, labels[keep], teacher_cls[keep])
                            counts = torch.zeros(prototypes[factor].shape[0], 1, device=device).index_add_(0, labels[keep], torch.ones_like(teacher_cls[keep, :1]))
                            seen = counts[:, 0] > 0
                            updated = prototypes[factor].clone()
                            updated[seen] = F.normalize(0.99 * updated[seen] + 0.01 * totals[seen] / counts[seen], dim=-1)
                            prototypes[factor] = updated
                for factor, sign in fino_cont:
                    values = continuous[factor].repeat(train_cfg["global_views"], 1)
                    keep = ~torch.isnan(values).any(1)
                    if keep.any():
                        prediction = predictors[factor](GradScale.apply(student_cls[keep], sign * gamma))
                        meta_loss = meta_loss + 0.03 * F.mse_loss(prediction, values[keep])
        return local_loss + global_loss, patch_loss, kde, meta_loss

    # Held-out validation pass: same DINO + patch + KDE losses on `val_batches` of the val split.
    # Schedule terms (teacher_temp, kde_scale) drift over training, so read val curves as same-step
    # diagnostics. RNG is snapshotted/restored so val masks don't perturb the next training step.
    def evaluate(eval_step, eval_teacher_temp, eval_kde_scale):
        for m in (student_backbone, student_dino_head, student_patch_head):
            m.eval()
        py_rng, cpu_rng, cuda_rng = random.getstate(), torch.random.get_rng_state(), torch.cuda.get_rng_state(device)
        random.seed(train_cfg["seed"] + eval_step)
        torch.manual_seed(train_cfg["seed"] + eval_step)
        sums = torch.zeros(4, device=device)
        n_batches = 0
        for vb_idx, vbatch in enumerate(val_loader):
            if vb_idx >= int(train_cfg["val_batches"]):
                break
            vg, vl = vbatch["global_views"].to(device, non_blocking=True), vbatch["local_views"].to(device, non_blocking=True)
            b = vg.shape[0]
            with torch.no_grad(), autocast:
                gf, lf = vg.transpose(0, 1).flatten(0, 1), vl.transpose(0, 1).flatten(0, 1)
                masks, mask_idx, mask_w = (make_block_mask(b * train_cfg["global_views"], global_grid, device, int(dino_cfg["jepa_blocks"]), float(dino_cfg["jepa_block_scale"])) if robust_norm else
                                           make_masks(b * train_cfg["global_views"], global_patches, device))
                dino_l, patch_l, kde_v, _ = compute_losses(gf, lf, b, masks, mask_idx, mask_w, eval_teacher_temp, eval_kde_scale)
            sums += torch.tensor([float(dino_l), float(patch_l), float(kde_v), float(dino_l + patch_l + kde_v)], device=device)
            n_batches += 1
        random.setstate(py_rng)
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        return dict(zip(("dino", "jepa" if robust_norm else "ibot", "kde", "total"), (sums / max(1, n_batches)).tolist()))

    # Ingest completed probe result JSONs into metrics.jsonl and wandb.
    def log_probe_results():
        if probe_state is not None:
            collect_probe_results(probe_state, wandb_run, metrics_path)

    # Queue a probe at `checkpoint_step` for the given sample target; no-op if already done.
    def run_probe_at(checkpoint_step, target_samples):
        if probe_state is None or (probe_state["paths"]["results_dir"] / f"step_{checkpoint_step:07d}.json").exists():
            log_probe_results()
            return
        queue_probe_job(probe_state, checkpoint_payload(checkpoint_step, full=False), checkpoint_step, train_flops, min(1.0, target_samples / max_train_samples))
        log_probe_results()

    # Queue the furthest crossed sample milestone so delayed probes do not run on stale checkpoints.
    def maybe_run_probe(checkpoint_step):
        nonlocal next_probe_idx
        if probe_state is None or next_probe_idx >= len(probe_targets) or examples_seen < probe_targets[next_probe_idx]:
            return
        while next_probe_idx + 1 < len(probe_targets) and examples_seen >= probe_targets[next_probe_idx + 1]:
            next_probe_idx += 1
        run_probe_at(checkpoint_step, probe_targets[next_probe_idx])
        next_probe_idx += 1

    log_probe_results()
    max_train_flops = int(train_cfg["max_train_flops"])
    warmup_train_samples = math.ceil(max_train_samples * dino_cfg["warmup_fraction"])
    # Probe targets are sample milestones: one tile counts once even with many global/local crops.
    probe_count = int(cfg["probe"]["count"]) if probe_enabled(cfg) else 0
    probe_targets = [math.ceil(max_train_samples * (i + 1) / probe_count) for i in range(probe_count)]
    if len(set(probe_targets)) != len(probe_targets):
        raise ValueError(f"probe.count={probe_count} is too large for max_train_samples={max_train_samples}")
    next_probe_idx = 0
    if probe_state is not None:
        completed = [round(float(json.loads(p.read_text()).get("target_fraction", -1)) * max_train_samples) for p in probe_state["paths"]["results_dir"].glob("step_*.json")]
        if completed:
            next_probe_idx = sum(target <= max(completed) for target in probe_targets)
    train_loop_started_at = time.monotonic()
    last_saved_step = step
    last_console_step = step
    last_console_monotonic = time.monotonic()
    data_wait_started_at = time.monotonic()
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if train_cfg["bf16"] else contextlib.nullcontext()
    # Per-step FLOPs are measured once via FlopCounterMode on the first wrapped step (forward +
    # backward + opt.step) and reused for every subsequent step since the shapes don't change.
    # Counts the EMA teacher forward + all objective heads, not just the backbone, so the
    # 1e18 leaderboard cap reflects real GPU work.
    measured_flops_per_step = None

    while examples_seen + batch_size <= train_sample_budget and train_flops < max_train_flops:
        for batch in train_loader:
            if examples_seen + batch_size > train_sample_budget or train_flops >= max_train_flops:
                break
            batch_started_at = time.monotonic()
            data_seconds = batch_started_at - data_wait_started_at
            student_backbone.train()
            student_dino_head.train()
            student_patch_head.train()
            completed_step = step + 1
            should_log = completed_step == 1 or completed_step % train_cfg["log_every"] == 0
            # Data identifiers stay on CPU and feed coverage metrics; image tensors move below.
            for key, batch_key in (("sample", "sample_idx"), ("slide", "slide_id"), ("patient", "patient_id")):
                pending_ids[key].update(int(x) for x in batch[batch_key].tolist())
            global_views, local_views = [batch[key].to(device, non_blocking=True) for key in ("global_views", "local_views")]
            visible_now = batch_size * (train_cfg["global_views"] * global_patches + train_cfg["local_views"] * local_patches)
            # LR warmup uses the 1M-tile sample cap; decay/WD/teacher/freeze/KDE stay on the public FLOP budget.
            frac = min(1.0, train_flops / max_train_flops)
            warmup = min(1.0, examples_seen / max(1, warmup_train_samples))
            if warmup < 1.0:
                lr = dino_cfg["lr"] * warmup
            else:
                lr = cosine_schedule(dino_cfg["lr"], dino_cfg["lr_min"], (frac - dino_cfg["warmup_fraction"]) / max(1e-9, 1 - dino_cfg["warmup_fraction"]))
            wd = cosine_schedule(0.04, 0.2, frac)
            teacher_temp = 0.04 + min(1.0, frac / 0.2727) * (0.07 - 0.04)
            last_layer_lr = 0.0 if frac < dino_cfg["freeze_last_layer_fraction"] else lr
            for group in opt.param_groups:
                base_lr = last_layer_lr if group["last_layer"] else lr
                group["lr"] = base_lr * group["lr_mult"]
                group["weight_decay"] = wd * group["wd_mult"]
            masks, mask_idx, mask_w = (make_block_mask(batch_size * train_cfg["global_views"], global_grid, device, int(dino_cfg["jepa_blocks"]), float(dino_cfg["jepa_block_scale"])) if robust_norm else
                                       make_masks(batch_size * train_cfg["global_views"], global_patches, device))
            kde_scale = min(1.0, max(0.0, (frac - 0.1) / 0.4))
            # Wrap forward + backward + opt.step in FlopCounterMode on the first step only;
            # subsequent steps reuse measured_flops_per_step (fixed shapes => fixed cost).
            flop_ctx = FlopCounterMode(display=False) if measured_flops_per_step is None else contextlib.nullcontext()
            with flop_ctx:
                with autocast:
                    # Crop-major flatten: collate shape is (B, V, 3, H, W) but DINO wants per-crop chunks
                    # so [crop0_img0, crop0_img1, ..., crop1_img0, ...] for clean teacher/student alignment.
                    gf = global_views.transpose(0, 1).flatten(0, 1)
                    lf = local_views.transpose(0, 1).flatten(0, 1)
                    sample_fraction = examples_seen / max_train_samples
                    gamma = fino_cfg["gamma_max"] * (2 / (1 + math.exp(-10 * sample_fraction)) - 1) if fino_cfg else 0.0
                    meta = ((gamma, batch["meta_disc"].to(device, non_blocking=True),
                             {factor: batch[f"mc_{factor}"].to(device, non_blocking=True) for factor, _ in fino_cont}) if fino_cfg else None)
                    dino_loss_value, patch_loss, kde, meta_loss = compute_losses(
                        gf, lf, batch_size, masks, mask_idx, mask_w, teacher_temp, kde_scale,
                        ckpt=activation_checkpointing, meta=meta,
                    )
                    total_loss = dino_loss_value + patch_loss + kde
                    if meta_loss is not None:
                        total_loss = total_loss + meta_loss
                opt.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    [*student_backbone.parameters(), *student_dino_head.parameters(), *student_patch_head.parameters()],
                    dino_cfg["clip_grad"],
                )
                opt.step()
            if measured_flops_per_step is None:
                measured_flops_per_step = int(flop_ctx.get_total_flops())
                print(f"{console_prefix()} measured_flops_per_step: {measured_flops_per_step:,}", flush=True)
            step_train_flops = measured_flops_per_step
            with torch.no_grad():
                m = cosine_schedule(0.994, 1.0, frac)
                update_ema(student_backbone, teacher_backbone, m)
                update_ema(student_dino_head, teacher_dino_head, m)
                if teacher_patch_head is not None:
                    update_ema(student_patch_head, teacher_patch_head, m)
            step_seconds = time.monotonic() - batch_started_at
            examples_seen += batch_size
            visible_patch_presentations += visible_now
            train_flops += step_train_flops
            if should_log:
                reduced = {
                    "dino": float(dino_loss_value.detach()),
                    patch_name: float(patch_loss.detach()),
                    "kde": float(kde.detach()),
                    "total": float(total_loss.detach()),
                }
                unique_counts = flush_unique_counts()
                now = time.time()
                elapsed = max(1e-6, now - last_time)
                items_per_sec = (examples_seen - last_examples) / elapsed
                visible_patches_per_sec = (visible_patch_presentations - last_visible_patch_presentations) / elapsed
                flops_per_sec = (train_flops - last_train_flops) / elapsed
                train_loop_wall_seconds = time.monotonic() - train_loop_started_at
                last_time = now
                last_examples = examples_seen
                last_visible_patch_presentations = visible_patch_presentations
                last_train_flops = train_flops
                gpu_mem_gb = torch.cuda.memory_allocated(device) / (1024**3)
                gpu_peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                console_now = time.monotonic()
                console_gap_ms = 1000.0 * (console_now - last_console_monotonic)
                steps_since_console = max(1, completed_step - last_console_step)
                flop_steps_remaining = math.ceil(max(0, max_train_flops - train_flops) / max(1, step_train_flops))
                sample_steps_remaining = max(0, train_sample_budget - examples_seen) // batch_size
                steps_remaining = min(flop_steps_remaining, sample_steps_remaining)
                total_steps_estimate = completed_step + steps_remaining
                eta_seconds = int(max(0.0, steps_remaining * console_gap_ms / 1000.0 / steps_since_console))
                eta_string = f"{eta_seconds // 3600}:{(eta_seconds % 3600) // 60:02d}:{eta_seconds % 60:02d}"
                current_lr = opt.param_groups[0]["lr"]
                train_log = {
                    "step": completed_step,
                    **reduced,
                    "items_per_sec": items_per_sec,
                    "visible_patches_per_sec": visible_patches_per_sec,
                    "flops_per_sec": flops_per_sec,
                    "wall_seconds": train_loop_wall_seconds,
                    "step_seconds": step_seconds,
                    "data_seconds": data_seconds,
                    "console_gap_ms": console_gap_ms,
                    "eta_seconds": eta_seconds,
                    "flop_fraction": min(1.0, float(train_flops) / float(max_train_flops)),
                    "sample_fraction": min(1.0, float(examples_seen) / float(max_train_samples)),
                    "lr": current_lr,
                    "wd": wd,
                    "teacher_temp": teacher_temp,
                    "teacher_momentum": m,
                    "kde_scale": kde_scale,
                    "batch_size": batch_size,
                    "examples_seen": examples_seen,
                    "visible_patch_presentations": visible_patch_presentations,
                    "train_flops": train_flops,
                    "gpu_mem_gb": gpu_mem_gb,
                    "gpu_peak_mem_gb": gpu_peak_mem_gb,
                    "grad_norm": float(grad_norm.detach()),
                }
                train_log.update(unique_counts)
                print(
                    f"{console_prefix()} Training  "
                    f"[{completed_step}/{total_steps_estimate}]  eta: {eta_string}  gap: {console_gap_ms:.2f} ms  "
                    f"lr: {current_lr:.6f}  total: {reduced['total']:.4f}  "
                    f"dino: {reduced['dino']:.4f}  {patch_name}: {reduced[patch_name]:.4f}  kde: {reduced['kde']:.4f}  "
                    f"grad_norm: {train_log['grad_norm']:.4f}  flops/s: {flops_per_sec:.3e}  "
                    f"time: {step_seconds:.6f}  data: {data_seconds:.6f}  "
                    f"max mem: {int(gpu_peak_mem_gb * 1024)}",
                    flush=True,
                )
                last_console_step = completed_step
                last_console_monotonic = console_now
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(train_log) + "\n")
                wandb_run.log(
                    {f"train/{key}": value for key, value in train_log.items() if key != "step"},
                    step=completed_step,
                )
                log_probe_results()
                torch.cuda.reset_peak_memory_stats(device)
            if save_checkpoints and completed_step % save_every == 0:
                # Atomic rename keeps the previous good latest.pt intact if a
                # kill lands mid-save.
                save_latest_checkpoint(completed_step)
            # Probe at intermediate sample milestones (probe.count > 1); the final probe
            # always runs after the loop exits, regardless of milestones.
            maybe_run_probe(completed_step)
            if completed_step % int(train_cfg["eval_every"]) == 0 or train_flops >= max_train_flops or examples_seen + batch_size > train_sample_budget:
                val = evaluate(completed_step, teacher_temp, kde_scale)
                val_log = {"step": completed_step, **{f"val_{k}": v for k, v in val.items()}}
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(val_log) + "\n")
                wandb_run.log({f"val/{k}": v for k, v in val.items()}, step=completed_step)
                print(f"{console_prefix()} Validation  [{completed_step}]  total: {val['total']:.4f}  dino: {val['dino']:.4f}  {patch_name}: {val[patch_name]:.4f}  kde: {val['kde']:.4f}", flush=True)
                # Reset rate clocks after validation so the next train log is train-rate only.
                last_console_step, last_console_monotonic = completed_step, time.monotonic()
                last_time, last_examples, last_visible_patch_presentations, last_train_flops = time.time(), examples_seen, visible_patch_presentations, train_flops
            step = completed_step
            data_wait_started_at = time.monotonic()
            if train_flops >= max_train_flops or examples_seen + batch_size > train_sample_budget:
                break
    train_loop_wall_seconds = time.monotonic() - train_loop_started_at
    stop_reason = "max_train_flops" if train_flops >= max_train_flops else "max_train_samples"
    final_unique_counts = flush_unique_counts()
    if step > 0:
        # Final probes have their own readers; close pretraining workers before they compete for CPU/IO.
        if train_cfg["num_workers"] > 0 and train_loader._iterator is not None:
            train_loader._iterator._shutdown_workers()
            train_loader._iterator = None
        # Probes get their own short-lived checkpoint via run_probe_at; only persist latest.pt
        # at end-of-run when periodic saving is on (save_every set) so smoke runs leave nothing.
        if robust_norm_tiles:
            # Fit scanner-response directions after optimization so the training trajectory is unchanged.
            started = time.monotonic()
            data_dir = Path(cfg["data"]["dataset_dir"])
            jpegs = [jpeg for shard in range(128) for jpeg in pq.ParquetFile(data_dir / f"shard-{shard:05d}.parquet").read_row_group(0, columns=["jpeg"])["jpeg"].to_pylist()[:48]]
            assert len(jpegs) == robust_norm_tiles
            resize = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor()])
            mean = torch.tensor(cfg["data"]["mean"], device=device).view(1, 3, 1, 1)
            std = torch.tensor(cfg["data"]["std"], device=device).view(1, 3, 1, 1)
            generator = torch.Generator().manual_seed(555)
            gamma = torch.empty(robust_norm_tiles, 3, 1, 1).uniform_(0.8, 1.25, generator=generator)
            gain = torch.empty_like(gamma).uniform_(0.85, 1.18, generator=generator)
            huesat = torch.empty(robust_norm_tiles, 2).uniform_(0, 1, generator=generator)

            @torch.no_grad()
            def robust_features(images):
                with autocast:
                    tokens, taps = teacher_backbone._prepare_tokens((images.to(device) - mean) / std), []
                    for i, block in enumerate(teacher_backbone.blocks):
                        tokens = block(tokens)
                        if i in (2, 4, 6, 8, 11):
                            taps.append(teacher_backbone.norm(tokens)[:, 0])
                    tokens = teacher_backbone.norm(tokens)
                return torch.stack([tokens[:, 0], tokens[:, 1 + teacher_backbone.registers :].mean(1), *taps], 1).float().cpu()

            bases, deltas = [], []
            for start in range(0, robust_norm_tiles, batch_size):
                base = torch.stack([resize(Image.open(io.BytesIO(jpeg)).convert("RGB")) for jpeg in jpegs[start : start + batch_size]])
                base_features = robust_features(base)
                views = (
                    base.clamp_min(1e-6) ** gamma[start : start + batch_size],
                    (base * gain[start : start + batch_size]).clamp(0, 1),
                    torch.stack([TF.adjust_saturation(TF.adjust_hue(tile, float((hs[0] - 0.5) * 0.1)), float(0.7 + hs[1] * 0.7)) for tile, hs in zip(base, huesat[start : start + batch_size])]),
                )
                bases.append(base_features)
                deltas.extend(robust_features(view) - base_features for view in views)
            base_features, delta_features = torch.cat(bases), torch.cat(deltas)
            directions = torch.linalg.svd((delta_features - delta_features.mean(0)).movedim(1, 0).to(device), full_matrices=False)[2]
            for model in (student_backbone, teacher_backbone):
                model.rn_mu.copy_(base_features.mean(0)[:2]); model.rn_v.copy_(directions[:2, :32])
                model.pf_mu.copy_(base_features.mean(0)[2:]); model.pf_v.copy_(directions[2:, :1])
                model.rn_fitted.fill_(True); model.pf_fitted.fill_(True)
            print(f"{console_prefix()} RobustNorm  [{step}]  fitted rank 32 + per-tap rank 1 from {len(jpegs)} tiles in {time.monotonic() - started:.0f}s", flush=True)
        if save_checkpoints and step != last_saved_step:
            save_latest_checkpoint(step)
        run_probe_at(step, examples_seen)
    log_probe_results()
    # Summary is the small, stable artifact downstream scripts and humans compare across runs.
    summary = {
        "project": cfg["project"]["name"],
        "family": cfg["project"]["family"],
        "recipe_id": cfg["project"]["recipe_id"],
        "config_path": cfg["config_path"],
        "train_seed": int(train_cfg["seed"]),
        "data_split_seed": int(cfg["data"]["split_seed"]),
        "wandb": wandb_meta,
        "slurm_job_id": slurm_job_id,
        "backbone_activated_params": backbone_activated_params,
        "batch_size": batch_size,
        "max_train_samples": max_train_samples,
        "max_train_flops": max_train_flops,
        "train_loop_wall_seconds": train_loop_wall_seconds,
        "stop_reason": stop_reason,
        "steps_completed": step,
        "tile_presentations": examples_seen,
        "visible_patch_presentations": visible_patch_presentations,
        **final_unique_counts,
        "train_flops": train_flops,
        "flop_fraction": min(1.0, float(train_flops) / float(max_train_flops)),
        "sample_fraction": min(1.0, float(examples_seen) / float(max_train_samples)),
        # Average throughput over the train loop; wall time is diagnostic, not an eligibility cap.
        "flops_per_sec": train_flops / max(1.0, train_loop_wall_seconds),
        "visible_patches_per_sec": visible_patch_presentations / max(1.0, train_loop_wall_seconds),
        "warmup_fraction": dino_cfg["warmup_fraction"],
        "warmup_train_samples": warmup_train_samples,
        "lr": dino_cfg["lr"],
        "adam_beta2": dino_cfg["adam_beta2"],
        "kde_loss_weight": dino_cfg["kde_loss_weight"],
        "kde_concentration": dino_cfg["kde_concentration"],
        "drop_path_rate": dino_cfg["drop_path_rate"],
        "layerwise_decay": dino_cfg["layerwise_decay"],
        "probe_target_samples": probe_targets,
        "probe_target_fractions": [None if max_train_samples == 0 else target / max_train_samples for target in probe_targets],
        **({} if probe_state is None else completed_probe_summary(output_dir)),
    }
    if probe_state is not None and "final_score" not in summary:
        raise ValueError("probe.enabled is true but final_score is missing; check probe.count, probe failures, and final checkpoint scheduling")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"{console_prefix()} Summary  "
        f"steps: {step}  train_wall: {train_loop_wall_seconds:.2f}s  "
        f"final_score: {summary.get('final_score')}",
        flush=True,
    )
    for key, value in summary.items():
        wandb_run.summary[key] = value
    wandb_run.finish()
    finish_labless_autosubmit(labless_autosubmit_file, output_dir, repo_dir)


if __name__ == "__main__":
    main()
