# Pretraining entry point. Continual DINOv2 pretraining of the Meta
# register-token student/teacher pair selected by `cfg["model"]["type"]`
# on TCGA tiles. Three loss terms:
# DINO CLS self-distillation (Sinkhorn-Knopp centred teacher targets),
# iBOT masked-patch self-distillation, and a cross-rank KDE uniformity term
# on the L2-normalised CLS tokens. YAML drives the few knobs we tune
# (backbone variant, LR, KDE weight + concentration, FLOP budget, batch
# shape); every other DINOv2 hyper is a module constant below — see LOG.md
# for the sweeps that picked those values. Researchers changing objectives
# should start at the loss block in main(); changing data preprocessing
# starts in dataloader.py; changing downstream comparisons starts in probe.py.

import contextlib
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import numpy as np
import pynvml
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.flop_counter import FlopCounterMode

from dataloader import TCGATileDataset, TILE_SIZE
from model import DINOHead, DinoV2ViT, load_dinov2_pretrained
from probe import (
    completed_probe_summary,
    collect_probe_results,
    prepare_probe_state,
    probe_enabled,
    queue_probe_job,
)


# Recipe constants (Tanishq's 3-seed-confirmed `kde_vmf_xgpu` values).
DROP_PATH_RATE = 0.1
LAYERWISE_DECAY = 0.7
PATCH_EMBED_LR_MULT = 0.2
LR_MIN = 1e-6
WEIGHT_DECAY = 0.04
WEIGHT_DECAY_END = 0.2
MOMENTUM_TEACHER = 0.994
FINAL_MOMENTUM_TEACHER = 1.0
TEACHER_TEMP = 0.07
WARMUP_TEACHER_TEMP = 0.04
TEACHER_TEMP_WARMUP_FRACTION = 0.2727
STUDENT_TEMP = 0.1
CLIP_GRAD = 3.0
FREEZE_LAST_LAYER_FRACTION = 0.0091
MASK_SAMPLE_PROBABILITY = 0.5
MASK_RATIO_MIN_MAX = (0.1, 0.45)
HEAD_N_PROTOTYPES = 131072
HEAD_HIDDEN_DIM = 2048
HEAD_BOTTLENECK_DIM = 384
HEAD_NLAYERS = 3
KDE_START_FRACTION = 0.1
KDE_WARMUP_FRACTION = 0.4
KDE_ALL_GATHER = True
WARMUP_FLOP_FRACTION = 0.0909


# Prefix every console line with wall time and job/process id so SLURM logs are easy to scan.
def console_prefix(): return f"{time.strftime('%H:%M:%S')} {os.environ.get('SLURM_JOB_ID', str(os.getpid()))}"


# Read the YAML recipe and fail before any GPU work if the parquet tile dataset is absent.
# expandvars is necessary to resolve `$USER` for checked-in configs.
def load_config():
    if len(sys.argv) != 2:
        raise ValueError("usage: python train.py <config.yaml>")
    cfg = yaml.safe_load(os.path.expandvars(Path(sys.argv[1]).read_text()))
    cfg["config_path"] = str(Path(sys.argv[1]).resolve())
    dataset_dir = Path(cfg["data"]["dataset_dir"])
    if not any(dataset_dir.glob("shard-*.parquet")):
        raise FileNotFoundError(
            f"No parquet shards (shard-*.parquet) under {dataset_dir}. Pull the 4M-tile "
            f"parquet dataset from medarc/nanopath on HF by running "
            f"`python prepare.py {cfg['config_path']} download=True`. Follow the data setup in "
            f"README.md before launching train.py."
        )
    return cfg


# Cosine schedule from `start` to `end` over fractional progress in [0, 1].
def cosine_schedule(start, end, frac):
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * min(1.0, max(0.0, frac))))


# Sinkhorn-Knopp centring across global batch (and ranks) used as DINO/iBOT teacher targets.
def sinkhorn(x, temp, world_size):
    q = torch.exp(x.float() / temp).t()
    b = q.shape[1] * world_size
    k = q.shape[0]
    s = q.sum()
    if world_size > 1:
        dist.all_reduce(s)
    q /= s
    for _ in range(3):
        rows = q.sum(1, keepdim=True)
        if world_size > 1:
            dist.all_reduce(rows)
        q /= rows * k
        q /= q.sum(0, keepdim=True) * b
    return (q * b).t()


# Cross-entropy between teacher distribution and softmax(student / student_temp).
def dino_ce(student, teacher, student_temp):
    return -(teacher * F.log_softmax(student / student_temp, dim=-1)).sum(-1).mean()


# KDE uniformity loss on L2-normalised CLS tokens; cross-rank gather widens the kernel support
# (the dominant signal in Tanishq's sweep). Detached gathers from other ranks keep gradients local.
def kde_loss(x, concentration, world_size):
    x = F.normalize(x, p=2, dim=-1)
    if KDE_ALL_GATHER and world_size > 1:
        gathered = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(gathered, x.detach())
        gathered[dist.get_rank()] = x
        x = torch.cat(gathered)
    sim = concentration * (x @ x.T)
    sim.fill_diagonal_(-float("inf"))
    return torch.logsumexp(sim, dim=1).mean() - math.log(max(1, sim.shape[1] - 1))


# Sample iBOT masking pattern: per-image bernoulli on whether to mask, then random patch ratio.
def make_masks(batch, patches, prob, ratio, device):
    masks = torch.zeros(batch, patches, dtype=torch.bool, device=device)
    for i in range(batch):
        if random.random() < prob:
            masks[i, torch.randperm(patches, device=device)[: int(patches * random.uniform(*ratio))]] = True
    idx = masks.flatten().nonzero().flatten()
    weights = (1 / masks.sum(-1).clamp(min=1)).unsqueeze(-1).expand_as(masks)[masks]
    return masks, idx, weights


# AdamW parameter groups with layer-wise LR decay on the backbone (Tanishq's recipe):
# block i gets lr * LAYERWISE_DECAY^(depth - 1 - i); patch_embed gets the deepest decay
# multiplied by PATCH_EMBED_LR_MULT; biases and norms get no weight decay; the head's
# final weight-norm last_layer parameters get an LR-freeze for the first FREEZE_LAST_LAYER_FRACTION.
def build_param_groups(student_backbone, student_dino_head, student_ibot_head):
    depth = len(student_backbone.blocks)
    groups = []
    modules = ((student_backbone, "backbone"), (student_dino_head, "dino_head"), (student_ibot_head, "ibot_head"))
    for module, kind in modules:
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            lr_mult = 1.0
            if kind == "backbone" and name.startswith("blocks."):
                lr_mult = LAYERWISE_DECAY ** (depth - 1 - int(name.split(".")[1]))
            elif kind == "backbone" and name.startswith("patch_embed."):
                lr_mult = (LAYERWISE_DECAY ** depth) * PATCH_EMBED_LR_MULT
            wd_mult = 0.0 if name.endswith("bias") or "norm" in name or p.ndim < 2 else 1.0
            groups.append({"params": [p], "lr_mult": lr_mult, "wd_mult": wd_mult, "last_layer": "last_layer" in name})
    return groups


# EMA-update teacher modules from student modules with a single multiplicative decay.
def update_ema(student_module, teacher_module, momentum):
    for ps, pt in zip(student_module.parameters(), teacher_module.parameters()):
        pt.mul_(momentum).add_(ps.detach(), alpha=1 - momentum)
    for bs, bt in zip(student_module.buffers(), teacher_module.buffers()):
        bt.copy_(bs)


# Orchestrates one pretraining run: setup, train+probe loop, checkpoint, summary.
def main():
    cfg = load_config()
    train_cfg = cfg["train"]
    dino_cfg = cfg["dino"]
    save_every = train_cfg["save_every"]
    save_checkpoints = save_every is not None
    # torchrun sets WORLD_SIZE/RANK/LOCAL_RANK; absence of those variables means local single-process.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    # `stop_requested` flips True when SLURM signals SIGUSR1 (wallclock approaching) or when
    # train_flops crosses max_train_flops; both paths then exit the loop and run final save + probes.
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    signal.signal(signal.SIGUSR1, request_stop)
    if distributed:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=45), device_id=device)
    # Rank-offset seeds keep DDP workers from sampling identical augmentation streams.
    random.seed(train_cfg["seed"] + rank)
    np.random.seed(train_cfg["seed"] + rank)
    torch.manual_seed(train_cfg["seed"] + rank)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    # Build student/teacher backbones; rank 0 downloads pretrained weights once (others read the cache).
    variant = cfg["model"]["type"]
    if rank == 0:
        load_dinov2_pretrained(DinoV2ViT(variant=variant))
    if distributed:
        dist.barrier()
    student_backbone = load_dinov2_pretrained(DinoV2ViT(variant=variant, drop_path_rate=DROP_PATH_RATE)).to(device)
    teacher_backbone = deepcopy(student_backbone)
    teacher_backbone.train(False)
    for p in teacher_backbone.parameters():
        p.requires_grad = False
    student_dino_head = DINOHead(student_backbone.embed_dim, HEAD_N_PROTOTYPES, HEAD_HIDDEN_DIM, HEAD_BOTTLENECK_DIM, HEAD_NLAYERS).to(device)
    student_ibot_head = DINOHead(student_backbone.embed_dim, HEAD_N_PROTOTYPES, HEAD_HIDDEN_DIM, HEAD_BOTTLENECK_DIM, HEAD_NLAYERS).to(device)
    teacher_dino_head = deepcopy(student_dino_head)
    teacher_ibot_head = deepcopy(student_ibot_head)
    for m in (teacher_dino_head, teacher_ibot_head):
        for p in m.parameters():
            p.requires_grad = False
    backbone_activated_params = sum(p.numel() for p in student_backbone.parameters() if p.requires_grad)
    # AdamW param groups carry per-parameter LR/WD multipliers (LWD + patch_embed + biases-no-WD).
    opt = torch.optim.AdamW(build_param_groups(student_backbone, student_dino_head, student_ibot_head), lr=1.0, betas=(0.9, 0.999))
    step = 0
    global_batch_size = int(train_cfg["global_batch_size"])
    if global_batch_size % world_size != 0:
        raise ValueError(f"global_batch_size={global_batch_size} not divisible by world_size={world_size}")
    batch_size = global_batch_size // world_size
    examples_seen = 0
    visible_patch_presentations = 0
    train_flops = 0
    output_dir = Path(cfg["project"]["output_dir"])
    wandb_dir = Path(cfg["project"]["wandb_dir"])
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    latest_checkpoint_path = output_dir / "latest.pt"
    # Pick a resume source. Priority: explicit train.resume override, then any
    # latest.pt already in output_dir (so a SLURM auto-requeue picks up where
    # the previous job was killed). With no resume source we treat this as a
    # fresh launch and wipe output_dir to avoid mixing stale artifacts.
    if train_cfg["resume"]:
        resume_path = Path(train_cfg["resume"])
    elif latest_checkpoint_path.exists():
        resume_path = latest_checkpoint_path
    else:
        resume_path = None
    if resume_path is None:
        if rank == 0 and output_dir.exists():
            shutil.rmtree(output_dir)
        if distributed:
            dist.barrier()
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    wandb_meta = None
    if resume_path is not None:
        if rank == 0:
            print(f"{console_prefix()} Resume  loading checkpoint: {resume_path}", flush=True)
        # Resume restores training progress, optimizer state, and wandb identity.
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        student_backbone.load_state_dict(checkpoint["model"])
        teacher_backbone.load_state_dict(checkpoint["model_ema"])
        student_dino_head.load_state_dict(checkpoint["dino_head"])
        student_ibot_head.load_state_dict(checkpoint["ibot_head"])
        teacher_dino_head.load_state_dict(checkpoint["dino_head_ema"])
        teacher_ibot_head.load_state_dict(checkpoint["ibot_head_ema"])
        opt.load_state_dict(checkpoint["opt"])
        step = int(checkpoint["step"])
        examples_seen = int(checkpoint["examples_seen"])
        visible_patch_presentations = int(checkpoint["visible_patch_presentations"])
        train_flops = int(checkpoint["train_flops"])
        wandb_meta = dict(checkpoint["wandb"])
    if rank == 0:
        wandb_init = {
            "project": "nanopath",
            "name": cfg["project"]["name"],
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
            f"{console_prefix()} Run  start: {cfg['project']['name']}  "
            f"config: {cfg['config_path']}  batch_size: {batch_size}  max_train_flops: {train_cfg['max_train_flops']}  "
            f"probe_count: {cfg['probe']['count']}  warmup_flop_fraction: {WARMUP_FLOP_FRACTION}  "
            f"lr: {dino_cfg['lr']}  kde_loss_weight: {dino_cfg['kde_loss_weight']}  "
            f"kde_concentration: {dino_cfg['kde_concentration']}  drop_path: {DROP_PATH_RATE}  "
            f"layerwise_decay: {LAYERWISE_DECAY}",
            flush=True,
        )
        source_artifact = wandb.Artifact(f"nanopath-source-{wandb_run.id}", type="code")
        repo_dir = Path(__file__).resolve().parent
        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = sorted(d for d in dirs if d not in {".git", ".venv", "__pycache__", ".claude"})
            for name in sorted(files):
                path = Path(root) / name
                source_artifact.add_file(str(path), name=str(path.relative_to(repo_dir)))
        wandb_run.log_artifact(source_artifact)
        wandb_meta = {"project": "nanopath", "id": wandb_run.id, "name": cfg["project"]["name"]}
    else:
        wandb_run = None
    train_ds = TCGATileDataset(cfg, is_train=True)
    val_ds = TCGATileDataset(cfg, is_train=False)
    probe_state = prepare_probe_state(cfg, output_dir) if rank == 0 and probe_enabled(cfg) else None

    # Train shuffles + drops last partial batch. Val is sequential and `drop_last=True` so every rank
    # processes the same number of batches per evaluate() call (required for the all_reduce inside).
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True) if distributed else None
    loader_kwargs = dict(batch_size=batch_size, drop_last=True, num_workers=train_cfg["num_workers"], pin_memory=True,
                         prefetch_factor=train_cfg["prefetch_factor"] if train_cfg["num_workers"] > 0 else None,
                         persistent_workers=train_cfg["persistent_workers"] and train_cfg["num_workers"] > 0)
    train_loader = DataLoader(train_ds, shuffle=sampler is None, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, sampler=val_sampler, **loader_kwargs)

    if distributed:
        student_backbone = DDP(student_backbone, device_ids=[local_rank], find_unused_parameters=False)
        student_dino_head = DDP(student_dino_head, device_ids=[local_rank], find_unused_parameters=False)
        student_ibot_head = DDP(student_ibot_head, device_ids=[local_rank], find_unused_parameters=False)
    root_backbone = student_backbone.module if distributed else student_backbone
    root_dino_head = student_dino_head.module if distributed else student_dino_head
    root_ibot_head = student_ibot_head.module if distributed else student_ibot_head

    activation_checkpointing = bool(train_cfg["activation_checkpointing"])
    global_patches = (train_cfg["global_size"] // root_backbone.patch_size) ** 2
    local_patches = (train_cfg["local_size"] // root_backbone.patch_size) ** 2
    last_time = time.time()
    last_examples = examples_seen
    last_visible_patch_presentations = visible_patch_presentations
    last_train_flops = train_flops
    unique_tile_patch_count = (TILE_SIZE // root_backbone.patch_size) ** 2
    seen_ids = {"sample": set(), "slide": set(), "patient": set()}
    pending_ids = {key: set() for key in seen_ids}

    # cpu_state(m) materializes an on-CPU copy of a module's state_dict for torch.save.
    def cpu_state(m): return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    # Full checkpoint (latest.pt) covers everything needed to resume; probe checkpoint is a slim
    # weights-only payload since probe.py never reads the optimizer or the projection heads.
    def checkpoint_payload(next_step, full):
        payload = {"model": cpu_state(root_backbone), "model_ema": cpu_state(teacher_backbone), "step": next_step, "config": cfg}
        if not full:
            return payload
        return {**payload, "dino_head": cpu_state(root_dino_head), "ibot_head": cpu_state(root_ibot_head),
                "dino_head_ema": cpu_state(teacher_dino_head), "ibot_head_ema": cpu_state(teacher_ibot_head),
                "opt": opt.state_dict(), "examples_seen": examples_seen,
                "visible_patch_presentations": visible_patch_presentations, "train_flops": train_flops, "wandb": wandb_meta}

    def save_latest_checkpoint(checkpoint_step):
        nonlocal last_saved_step
        print(f"{console_prefix()} Checkpoint  [{checkpoint_step}]  save: latest.pt", flush=True)
        tmp_path = latest_checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint_payload(checkpoint_step, full=True), tmp_path)
        os.replace(tmp_path, latest_checkpoint_path)
        for stale_checkpoint_path in output_dir.glob("step_*.pt"):
            stale_checkpoint_path.unlink()
        last_saved_step = checkpoint_step

    # Count unique tiles/slides/patients across ranks for data-coverage diagnostics.
    def flush_unique_counts():
        payload = {key: list(values) for key, values in pending_ids.items()}
        if distributed:
            gathered = [None] * world_size
            dist.all_gather_object(gathered, payload)
        else:
            gathered = [payload]
        if rank == 0:
            for entry in gathered:
                for key in seen_ids:
                    seen_ids[key].update(entry[key])
        for values in pending_ids.values():
            values.clear()
        if rank == 0:
            unique_tiles_seen = len(seen_ids["sample"])
            return {
                "unique_slides_seen": len(seen_ids["slide"]),
                "unique_patients_seen": len(seen_ids["patient"]),
                "unique_tiles_seen": unique_tiles_seen,
                "unique_patches_seen": unique_tiles_seen * unique_tile_patch_count,
            }
        return None

    # Compute (dino_loss, ibot_loss, kde) for one batch of (gf, lf) crops with the given masks +
    # schedule values. Used by both the train step (DDP-wrapped student modules) and evaluate()
    # (bare modules, no_grad). `bb`/`dh`/`ih` is the student trio; teacher_* are always the bare EMA copies.
    def compute_losses(gf, lf, b, masks, mask_idx, mask_w, t_temp, k_scale, bb, dh, ih, ckpt=False):
        with torch.no_grad():
            t = teacher_backbone(gf)
            t_cls = teacher_dino_head(t["x_norm_clstoken"]).chunk(train_cfg["global_views"])
            t_prob = sinkhorn(torch.cat((t_cls[1], t_cls[0])), t_temp, world_size).view(2, b, -1)
            t_patch_prob = sinkhorn(teacher_ibot_head(t["x_norm_patchtokens"].flatten(0, 1)[mask_idx]), t_temp, world_size)
        sg = bb(gf, masks=masks, checkpoint=ckpt)
        sl = bb(lf, checkpoint=ckpt)
        sg_cls, sl_cls = dh(sg["x_norm_clstoken"]), dh(sl["x_norm_clstoken"])
        L = train_cfg["local_views"]
        local_loss = sum(dino_ce(x, y, STUDENT_TEMP) for x in sl_cls.chunk(L) for y in t_prob) / (2 * L + 2)
        global_loss = dino_ce(sg_cls, t_prob.flatten(0, 1), STUDENT_TEMP) * 2 / (2 * L + 2)
        s_patch = ih(sg["x_norm_patchtokens"].flatten(0, 1)[mask_idx])
        ibot_loss = -(t_patch_prob * F.log_softmax(s_patch / STUDENT_TEMP, dim=-1)).sum(-1).mul(mask_w).sum() / max(1, b * 2)
        kde = dino_cfg["kde_loss_weight"] * k_scale * sum(kde_loss(x, dino_cfg["kde_concentration"], world_size) for x in sg["x_norm_clstoken"].chunk(train_cfg["global_views"]))
        return local_loss + global_loss, ibot_loss, kde

    # Held-out validation pass: same DINO + iBOT + KDE losses on `val_batches` of the val split.
    # Schedule terms (teacher_temp, kde_scale) drift over training, so read val curves as same-step
    # diagnostics. RNG is snapshotted/restored so val masks don't perturb the next training step.
    def evaluate(eval_step, eval_teacher_temp, eval_kde_scale):
        for m in (student_backbone, student_dino_head, student_ibot_head):
            m.eval()
        py_rng, cpu_rng, cuda_rng = random.getstate(), torch.random.get_rng_state(), torch.cuda.get_rng_state(device)
        random.seed(train_cfg["seed"] + eval_step)
        torch.manual_seed(train_cfg["seed"] + eval_step + rank)
        sums = torch.zeros(4, device=device)
        n_batches = 0
        for vb_idx, vbatch in enumerate(val_loader):
            if vb_idx >= int(train_cfg["val_batches"]):
                break
            vg, vl = vbatch["global_views"].to(device, non_blocking=True), vbatch["local_views"].to(device, non_blocking=True)
            b = vg.shape[0]
            with torch.no_grad(), autocast:
                gf, lf = vg.transpose(0, 1).flatten(0, 1), vl.transpose(0, 1).flatten(0, 1)
                masks, mask_idx, mask_w = make_masks(b * train_cfg["global_views"], global_patches, MASK_SAMPLE_PROBABILITY, MASK_RATIO_MIN_MAX, device)
                dino_l, ibot_l, kde_v = compute_losses(gf, lf, b, masks, mask_idx, mask_w, eval_teacher_temp, eval_kde_scale, root_backbone, root_dino_head, root_ibot_head)
            sums += torch.tensor([float(dino_l), float(ibot_l), float(kde_v), float(dino_l + ibot_l + kde_v)], device=device)
            n_batches += 1
        random.setstate(py_rng)
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        if distributed:
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            sums = sums / world_size
        return dict(zip(("dino", "ibot", "kde", "total"), (sums / max(1, n_batches)).tolist()))

    # Ingest completed probe result JSONs into metrics.jsonl and wandb.
    def log_probe_results():
        if probe_state is not None:
            collect_probe_results(probe_state, wandb_run, metrics_path)

    # Queue a probe at `checkpoint_step` for the given FLOP target; no-op if already done.
    def run_probe_at(checkpoint_step, target_flops):
        if probe_state is None or (probe_state["paths"]["results_dir"] / f"step_{checkpoint_step:07d}.json").exists():
            log_probe_results()
            return
        queue_probe_job(probe_state, checkpoint_payload(checkpoint_step, full=False), checkpoint_step, target_flops, min(1.0, target_flops / max_train_flops))
        log_probe_results()

    # Queue the furthest crossed FLOP milestone so delayed probes do not run on stale checkpoints.
    def maybe_run_probe(checkpoint_step):
        nonlocal next_probe_idx
        if probe_state is None or next_probe_idx >= len(probe_targets) or train_flops < probe_targets[next_probe_idx]:
            return
        while next_probe_idx + 1 < len(probe_targets) and train_flops >= probe_targets[next_probe_idx + 1]:
            next_probe_idx += 1
        run_probe_at(checkpoint_step, probe_targets[next_probe_idx])
        next_probe_idx += 1

    if rank == 0:
        log_probe_results()
    if distributed:
        dist.barrier()
    max_train_flops = int(train_cfg["max_train_flops"])
    warmup_train_flops = math.ceil(max_train_flops * WARMUP_FLOP_FRACTION)
    # Probe targets are FLOP milestones, not step milestones, so comparisons survive batch-size changes.
    probe_count = int(cfg["probe"]["count"]) if probe_enabled(cfg) else 0
    probe_targets = [math.ceil(max_train_flops * (i + 1) / probe_count) for i in range(probe_count)]
    if len(set(probe_targets)) != len(probe_targets):
        raise ValueError(f"probe.count={probe_count} is too large for max_train_flops={max_train_flops}")
    next_probe_idx = 0
    if probe_state is not None:
        completed = [int(json.loads(p.read_text()).get("target_flops", -1)) for p in probe_state["paths"]["results_dir"].glob("step_*.json")]
        if completed:
            next_probe_idx = sum(target <= max(completed) for target in probe_targets)
    train_loop_started_at = time.monotonic()
    if train_flops >= max_train_flops:
        stop_requested = True
    last_saved_step = step if resume_path is not None else 0
    last_console_step = step
    last_console_monotonic = time.monotonic()
    data_wait_started_at = time.monotonic()
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if train_cfg["bf16"] else contextlib.nullcontext()
    # Per-step FLOPs are measured once via FlopCounterMode on the first wrapped step (forward +
    # backward + opt.step) and reused for every subsequent step since the shapes don't change.
    # Counts the EMA teacher forward + DINO/iBOT projection heads, not just the backbone, so the
    # 1e18 leaderboard cap reflects real GPU work. Resumed runs re-measure on their first step.
    measured_flops_per_step = None

    while not stop_requested:
        if distributed:
            train_loader.sampler.set_epoch(step + train_cfg["seed"])
        for batch in train_loader:
            if stop_requested:
                break
            batch_started_at = time.monotonic()
            data_seconds = batch_started_at - data_wait_started_at
            student_backbone.train()
            student_dino_head.train()
            student_ibot_head.train()
            completed_step = step + 1
            should_log = completed_step == 1 or completed_step % train_cfg["log_every"] == 0
            # Data identifiers stay on CPU and feed coverage metrics; image tensors move below.
            for key, batch_key in (("sample", "sample_idx"), ("slide", "slide_id"), ("patient", "patient_id")):
                pending_ids[key].update(int(x) for x in batch[batch_key].tolist())
            global_views, local_views = [batch[key].to(device, non_blocking=True) for key in ("global_views", "local_views")]
            current_batch = global_views.shape[0]
            global_batch = current_batch * world_size
            visible_now = global_batch * (train_cfg["global_views"] * global_patches + train_cfg["local_views"] * local_patches)
            # Linear warmup then cosine decay to LR_MIN, all keyed off train_flops not step count.
            frac = min(1.0, train_flops / max_train_flops)
            warmup = min(1.0, train_flops / max(1, warmup_train_flops))
            if warmup < 1.0:
                lr = dino_cfg["lr"] * warmup
            else:
                lr = cosine_schedule(dino_cfg["lr"], LR_MIN, (frac - WARMUP_FLOP_FRACTION) / max(1e-9, 1 - WARMUP_FLOP_FRACTION))
            wd = cosine_schedule(WEIGHT_DECAY, WEIGHT_DECAY_END, frac)
            teacher_temp = WARMUP_TEACHER_TEMP + min(1.0, frac / TEACHER_TEMP_WARMUP_FRACTION) * (TEACHER_TEMP - WARMUP_TEACHER_TEMP)
            last_layer_lr = 0.0 if frac < FREEZE_LAST_LAYER_FRACTION else lr
            for group in opt.param_groups:
                base_lr = last_layer_lr if group["last_layer"] else lr
                group["lr"] = base_lr * group["lr_mult"]
                group["weight_decay"] = wd * group["wd_mult"]
            # iBOT mask sample is in the same per-rank order as the global tokens (b * 2 globals).
            masks, mask_idx, mask_w = make_masks(current_batch * train_cfg["global_views"], global_patches, MASK_SAMPLE_PROBABILITY, MASK_RATIO_MIN_MAX, device)
            kde_scale = min(1.0, max(0.0, (frac - KDE_START_FRACTION) / KDE_WARMUP_FRACTION))
            # Wrap forward + backward + opt.step in FlopCounterMode on the first step only;
            # subsequent steps reuse measured_flops_per_step (fixed shapes => fixed cost).
            flop_ctx = FlopCounterMode(display=False) if measured_flops_per_step is None else contextlib.nullcontext()
            with flop_ctx:
                with autocast:
                    # Crop-major flatten: collate shape is (B, V, 3, H, W) but DINO wants per-crop chunks
                    # so [crop0_img0, crop0_img1, ..., crop1_img0, ...] for clean teacher/student alignment.
                    gf = global_views.transpose(0, 1).flatten(0, 1)
                    lf = local_views.transpose(0, 1).flatten(0, 1)
                    # Pass DDP-wrapped student modules so backward hooks fire and grads sync across ranks.
                    dino_loss_value, ibot_loss, kde = compute_losses(
                        gf, lf, current_batch, masks, mask_idx, mask_w, teacher_temp, kde_scale,
                        student_backbone, student_dino_head, student_ibot_head, ckpt=activation_checkpointing,
                    )
                    total_loss = dino_loss_value + ibot_loss + kde
                opt.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    [*root_backbone.parameters(), *root_dino_head.parameters(), *root_ibot_head.parameters()],
                    CLIP_GRAD,
                )
                opt.step()
            if measured_flops_per_step is None:
                measured_flops_per_step = int(flop_ctx.get_total_flops()) * world_size
                # prevent ranks disagreeing on `train_flops >= max_train_flops`, where one rank enters
                # the eval pass while others don't — leading to NCCL collectives that never match.
                if distributed:
                    t = torch.tensor([measured_flops_per_step], device=device)
                    dist.broadcast(t, src=0)
                    measured_flops_per_step = int(t.item())
                if rank == 0:
                    print(f"{console_prefix()} measured_flops_per_step: {measured_flops_per_step:,}", flush=True)
            step_train_flops = measured_flops_per_step
            with torch.no_grad():
                m = cosine_schedule(MOMENTUM_TEACHER, FINAL_MOMENTUM_TEACHER, frac)
                update_ema(root_backbone, teacher_backbone, m)
                update_ema(root_dino_head, teacher_dino_head, m)
                update_ema(root_ibot_head, teacher_ibot_head, m)
            step_seconds = time.monotonic() - batch_started_at
            examples_seen += global_batch
            visible_patch_presentations += visible_now
            train_flops += step_train_flops
            reduced = None
            if should_log:
                # Average scalar training losses across ranks so rank 0 logs global batch behavior.
                if distributed:
                    reduced = torch.tensor(
                        [
                            float(dino_loss_value.detach()),
                            float(ibot_loss.detach()),
                            float(kde.detach()),
                            float(total_loss.detach()),
                        ],
                        device=device,
                    )
                    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                    reduced = (reduced / world_size).tolist()
                else:
                    reduced = [float(dino_loss_value.detach()), float(ibot_loss.detach()), float(kde.detach()), float(total_loss.detach())]
                reduced = dict(zip(("dino", "ibot", "kde", "total"), reduced))
            unique_counts = flush_unique_counts() if should_log else None
            if rank == 0 and should_log:
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
                gpu_util_pct = float(pynvml.nvmlDeviceGetUtilizationRates(nvml_handle).gpu)
                gpu_mem_gb = torch.cuda.memory_allocated(device) / (1024**3)
                gpu_peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                console_now = time.monotonic()
                console_gap_ms = 1000.0 * (console_now - last_console_monotonic)
                steps_since_console = max(1, completed_step - last_console_step)
                steps_remaining = max(0, math.ceil((max_train_flops - train_flops) / max(1, step_train_flops)))
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
                    "lr": current_lr,
                    "wd": wd,
                    "teacher_temp": teacher_temp,
                    "teacher_momentum": m,
                    "kde_scale": kde_scale,
                    "batch_size": batch_size,
                    "global_batch_size": global_batch,
                    "examples_seen": examples_seen,
                    "visible_patch_presentations": visible_patch_presentations,
                    "train_flops": train_flops,
                    "gpu_util_pct": gpu_util_pct,
                    "gpu_mem_gb": gpu_mem_gb,
                    "gpu_peak_mem_gb": gpu_peak_mem_gb,
                    "grad_norm": float(grad_norm.detach()),
                }
                train_log.update(unique_counts)
                print(
                    f"{console_prefix()} Training  "
                    f"[{completed_step}/{total_steps_estimate}]  eta: {eta_string}  gap: {console_gap_ms:.2f} ms  "
                    f"lr: {current_lr:.6f}  total: {reduced['total']:.4f}  "
                    f"dino: {reduced['dino']:.4f}  ibot: {reduced['ibot']:.4f}  kde: {reduced['kde']:.4f}  "
                    f"grad_norm: {train_log['grad_norm']:.4f}  flops/s: {flops_per_sec:.3e}  gpu: {gpu_util_pct:.0f}  "
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
            if rank == 0 and save_checkpoints and completed_step % save_every == 0:
                # Atomic rename: a SLURM kill or scontrol requeue mid-save
                # leaves the previous good latest.pt intact rather than a
                # half-written file the requeued job would refuse to load.
                save_latest_checkpoint(completed_step)
            if rank == 0:
                # Probe at intermediate FLOP milestones (probe.count > 1); the final probe
                # always runs after the loop exits, regardless of milestones.
                maybe_run_probe(completed_step)
            if completed_step % int(train_cfg["eval_every"]) == 0 or train_flops >= max_train_flops:
                val = evaluate(completed_step, teacher_temp, kde_scale)
                if rank == 0:
                    val_log = {"step": completed_step, **{f"val_{k}": v for k, v in val.items()}}
                    with metrics_path.open("a") as handle:
                        handle.write(json.dumps(val_log) + "\n")
                    wandb_run.log({f"val/{k}": v for k, v in val.items()}, step=completed_step)
                    print(f"{console_prefix()} Validation  [{completed_step}]  total: {val['total']:.4f}  dino: {val['dino']:.4f}  ibot: {val['ibot']:.4f}  kde: {val['kde']:.4f}", flush=True)
            step = completed_step
            data_wait_started_at = time.monotonic()
            if train_flops >= max_train_flops:
                stop_requested = True
            if stop_requested:
                break
    train_loop_wall_seconds = time.monotonic() - train_loop_started_at
    stop_reason = "max_train_flops" if train_flops >= max_train_flops else "wallclock"
    final_unique_counts = flush_unique_counts()
    if distributed:
        dist.barrier()
    if step > 0:
        # Final probes have their own readers; close pretraining workers before they compete for CPU/IO.
        if train_cfg["num_workers"] > 0:
            if train_loader._iterator is not None:
                train_loader._iterator._shutdown_workers()
                train_loader._iterator = None
        if rank == 0:
            # Probes get their own short-lived checkpoint via run_probe_at; only persist latest.pt
            # at end-of-run when periodic saving is on (save_every set) so smoke runs leave nothing.
            if save_checkpoints and step != last_saved_step:
                save_latest_checkpoint(step)
            run_probe_at(step, train_flops)
        if distributed:
            dist.barrier()
    if rank == 0:
        log_probe_results()
        # Summary is the small, stable artifact downstream scripts and humans compare across runs.
        summary = {
            "project": cfg["project"]["name"],
            "family": cfg["project"]["family"],
            "recipe_id": cfg["project"]["recipe_id"],
            "config_path": cfg["config_path"],
            "slurm_job_id": slurm_job_id,
            "backbone_activated_params": backbone_activated_params,
            "world_size": world_size,
            "batch_size_per_rank": batch_size,
            "global_batch_size": batch_size * world_size,
            "max_train_flops": max_train_flops,
            "train_loop_wall_seconds": train_loop_wall_seconds,
            "stop_reason": stop_reason,
            "steps_completed": step,
            "tile_presentations": examples_seen,
            "visible_patch_presentations": visible_patch_presentations,
            **final_unique_counts,
            "train_flops": train_flops,
            "flop_fraction": min(1.0, float(train_flops) / float(max_train_flops)),
            # Average throughput over the whole train loop; useful for spotting submissions that
            # left the FLOP budget unspent (low flops/sec from a wallclock stop with slack).
            "flops_per_sec": train_flops / max(1.0, train_loop_wall_seconds),
            "visible_patches_per_sec": visible_patch_presentations / max(1.0, train_loop_wall_seconds),
            "warmup_flop_fraction": WARMUP_FLOP_FRACTION,
            "warmup_train_flops": warmup_train_flops,
            "lr": dino_cfg["lr"],
            "kde_loss_weight": dino_cfg["kde_loss_weight"],
            "kde_concentration": dino_cfg["kde_concentration"],
            "drop_path_rate": DROP_PATH_RATE,
            "layerwise_decay": LAYERWISE_DECAY,
            "probe_target_flops": probe_targets,
            "probe_target_fractions": [None if max_train_flops == 0 else target / max_train_flops for target in probe_targets],
            **({} if probe_state is None else completed_probe_summary(output_dir)),
        }
        if probe_state is not None and "final_probe_score" not in summary:
            raise ValueError("probe.enabled is true but final_probe_score is missing; check probe.count, probe failures, and final checkpoint scheduling")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"{console_prefix()} Summary  "
            f"steps: {step}  train_wall: {train_loop_wall_seconds:.2f}s  "
            f"final_probe_score: {summary.get('final_probe_score')}",
            flush=True,
        )
        for key in summary.keys():
            wandb_run.summary[key] = summary[key]
        wandb_run.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
