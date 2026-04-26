# Pretraining entry point. Runs the JEPA + SIGReg loop end-to-end: parse a YAML
# config (e.g. configs/small.yaml), build the TCGA dataloader, construct the
# NanoPathFM model and EMA copy, optimize across DDP ranks under torchrun, log
# to wandb + metrics.jsonl, write a single rolling latest.pt checkpoint, and
# queue inline downstream probes (probe.py) at evenly spaced FLOP milestones.
# Launch: `torchrun --standalone --nproc_per_node N train.py <config.yaml>`.

import contextlib
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pynvml
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dataloader import RandomTCGADataset, SAMPLE_LIST_PATCH_SIZE, prepare_sample_list_offsets
from model import SIGReg, NanoPathFM
from probe import (
    completed_probe_summary,
    collect_probe_results,
    prepare_probe_state,
    probe_enabled,
    queue_probe_job,
)


torch.multiprocessing.set_sharing_strategy("file_system")

EVAL_KEYS = ("jepa_pred", "sig", "total")
TRAIN_KEYS = ("jepa_pred", "sig", "total", "proj_std")


def console_prefix(): return f"{time.strftime('%H:%M:%S')} {os.environ.get('SLURM_JOB_ID', str(os.getpid()))}"


def validate_ema_config(ema_decay, probe_model_weights):
    if ema_decay < 0.0 or ema_decay >= 1.0:
        raise ValueError(f"train.ema_decay must be in [0, 1), got {ema_decay}")
    if probe_model_weights not in {"raw", "ema"}:
        raise ValueError(f"probe.model_weights must be raw or ema, got {probe_model_weights}")
    if probe_model_weights == "ema" and ema_decay == 0.0:
        raise ValueError("probe.model_weights is ema but train.ema_decay is 0")


def clone_state_to_device(state, device):
    return {key: value.detach().to(device).clone() for key, value in state.items()}


def clone_state_to_cpu(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def validate_state_keys(label, reference, candidate):
    if set(reference) != set(candidate):
        missing = sorted(set(reference) - set(candidate))
        extra = sorted(set(candidate) - set(reference))
        raise ValueError(f"{label} state keys do not match model state; missing={missing}, extra={extra}")


def update_ema_state(ema_state, model_state, ema_decay):
    ema_floating = []
    model_floating = []
    for key, value in model_state.items():
        source = value.detach()
        if torch.is_floating_point(ema_state[key]):
            ema_floating.append(ema_state[key])
            model_floating.append(source)
        else:
            ema_state[key].copy_(source)
    torch._foreach_mul_(ema_floating, ema_decay)
    torch._foreach_add_(ema_floating, model_floating, alpha=1.0 - ema_decay)


def load_config():
    if len(sys.argv) != 2:
        raise ValueError("usage: python train.py <config.yaml>")
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    cfg["config_path"] = str(Path(sys.argv[1]).resolve())
    return cfg


def loss_terms(sigreg, proj, train_cfg, sigreg_generator, proj_sigreg=None):
    # jepa_pred_loss is the JEPA-style multi-view consistency term on the projector outputs:
    # if global/local views of the same tile disagree, this grows.
    # sig_loss is the anti-collapse regularizer on those same projector outputs.
    # total_loss is the objective we backprop through.
    proj_sigreg = proj if proj_sigreg is None else proj_sigreg
    jepa_pred_loss = (proj - proj.mean(dim=1, keepdim=True)).square().mean()
    sig_loss = sigreg(proj_sigreg.transpose(0, 1), generator=sigreg_generator)
    total_loss = train_cfg["lambda_jepa_pred"] * jepa_pred_loss + train_cfg["lambda_sig"] * sig_loss
    return jepa_pred_loss, sig_loss, total_loss


def evaluate(model, sigreg, loader, cfg, device, world_size):
    model.eval()
    train_cfg = cfg["train"]
    losses = [0.0] * len(EVAL_KEYS)
    batches = 0
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" and train_cfg["bf16"] else contextlib.nullcontext()
    eval_generator = torch.Generator(device=device.type)
    eval_generator.manual_seed(train_cfg["seed"])
    with torch.no_grad():
        for batch in loader:
            with autocast:
                global_views = batch["global_views"].to(device, non_blocking=True)
                local_views = batch["local_views"].to(device, non_blocking=True)
                proj = model(global_views, local_views, train_cfg)
                jepa_pred_loss, sig_loss, total_loss = loss_terms(
                    sigreg,
                    proj,
                    train_cfg,
                    eval_generator,
                    torch.cat(dist_nn.all_gather(proj), dim=0) if world_size > 1 else proj,
                )
            for i, value in enumerate((jepa_pred_loss, sig_loss, total_loss)):
                losses[i] += value.item()
            batches += 1
            if batches >= train_cfg["val_batches"]:
                break
    if world_size > 1:
        tensor = torch.tensor([*losses, batches], device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        losses = tensor[:-1].tolist()
        batches = int(tensor[-1].item())
    return {key: value / batches for key, value in zip(EVAL_KEYS, losses)}


def main():
    cfg = load_config()
    train_cfg = cfg["train"]
    ema_decay = float(train_cfg["ema_decay"])
    probe_model_weights = str(cfg["probe"]["model_weights"])
    validate_ema_config(ema_decay, probe_model_weights)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=45))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(train_cfg["device"])
    random.seed(train_cfg["seed"] + rank)
    np.random.seed(train_cfg["seed"] + rank)
    torch.manual_seed(train_cfg["seed"] + rank)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        pynvml.nvmlInit()
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    else:
        nvml_handle = None
    model = NanoPathFM(cfg).to(device)
    model.sigreg = SIGReg().to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith("bias") or "norm" in name or "register_tokens" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": train_cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1.0,
        betas=(0.9, 0.95),
    )
    step = 0
    global_batch_size = int(train_cfg["global_batch_size"])
    if global_batch_size % world_size != 0:
        raise ValueError(f"global_batch_size={global_batch_size} not divisible by world_size={world_size}")
    batch_size = global_batch_size // world_size
    lr = train_cfg["lr_ref"]
    examples_seen = 0
    visible_patch_presentations = 0
    train_flops = 0
    best_val_total = float("inf")
    best_val_mean_probe_score = float("-inf")
    best_val_mean_probe_score_step = -1
    best_probe_scores = {}
    output_dir = Path(cfg["project"]["output_dir"])
    wandb_dir = Path("/data/nanopath/wandb")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_stdout_path = os.environ.get("NANOPATH_SLURM_STDOUT")
    slurm_stderr_path = os.environ.get("NANOPATH_SLURM_STDERR")
    # Fresh launches fully replace the run directory so repeated use of a checked-in
    # output_dir like /data/nanopath/small never trips over stale artifacts.
    if train_cfg["resume"] is None:
        if rank == 0 and output_dir.exists():
            shutil.rmtree(output_dir)
        if distributed:
            dist.barrier()
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    latest_checkpoint_path = output_dir / "latest.pt"
    wandb_meta = None
    resume_path = train_cfg["resume"]
    checkpoint = None
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["opt"])
        (
            step,
            lr,
            best_val_total,
            examples_seen,
            visible_patch_presentations,
            train_flops,
            best_val_mean_probe_score,
            best_val_mean_probe_score_step,
        ) = (
            int(checkpoint["step"]),
            float(checkpoint["lr"]),
            float(checkpoint["best_val_total"]),
            int(checkpoint["examples_seen"]),
            int(checkpoint["visible_patch_presentations"]),
            int(checkpoint["train_flops"]),
            float(checkpoint["best_val_mean_probe_score"]),
            int(checkpoint["best_val_mean_probe_score_step"]),
        )
        best_probe_scores = dict(checkpoint["best_probe_scores"])
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
            f"eval_every: {train_cfg['eval_every']}  probe_count: {cfg['probe']['count']}  "
            f"warmdown_flop_fraction: {train_cfg['warmdown_flop_fraction']}  final_lr_frac: {train_cfg['final_lr_frac']}  "
            f"ema_decay: {ema_decay}  probe_model_weights: {probe_model_weights}",
            flush=True,
        )
        source_artifact = wandb.Artifact(f"nanopath-source-{wandb_run.id}", type="code")
        repo_dir = Path(__file__).resolve().parent
        for path in sorted({*repo_dir.rglob("*"), *repo_dir.glob(".*")}):
            if not path.is_file() or any(part in {".git", ".venv", "__pycache__", ".claude"} for part in path.parts):
                continue
            source_artifact.add_file(str(path), name=str(path.relative_to(repo_dir)))
        wandb_run.log_artifact(source_artifact)
        wandb_meta = {"project": "nanopath", "id": wandb_run.id, "name": cfg["project"]["name"]}
    else:
        wandb_run = None
    if rank == 0:
        prepare_sample_list_offsets(cfg)
    if distributed:
        dist.barrier()
    train_ds = RandomTCGADataset(cfg, cfg["data"]["train_split"])
    val_ds = RandomTCGADataset(cfg, cfg["data"]["val_split"])
    probe_state = prepare_probe_state(cfg, output_dir) if rank == 0 and probe_enabled(cfg) else None

    def make_loader(dataset, shuffle, batch):
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=shuffle) if distributed else None
        return DataLoader(
            dataset,
            batch_size=batch,
            shuffle=sampler is None and shuffle,
            sampler=sampler,
            drop_last=shuffle,
            num_workers=train_cfg["num_workers"],
            pin_memory=device.type == "cuda",
            prefetch_factor=train_cfg["prefetch_factor"] if train_cfg["num_workers"] > 0 else None,
            persistent_workers=train_cfg["persistent_workers"] and train_cfg["num_workers"] > 0,
        )

    train_loader = make_loader(train_ds, True, batch_size)
    val_loader = make_loader(val_ds, False, batch_size)

    def shutdown_loader_workers(loader):
        if train_cfg["num_workers"] == 0:
            return
        iterator = loader._iterator
        if iterator is not None:
            iterator._shutdown_workers()
            loader._iterator = None

    def shutdown_data_loaders():
        # Final probes have their own readers; close pretraining workers before they compete for CPU and IO.
        shutdown_loader_workers(train_loader)
        shutdown_loader_workers(val_loader)

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    root_model = model.module if distributed else model
    model = torch.compile(model, dynamic=False)
    model_state = root_model.state_dict()
    ema_state = clone_state_to_device(model_state, device)
    if resume_path is not None:
        if "model_ema" not in checkpoint:
            raise KeyError(f"resume checkpoint is missing model_ema: {resume_path}")
        validate_state_keys("model_ema", model_state, checkpoint["model_ema"])
        ema_state = clone_state_to_device(checkpoint["model_ema"], device)
    global_patches = (train_cfg["global_size"] // cfg["model"]["patch_size"]) ** 2
    local_patches = (train_cfg["local_size"] // cfg["model"]["patch_size"]) ** 2
    last_time = time.time()
    last_examples = examples_seen
    last_visible_patch_presentations = visible_patch_presentations
    last_train_flops = train_flops
    unique_tile_patch_count = (SAMPLE_LIST_PATCH_SIZE // cfg["model"]["patch_size"]) ** 2
    seen_sample_ids = set()
    seen_slide_ids = set()
    seen_patient_ids = set()
    pending_sample_ids = set()
    pending_slide_ids = set()
    pending_patient_ids = set()

    def checkpoint_payload(next_step):
        return {
            "model": clone_state_to_cpu(root_model.state_dict()),
            "model_ema": clone_state_to_cpu(ema_state),
            "opt": opt.state_dict(),
            "step": next_step,
            "best_val_total": best_val_total,
            "best_val_mean_probe_score": best_val_mean_probe_score,
            "best_val_mean_probe_score_step": best_val_mean_probe_score_step,
            "examples_seen": examples_seen,
            "visible_patch_presentations": visible_patch_presentations,
            "train_flops": train_flops,
            "best_probe_scores": best_probe_scores,
            "lr": lr,
            "ema_decay": ema_decay,
            "probe_model_weights": probe_model_weights,
            "wandb": wandb_meta,
            "config": cfg,
        }

    def probe_checkpoint_payload(next_step):
        return {
            "model": clone_state_to_cpu(root_model.state_dict()),
            "model_ema": clone_state_to_cpu(ema_state),
            "step": next_step,
            "ema_decay": ema_decay,
            "probe_model_weights": probe_model_weights,
            "config": cfg,
        }

    def flush_unique_counts():
        nonlocal pending_sample_ids, pending_slide_ids, pending_patient_ids
        payload = {"samples": list(pending_sample_ids), "slides": list(pending_slide_ids), "patients": list(pending_patient_ids)}
        if distributed:
            gathered = [None] * world_size
            dist.all_gather_object(gathered, payload)
        else:
            gathered = [payload]
        if rank == 0:
            for entry in gathered:
                seen_sample_ids.update(entry["samples"])
                seen_slide_ids.update(entry["slides"])
                seen_patient_ids.update(entry["patients"])
        pending_sample_ids.clear()
        pending_slide_ids.clear()
        pending_patient_ids.clear()
        if rank == 0:
            unique_tiles_seen = len(seen_sample_ids)
            return {
                "unique_slides_seen": len(seen_slide_ids),
                "unique_patients_seen": len(seen_patient_ids),
                "unique_tiles_seen": unique_tiles_seen,
                "unique_patches_seen": unique_tiles_seen * unique_tile_patch_count,
            }
        return None

    def log_probe_results():
        nonlocal best_val_mean_probe_score, best_val_mean_probe_score_step, best_probe_scores
        if probe_state is None:
            return
        best_val_mean_probe_score, best_val_mean_probe_score_step, best_probe_scores = collect_probe_results(
            probe_state, wandb_run, metrics_path, output_dir, best_val_mean_probe_score, best_val_mean_probe_score_step, best_probe_scores
        )

    def log_val(step_value, val, event=None):
        payload = {"step": step_value, **{f"val_{key}": val[key] for key in EVAL_KEYS}}
        if event is not None:
            payload["event"] = event
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")
        wandb_run.log({f"val/{key}": val[key] for key in EVAL_KEYS}, step=step_value)
        print(
            f"{console_prefix()} Validation  [{step_value}]  "
            f"event: {event or 'scheduled'}  "
            f"jepa_pred: {val['jepa_pred']:.6f}  sig: {val['sig']:.6f}  total: {val['total']:.6f}  "
            f"best_total: {best_val_total:.6f}",
            flush=True,
        )

    def maybe_run_probe(checkpoint_step):
        nonlocal next_probe_idx
        if probe_state is None or next_probe_idx >= len(probe_targets) or train_flops < probe_targets[next_probe_idx]:
            return
        target_idx = next_probe_idx
        while target_idx + 1 < len(probe_targets) and train_flops >= probe_targets[target_idx + 1]:
            target_idx += 1
        queue_probe_job(
            probe_state,
            probe_checkpoint_payload(checkpoint_step),
            checkpoint_step,
            probe_targets[target_idx],
            probe_targets[target_idx] / max_train_flops,
        )
        next_probe_idx = target_idx + 1

    if rank == 0 and probe_state is not None:
        log_probe_results()
    if distributed:
        dist.barrier()
    max_train_flops = int(train_cfg["max_train_flops"])
    warmup_flop_fraction = float(train_cfg["warmup_flop_fraction"])
    warmup_train_flops = math.ceil(max_train_flops * warmup_flop_fraction)
    warmdown_train_flops = round(max_train_flops * float(train_cfg["warmdown_flop_fraction"]))
    final_lr_frac = float(train_cfg["final_lr_frac"])
    probe_targets = []
    next_probe_idx = 0
    if probe_enabled(cfg):
        probe_count = int(cfg["probe"]["count"])
        if probe_count < 1:
            raise ValueError(f"probe.count must be at least 1, got {probe_count}")
        probe_targets = [math.ceil(max_train_flops * (i + 1) / probe_count) for i in range(probe_count)]
        if len(set(probe_targets)) != len(probe_targets):
            raise ValueError(f"probe.count={probe_count} is too large for max_train_flops={max_train_flops}")
        if probe_state is not None:
            completed_probe_flops = []
            for result_path in probe_state["paths"]["results_dir"].glob("step_*.json"):
                result = json.loads(result_path.read_text())
                if "target_flops" in result:
                    completed_probe_flops.append(int(result["target_flops"]))
            if len(completed_probe_flops) > 0:
                max_completed_probe_flops = max(completed_probe_flops)
                next_probe_idx = min(len(probe_targets), sum(target <= max_completed_probe_flops for target in probe_targets))
    train_loop_started_at = time.monotonic()
    train_loop_wall_seconds = 0.0
    stop_reason = "max_train_flops" if train_flops >= max_train_flops else None
    last_eval_step = step if math.isfinite(best_val_total) else -1
    last_saved_step = step if resume_path is not None else 0
    last_console_step = step
    last_console_monotonic = time.monotonic()
    data_wait_started_at = time.monotonic()
    train_log_count = 0
    train_lr_sum = 0.0
    train_batch_sum = 0.0
    train_step_time_sum = 0.0
    train_data_time_sum = 0.0
    train_loss_sums = {key: 0.0 for key in TRAIN_KEYS}

    while stop_reason is None:
        if distributed:
            train_loader.sampler.set_epoch(step + train_cfg["seed"])
        for batch in train_loader:
            batch_started_at = time.monotonic()
            data_seconds = batch_started_at - data_wait_started_at
            model.train()
            completed_step = step + 1
            should_log = completed_step == 1 or completed_step % train_cfg["log_every"] == 0
            pending_sample_ids.update(int(x) for x in batch["sample_idx"].tolist())
            pending_slide_ids.update(int(x) for x in batch["slide_id"].tolist())
            pending_patient_ids.update(int(x) for x in batch["patient_id"].tolist())
            global_views, local_views = [batch[key].to(device, non_blocking=True) for key in ("global_views", "local_views")]
            current_batch = global_views.shape[0]
            global_batch = current_batch * world_size
            visible_now = global_batch * (train_cfg["global_views"] * global_patches + train_cfg["local_views"] * local_patches)
            step_train_flops = int(6 * model_params * visible_now)
            lr_flops = min(max_train_flops, train_flops + step_train_flops)
            lr_multiplier = min(1.0, lr_flops / warmup_train_flops)
            lr_multiplier = min(lr_multiplier, final_lr_frac + (1.0 - final_lr_frac) * (max_train_flops - lr_flops) / warmdown_train_flops)
            for group in opt.param_groups:
                group["lr"] = lr * lr_multiplier
            autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" and train_cfg["bf16"] else contextlib.nullcontext()
            sigreg_generator = torch.Generator(device=device.type)
            sigreg_generator.manual_seed(train_cfg["seed"] + step)
            with autocast:
                proj = model(global_views, local_views, train_cfg)
                jepa_pred_loss, sig_loss, total_loss = loss_terms(
                    root_model.sigreg,
                    proj,
                    train_cfg,
                    sigreg_generator,
                    torch.cat(dist_nn.all_gather(proj), dim=0) if distributed else proj,
                )
                proj_std = proj.float().reshape(-1, proj.shape[-1]).std(dim=0).mean() if should_log else None
            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            clip_grad_norm = nn.utils.clip_grad_norm_(root_model.parameters(), train_cfg["grad_clip"]) if train_cfg["grad_clip"] > 0 else None
            grad_norm = None
            param_norm = None
            grad_param_ratio = None
            grad_clip_scale = None
            if should_log:
                if clip_grad_norm is None:
                    grad_sq = 0.0
                    for param in root_model.parameters():
                        if param.grad is not None:
                            grad_sq += param.grad.detach().float().square().sum().item()
                    grad_norm = grad_sq ** 0.5
                else:
                    grad_norm = float(clip_grad_norm.detach().float().item())
                param_sq = 0.0
                for param in root_model.parameters():
                    param_sq += param.detach().float().square().sum().item()
                param_norm = param_sq**0.5
                grad_param_ratio = grad_norm / max(param_norm, 1e-12)
                grad_clip_scale = min(1.0, train_cfg["grad_clip"] / max(grad_norm, 1e-12)) if train_cfg["grad_clip"] > 0 else 1.0
            opt.step()
            update_ema_state(ema_state, root_model.state_dict(), ema_decay)
            step_seconds = time.monotonic() - batch_started_at
            examples_seen += global_batch
            visible_patch_presentations += visible_now
            train_flops += step_train_flops
            reduced = None
            if should_log:
                if distributed:
                    reduced = torch.tensor(
                        [
                            jepa_pred_loss.item(),
                            sig_loss.item(),
                            total_loss.item(),
                            proj_std.item(),
                        ],
                        device=device,
                    )
                    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                    reduced = (reduced / world_size).tolist()
                else:
                    reduced = [jepa_pred_loss.item(), sig_loss.item(), total_loss.item(), proj_std.item()]
                reduced = dict(zip(TRAIN_KEYS, reduced))
            unique_counts = flush_unique_counts() if should_log else None
            if rank == 0 and should_log:
                finite_mpp = batch["sampled_mpp"].float()
                finite_mpp = finite_mpp[torch.isfinite(finite_mpp)]
                sampled_mpp_mean = float(finite_mpp.mean().item()) if finite_mpp.numel() > 0 else float("nan")
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
                gpu_util_pct = float(pynvml.nvmlDeviceGetUtilizationRates(nvml_handle).gpu) if nvml_handle is not None else 0.0
                gpu_mem_gb = torch.cuda.memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
                gpu_peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
                console_now = time.monotonic()
                console_gap_ms = 1000.0 * (console_now - last_console_monotonic)
                steps_since_console = max(1, completed_step - last_console_step)
                steps_remaining = max(0, math.ceil((max_train_flops - train_flops) / max(1, step_train_flops)))
                total_steps_estimate = completed_step + steps_remaining
                eta_seconds = int(max(0.0, steps_remaining * console_gap_ms / 1000.0 / steps_since_console))
                eta_string = f"{eta_seconds // 3600}:{(eta_seconds % 3600) // 60:02d}:{eta_seconds % 60:02d}"
                train_log_count += 1
                current_lr = opt.param_groups[0]["lr"]
                train_lr_sum += current_lr
                train_batch_sum += float(batch_size)
                train_step_time_sum += step_seconds
                train_data_time_sum += data_seconds
                for key in TRAIN_KEYS:
                    train_loss_sums[key] += float(reduced[key])
                train_log = {
                    "step": completed_step,
                    **reduced,
                    "sampled_mpp_mean": sampled_mpp_mean,
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
                    "lr_multiplier": lr_multiplier,
                    "batch_size": batch_size,
                    "global_batch_size": global_batch,
                    "examples_seen": examples_seen,
                    "visible_patch_presentations": visible_patch_presentations,
                    "train_flops": train_flops,
                    "gpu_util_pct": gpu_util_pct,
                    "gpu_mem_gb": gpu_mem_gb,
                    "gpu_peak_mem_gb": gpu_peak_mem_gb,
                }
                train_log.update(
                    {
                        "grad_norm": grad_norm,
                        "param_norm": param_norm,
                        "grad_param_ratio": grad_param_ratio,
                        "grad_clip_scale": grad_clip_scale,
                    }
                )
                train_log.update(unique_counts)
                print(
                    f"{console_prefix()} Training  "
                    f"[{completed_step}/{total_steps_estimate}]  eta: {eta_string}  gap: {console_gap_ms:.2f} ms  "
                    f"lr: {current_lr:.6f} ({train_lr_sum / train_log_count:.6f})  lrm: {lr_multiplier:.4f}  "
                    f"current_batch_size: {batch_size:.4f} ({train_batch_sum / train_log_count:.4f})  "
                    f"total_loss: {reduced['total']:.4f} ({train_loss_sums['total'] / train_log_count:.4f})  "
                    f"jepa_pred_loss: {reduced['jepa_pred']:.4f} ({train_loss_sums['jepa_pred'] / train_log_count:.4f})  "
                    f"sig_loss: {reduced['sig']:.4f} ({train_loss_sums['sig'] / train_log_count:.4f})  "
                    f"proj_std: {reduced['proj_std']:.4f} ({train_loss_sums['proj_std'] / train_log_count:.4f})  "
                    f"grad_norm: {grad_norm:.4f}  flops/s: {flops_per_sec:.3e}  gpu: {gpu_util_pct:.0f}  "
                    f"time: {step_seconds:.6f} ({train_step_time_sum / train_log_count:.6f})  "
                    f"data: {data_seconds:.6f} ({train_data_time_sum / train_log_count:.6f})  "
                    f"max mem: {int(gpu_peak_mem_gb * 1024)}",
                    flush=True,
                )
                last_console_step = completed_step
                last_console_monotonic = console_now
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(train_log) + "\n")
                wandb_run.log(
                    {
                        f"train/{key}": value
                        for key, value in train_log.items()
                        if key != "step"
                    },
                    step=completed_step,
                )
                if probe_state is not None:
                    log_probe_results()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
            if completed_step % train_cfg["eval_every"] == 0:
                if rank == 0:
                    print(f"{console_prefix()} Validation  [{completed_step}]  start", flush=True)
                val = evaluate(model, root_model.sigreg, val_loader, cfg, device, world_size)
                if val["total"] < best_val_total:
                    best_val_total = val["total"]
                last_eval_step = completed_step
                if rank == 0:
                    log_val(completed_step, val)
                    maybe_run_probe(completed_step)
                    if probe_state is not None:
                        log_probe_results()
                if distributed:
                    dist.barrier()
            if rank == 0 and completed_step % train_cfg["save_every"] == 0:
                print(f"{console_prefix()} Checkpoint  [{completed_step}]  save: latest.pt", flush=True)
                torch.save(checkpoint_payload(completed_step), latest_checkpoint_path)
                for stale_checkpoint_path in output_dir.glob("step_*.pt"):
                    stale_checkpoint_path.unlink()
                last_saved_step = completed_step
            step = completed_step
            train_loop_wall_seconds = time.monotonic() - train_loop_started_at
            data_wait_started_at = time.monotonic()
            if train_flops >= max_train_flops:
                stop_reason = "max_train_flops"
                break
            if stop_reason is not None:
                break
    train_loop_wall_seconds = time.monotonic() - train_loop_started_at
    final_unique_counts = flush_unique_counts()
    if distributed:
        dist.barrier()
    if step > 0 and last_eval_step != step:
        if rank == 0:
            print(f"{console_prefix()} Validation  [{step}]  start final_eval", flush=True)
        val = evaluate(model, root_model.sigreg, val_loader, cfg, device, world_size)
        if val["total"] < best_val_total:
            best_val_total = val["total"]
        last_eval_step = step
        shutdown_data_loaders()
        if rank == 0:
            log_val(step, val, "final_eval")
            maybe_run_probe(step)
            if probe_state is not None:
                log_probe_results()
        if distributed:
            dist.barrier()
    if not math.isfinite(best_val_total):
        raise ValueError("run finished without a finite best_val_total; check max_train_flops, eval_every, validation data, and loss stability")
    if rank == 0:
        if probe_state is not None:
            log_probe_results()
        if step > 0 and step != last_saved_step:
            print(f"{console_prefix()} Checkpoint  [{step}]  save: latest.pt", flush=True)
            torch.save(checkpoint_payload(step), latest_checkpoint_path)
            for stale_checkpoint_path in output_dir.glob("step_*.pt"):
                stale_checkpoint_path.unlink()
            last_saved_step = step
        summary = {
            "project": cfg["project"]["name"],
            "family": cfg["project"]["family"],
            "recipe_id": cfg["project"]["recipe_id"],
            "config_path": cfg["config_path"],
            "slurm_job_id": slurm_job_id,
            "slurm_stdout_path": slurm_stdout_path,
            "slurm_stderr_path": slurm_stderr_path,
            "model_params": model_params,
            "world_size": world_size,
            "batch_size_per_rank": batch_size,
            "global_batch_size": batch_size * world_size,
            "max_train_flops": max_train_flops,
            "train_loop_wall_seconds": train_loop_wall_seconds,
            "stop_reason": stop_reason,
            "steps_completed": step,
            "best_val_total": best_val_total,
            "best_val_mean_probe_score": None if not math.isfinite(best_val_mean_probe_score) else best_val_mean_probe_score,
            "best_val_mean_probe_score_step": best_val_mean_probe_score_step,
            "tile_presentations": examples_seen,
            "visible_patch_presentations": visible_patch_presentations,
            **final_unique_counts,
            "train_flops": train_flops,
            "flop_fraction": min(1.0, float(train_flops) / float(max_train_flops)),
            "warmup_flop_fraction": warmup_flop_fraction,
            "warmup_train_flops": warmup_train_flops,
            "ema_decay": ema_decay,
            "probe_model_weights": probe_model_weights,
            "probe_target_flops": probe_targets,
            "probe_target_fractions": [None if max_train_flops == 0 else target / max_train_flops for target in probe_targets],
            **best_probe_scores,
            **({} if probe_state is None else completed_probe_summary(output_dir)),
        }
        if probe_state is not None and "final_probe_score" not in summary:
            raise ValueError("probe.enabled is true but final_probe_score is missing; check probe.count, probe failures, and final checkpoint scheduling")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"{console_prefix()} Summary  "
            f"steps: {step}  train_wall: {train_loop_wall_seconds:.2f}s  "
            f"best_val_total: {best_val_total:.6f}  "
            f"final_probe_score: {summary.get('final_probe_score')}",
            flush=True,
        )
        for key, value in summary.items():
            wandb_run.summary[key] = value
        wandb_run.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
