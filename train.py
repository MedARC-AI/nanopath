# This file is the main pretraining loop for NanoPath.
# It keeps the recipe in one place: YAML load, DDP setup, dataloaders, forward/backward,
# logging, evaluation, checkpointing, one-hour exits, CBS batch doublings, and resume.
# The goal is a lean nanoGPT-style script rather than a framework stack.
# The training objective is single-stage LeJEPA plus latent masked prediction so we can
# scale one coherent recipe from 1xH100 micro runs to 8xH100 node runs without adding
# stage boundaries, hidden automation, or launcher-specific behavior.

import contextlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dataloader import RandomTCGADataset, prepare_sample_list_offsets
from model import SIGReg, NanoPathFM
from probe import collect_probe_results, prepare_probe_state, probe_enabled, queue_probe_job


GNS_EVERY = 100


def load_config():
    if len(sys.argv) != 2:
        raise ValueError("usage: python train.py <config.yaml>")
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    cfg["config_path"] = str(Path(sys.argv[1]).resolve())
    return cfg


def evaluate(model, loader, cfg, device, world_size):
    model.eval()
    train_cfg = cfg["train"]
    # These four validation logs are not interchangeable (lower is better for all of them):
    # val/pred is the JEPA view-consistency term, val/sig is the anti-collapse regularizer,
    # val/latent is the latent masked-prediction loss, and val/lejepa_proxy is only a
    # JEPA-side recipe-selection proxy. Lower is better for all four, but only the probe
    # accuracies are directly interpretable as downstream task performance.
    losses = {"pred": 0.0, "sig": 0.0, "latent": 0.0, "proxy": 0.0}
    batches = 0
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" and train_cfg["bf16"] else contextlib.nullcontext()
    eval_generator = torch.Generator(device=device.type)
    eval_generator.manual_seed(train_cfg["seed"])
    with torch.no_grad():
        for batch in loader:
            with autocast:
                global_views = batch["global_views"].to(device, non_blocking=True)
                local_views = batch["local_views"].to(device, non_blocking=True)
                latent_view = batch["latent_view"].to(device, non_blocking=True)
                proj = model.encode_views(global_views, local_views, False)
                proj_sigreg = torch.cat(dist_nn.all_gather(proj), dim=0) if world_size > 1 else proj
                full = model.latent_targets(latent_view, False).detach()
                pred, mask = model.latent_predictions(latent_view, train_cfg, generator=eval_generator)
                pred_loss = (proj - proj.mean(dim=1, keepdim=True)).square().mean()
                sig_loss = model.sigreg(proj_sigreg.transpose(0, 1), generator=eval_generator)
                latent_loss = F.l1_loss(pred, full) if train_cfg["latent_predict_visible"] else F.l1_loss(pred[mask], full[mask])
                proxy = (pred_loss + train_cfg["lambda_sig"] * sig_loss) / (train_cfg["lambda_sig"] ** 0.4)
            losses["pred"] += pred_loss.item()
            losses["sig"] += sig_loss.item()
            losses["latent"] += latent_loss.item()
            losses["proxy"] += proxy.item()
            batches += 1
            if batches >= train_cfg["val_batches"]:
                break
    if world_size > 1:
        tensor = torch.tensor([losses["pred"], losses["sig"], losses["latent"], losses["proxy"], batches], device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        losses["pred"], losses["sig"], losses["latent"], losses["proxy"], batches = tensor.tolist()
    return {k: v / batches for k, v in losses.items()}


def estimate_flops(model_params, visible_patch_presentations):
    return int(6 * model_params * visible_patch_presentations)


def positive_finite_mean(x):
    valid = torch.isfinite(x) & (x > 0)
    if not valid.any():
        return -1.0
    return float(x[valid].mean().item())


def main():
    cfg = load_config()
    train_cfg = cfg["train"]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(train_cfg["device"])
    random.seed(train_cfg["seed"] + rank)
    np.random.seed(train_cfg["seed"] + rank)
    torch.manual_seed(train_cfg["seed"] + rank)
    torch.set_float32_matmul_precision("high")
    model = NanoPathFM(cfg).to(device)
    model.sigreg = SIGReg().to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith("bias") or "norm" in name or "register_tokens" in name or "predictor_query" in name:
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
    batch_size = int(train_cfg["batch_size"])
    lr = train_cfg["lr_ref"] * math.sqrt((batch_size * world_size) / train_cfg["batch_ref"])
    examples_seen = 0
    visible_patch_presentations = 0
    masked_target_presentations = 0
    train_flops = 0
    best_val_proxy = float("inf")
    best_val_mean_probe_f1 = float("-inf")
    best_val_mean_probe_f1_step = -1
    best_probe_scores = {}
    output_dir = Path(cfg["project"]["output_dir"])
    wandb_dir = Path("/data/nanopath/wandb")
    # Fresh launches fully replace the run directory so repeated use of a checked-in
    # output_dir like /data/nanopath/nano never trips over stale artifacts.
    if train_cfg["resume"] is None:
        if rank == 0 and output_dir.exists():
            shutil.rmtree(output_dir)
        if distributed:
            dist.barrier()
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    resume = None
    wandb_meta = None
    if train_cfg["resume"] is not None:
        resume = torch.load(train_cfg["resume"], map_location=device, weights_only=False)
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        step = int(resume["step"])
        batch_size = int(resume["batch_size"])
        lr = float(resume["lr"])
        best_val_proxy = float(resume["best_val_proxy"])
        examples_seen = int(resume["examples_seen"])
        visible_patch_presentations = int(resume["visible_patch_presentations"])
        masked_target_presentations = int(resume["masked_target_presentations"])
        train_flops = int(resume["train_flops"])
        best_probe_scores = dict(resume["best_probe_scores"])
        best_val_mean_probe_f1 = float(resume["best_val_mean_probe_f1"])
        best_val_mean_probe_f1_step = int(resume["best_val_mean_probe_f1_step"])
        wandb_meta = dict(resume["wandb"])
    if rank == 0:
        wandb_init = {
            "project": "nanopath",
            "name": cfg["project"]["name"],
            "dir": str(wandb_dir),
            "config": cfg,
        }
        if wandb_meta is not None:
            wandb_init["id"] = wandb_meta["id"]
            wandb_init["resume"] = "must"
        wandb_run = wandb.init(**wandb_init)
        wandb_run.define_metric("val_thunder/step")
        wandb_run.define_metric("val_thunder/*", step_metric="val_thunder/step")
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
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    root_model = model.module if distributed else model
    global_patches = (train_cfg["global_size"] // cfg["model"]["patch_size"]) ** 2
    local_patches = (train_cfg["local_size"] // cfg["model"]["patch_size"]) ** 2
    latent_patches = (train_cfg["latent_size"] // cfg["model"]["patch_size"]) ** 2
    last_time = time.time()
    last_examples = examples_seen
    last_visible_patch_presentations = visible_patch_presentations
    last_train_flops = train_flops
    last_gns_simple = None
    last_gns_batch_ratio = None

    def checkpoint_payload(next_step):
        return {
            "model": root_model.state_dict(),
            "opt": opt.state_dict(),
            "step": next_step,
            "best_val_proxy": best_val_proxy,
            "best_val_mean_probe_f1": best_val_mean_probe_f1,
            "best_val_mean_probe_f1_step": best_val_mean_probe_f1_step,
            "examples_seen": examples_seen,
            "visible_patch_presentations": visible_patch_presentations,
            "masked_target_presentations": masked_target_presentations,
            "train_flops": train_flops,
            "batch_size": batch_size,
            "best_probe_scores": best_probe_scores,
            "lr": lr,
            "wandb": wandb_meta,
            "config": cfg,
        }

    def probe_checkpoint_payload(next_step):
        return {
            "model": root_model.state_dict(),
            "step": next_step,
            "config": cfg,
        }

    if rank == 0 and probe_state is not None:
        best_val_mean_probe_f1, best_val_mean_probe_f1_step, best_probe_scores = collect_probe_results(
            probe_state,
            wandb_run,
            metrics_path,
            output_dir,
            best_val_mean_probe_f1,
            best_val_mean_probe_f1_step,
            best_probe_scores,
            step,
        )
    if distributed:
        dist.barrier()
    max_wall_seconds = int(train_cfg["max_wall_seconds"])
    train_loop_started_at = time.monotonic()
    train_loop_wall_seconds = 0.0
    stop_reason = None
    cooldown_started_step = None
    cooldown_started_wall_seconds = None
    last_eval_step = step if math.isfinite(best_val_proxy) else -1
    last_saved_step = step if train_cfg["resume"] is not None else 0

    while True:
        if distributed:
            train_loader.sampler.set_epoch(step + train_cfg["seed"])
        for batch in train_loader:
            root_model.train()
            global_views = batch["global_views"].to(device, non_blocking=True)
            local_views = batch["local_views"].to(device, non_blocking=True)
            latent_view = batch["latent_view"].to(device, non_blocking=True)
            current_batch = latent_view.shape[0]
            sampled_mpp_mean = positive_finite_mean(batch["sampled_mpp"].float())
            warmup = min(1.0, float(step + 1) / float(train_cfg["warmup_steps"]))
            for group in opt.param_groups:
                group["lr"] = lr * warmup
            autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" and train_cfg["bf16"] else contextlib.nullcontext()
            sigreg_generator = torch.Generator(device=device.type)
            sigreg_generator.manual_seed(train_cfg["seed"] + step)
            gns_grad_sq_small = None
            gns_batch_small = None
            gns_batch_big = None
            if step == 0 or (step + 1) % GNS_EVERY == 0:
                if distributed:
                    opt.zero_grad(set_to_none=True)
                    with model.no_sync():
                        with autocast:
                            proj = root_model.encode_views(global_views, local_views, train_cfg["activation_checkpointing"])
                            proj_sigreg = torch.cat(dist_nn.all_gather(proj), dim=0)
                            full = root_model.latent_targets(latent_view, train_cfg["activation_checkpointing"]).detach()
                            pred, mask = root_model.latent_predictions(latent_view, train_cfg)
                            pred_loss = (proj - proj.mean(dim=1, keepdim=True)).square().mean()
                            sig_loss = root_model.sigreg(proj_sigreg.transpose(0, 1), generator=sigreg_generator)
                            latent_loss = F.l1_loss(pred, full) if train_cfg["latent_predict_visible"] else F.l1_loss(pred[mask], full[mask])
                            total_loss = pred_loss + train_cfg["lambda_sig"] * sig_loss + train_cfg["lambda_lat"] * latent_loss
                        total_loss.backward()
                    grad_sq_small = 0.0
                    for param in root_model.parameters():
                        if param.grad is not None:
                            grad_sq_small += param.grad.detach().float().square().sum().item()
                    grad_sq_small = torch.tensor([grad_sq_small], device=device)
                    dist.all_reduce(grad_sq_small, op=dist.ReduceOp.SUM)
                    gns_grad_sq_small = float((grad_sq_small / world_size).item())
                    gns_batch_small = current_batch
                    gns_batch_big = current_batch * world_size
                    opt.zero_grad(set_to_none=True)
                elif current_batch >= 2 and current_batch % 2 == 0:
                    half = current_batch // 2
                    half_grad_sqs = []
                    for start in [0, half]:
                        opt.zero_grad(set_to_none=True)
                        with autocast:
                            proj = root_model.encode_views(global_views[start : start + half], local_views[start : start + half], train_cfg["activation_checkpointing"])
                            full = root_model.latent_targets(latent_view[start : start + half], train_cfg["activation_checkpointing"]).detach()
                            pred, mask = root_model.latent_predictions(latent_view[start : start + half], train_cfg)
                            pred_loss = (proj - proj.mean(dim=1, keepdim=True)).square().mean()
                            sig_loss = root_model.sigreg(proj.transpose(0, 1), generator=sigreg_generator)
                            latent_loss = F.l1_loss(pred, full) if train_cfg["latent_predict_visible"] else F.l1_loss(pred[mask], full[mask])
                            total_loss = pred_loss + train_cfg["lambda_sig"] * sig_loss + train_cfg["lambda_lat"] * latent_loss
                        total_loss.backward()
                        grad_sq_small = 0.0
                        for param in root_model.parameters():
                            if param.grad is not None:
                                grad_sq_small += param.grad.detach().float().square().sum().item()
                        half_grad_sqs.append(grad_sq_small)
                    gns_grad_sq_small = float(sum(half_grad_sqs) / len(half_grad_sqs))
                    gns_batch_small = half
                    gns_batch_big = current_batch
                    opt.zero_grad(set_to_none=True)
            with autocast:
                proj = root_model.encode_views(global_views, local_views, train_cfg["activation_checkpointing"])
                proj_sigreg = torch.cat(dist_nn.all_gather(proj), dim=0) if distributed else proj
                full = root_model.latent_targets(latent_view, train_cfg["activation_checkpointing"]).detach()
                pred, mask = root_model.latent_predictions(latent_view, train_cfg)
                pred_loss = (proj - proj.mean(dim=1, keepdim=True)).square().mean()
                sig_loss = root_model.sigreg(proj_sigreg.transpose(0, 1), generator=sigreg_generator)
                latent_loss = F.l1_loss(pred, full) if train_cfg["latent_predict_visible"] else F.l1_loss(pred[mask], full[mask])
                total_loss = pred_loss + train_cfg["lambda_sig"] * sig_loss + train_cfg["lambda_lat"] * latent_loss
                proxy = (pred_loss + train_cfg["lambda_sig"] * sig_loss) / (train_cfg["lambda_sig"] ** 0.4)
                proj_std = proj.float().reshape(-1, proj.shape[-1]).std(dim=0).mean()
                target_std = full.float().reshape(-1, full.shape[-1]).std(dim=0).mean()
                if train_cfg["latent_predict_visible"]:
                    pred_flat = pred.float().reshape(-1, pred.shape[-1])
                    full_flat = full.float().reshape(-1, full.shape[-1])
                else:
                    pred_flat = pred[mask].float()
                    full_flat = full[mask].float()
                pred_target_cos = F.cosine_similarity(pred_flat, full_flat, dim=-1).mean()
                mask_fraction = mask.float().mean()
            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_sq = 0.0
            param_sq = 0.0
            for param in root_model.parameters():
                param_sq += param.detach().float().square().sum().item()
                if param.grad is not None:
                    grad_sq += param.grad.detach().float().square().sum().item()
            grad_norm = grad_sq ** 0.5
            param_norm = param_sq ** 0.5
            grad_param_ratio = grad_norm / max(param_norm, 1e-12)
            grad_clip_scale = min(1.0, train_cfg["grad_clip"] / max(grad_norm, 1e-12)) if train_cfg["grad_clip"] > 0 else 1.0
            gns_simple = None
            gns_noise = None
            gns_signal = None
            gns_batch_ratio = None
            if gns_grad_sq_small is not None:
                gns_noise = max(0.0, (gns_batch_small * gns_batch_big * (gns_grad_sq_small - grad_sq)) / (gns_batch_big - gns_batch_small))
                gns_signal = max(0.0, (gns_batch_big * grad_sq - gns_batch_small * gns_grad_sq_small) / (gns_batch_big - gns_batch_small))
                if gns_signal > 0.0:
                    gns_simple = gns_noise / gns_signal
                    last_gns_simple = gns_simple
                    if gns_simple > 0.0:
                        gns_batch_ratio = gns_batch_big / gns_simple
                        last_gns_batch_ratio = gns_batch_ratio
            if train_cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(root_model.parameters(), train_cfg["grad_clip"])
            opt.step()
            global_batch = current_batch * world_size
            visible_now = global_batch * (
                train_cfg["global_views"] * global_patches
                + train_cfg["local_views"] * local_patches
                + latent_patches
                + int(round(latent_patches * (1.0 - train_cfg["latent_mask_ratio"])))
            )
            masked_now = global_batch * (latent_patches - int(round(latent_patches * (1.0 - train_cfg["latent_mask_ratio"]))))
            examples_seen += global_batch
            visible_patch_presentations += visible_now
            masked_target_presentations += masked_now
            train_flops += estimate_flops(model_params, visible_now)
            if distributed:
                reduced = torch.tensor(
                    [
                        pred_loss.item(),
                        sig_loss.item(),
                        latent_loss.item(),
                        proxy.item(),
                        total_loss.item(),
                        proj_std.item(),
                        target_std.item(),
                        pred_target_cos.item(),
                        mask_fraction.item(),
                    ],
                    device=device,
                )
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                reduced = (reduced / world_size).tolist()
            else:
                reduced = [pred_loss.item(), sig_loss.item(), latent_loss.item(), proxy.item(), total_loss.item(), proj_std.item(), target_std.item(), pred_target_cos.item(), mask_fraction.item()]
            if rank == 0 and step % train_cfg["log_every"] == 0:
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
                gpu_mem_gb = torch.cuda.memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
                gpu_peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
                train_log = {
                    "step": step,
                    "train_pred": reduced[0],
                    "train_sig": reduced[1],
                    "train_latent": reduced[2],
                    "train_lejepa_proxy": reduced[3],
                    "train_total": reduced[4],
                    "train_proj_std": reduced[5],
                    "train_target_std": reduced[6],
                    "train_pred_target_cos": reduced[7],
                    "train_mask_fraction": reduced[8],
                    "examples_seen": examples_seen,
                    "visible_patch_presentations": visible_patch_presentations,
                    "masked_target_presentations": masked_target_presentations,
                    "train_flops": train_flops,
                    "sampled_mpp_mean": sampled_mpp_mean,
                    "items_per_sec": items_per_sec,
                    "visible_patches_per_sec": visible_patches_per_sec,
                    "flops_per_sec": flops_per_sec,
                    "train_loop_wall_seconds": train_loop_wall_seconds,
                    "train_loop_wall_fraction": min(1.0, train_loop_wall_seconds / max_wall_seconds),
                    "lr": opt.param_groups[0]["lr"],
                    "batch_size": batch_size,
                    "global_batch_size": global_batch,
                    "grad_norm": grad_norm,
                    "param_norm": param_norm,
                    "grad_param_ratio": grad_param_ratio,
                    "grad_clip_scale": grad_clip_scale,
                    "gpu_mem_gb": gpu_mem_gb,
                    "gpu_peak_mem_gb": gpu_peak_mem_gb,
                }
                if gns_simple is not None:
                    train_log["gns_simple"] = gns_simple
                    train_log["gns_noise"] = gns_noise
                    train_log["gns_signal"] = gns_signal
                    train_log["gns_batch_small"] = gns_batch_small
                    train_log["gns_batch_big"] = gns_batch_big
                    if gns_batch_ratio is not None:
                        train_log["gns_batch_ratio"] = gns_batch_ratio
                with metrics_path.open("a") as f:
                    f.write(json.dumps(train_log) + "\n")
                wandb_log = {
                    "train/pred": reduced[0],
                    "train/sig": reduced[1],
                    "train/latent": reduced[2],
                    "train/lejepa_proxy": reduced[3],
                    "train/total": reduced[4],
                    "train/proj_std": reduced[5],
                    "train/target_std": reduced[6],
                    "train/pred_target_cos": reduced[7],
                    "train/mask_fraction": reduced[8],
                    "train/examples_seen": examples_seen,
                    "train/visible_patch_presentations": visible_patch_presentations,
                    "train/masked_target_presentations": masked_target_presentations,
                    "train/flops": train_flops,
                    "train/sampled_mpp_mean": sampled_mpp_mean,
                    "train/items_per_sec": items_per_sec,
                    "train/visible_patches_per_sec": visible_patches_per_sec,
                    "train/flops_per_sec": flops_per_sec,
                    "train/wall_seconds": train_loop_wall_seconds,
                    "train/wall_fraction": min(1.0, train_loop_wall_seconds / max_wall_seconds),
                    "train/lr": opt.param_groups[0]["lr"],
                    "train/batch_size": batch_size,
                    "train/global_batch_size": global_batch,
                    "train/grad_norm": grad_norm,
                    "train/param_norm": param_norm,
                    "train/grad_param_ratio": grad_param_ratio,
                    "train/grad_clip_scale": grad_clip_scale,
                    "train/gpu_mem_gb": gpu_mem_gb,
                    "train/gpu_peak_mem_gb": gpu_peak_mem_gb,
                }
                if gns_simple is not None:
                    wandb_log["train/gns_simple"] = gns_simple
                    wandb_log["train/gns_noise"] = gns_noise
                    wandb_log["train/gns_signal"] = gns_signal
                    wandb_log["train/gns_batch_small"] = gns_batch_small
                    wandb_log["train/gns_batch_big"] = gns_batch_big
                    if gns_batch_ratio is not None:
                        wandb_log["train/gns_batch_ratio"] = gns_batch_ratio
                wandb_run.log(wandb_log, step=step)
                if probe_state is not None:
                    best_val_mean_probe_f1, best_val_mean_probe_f1_step, best_probe_scores = collect_probe_results(
                        probe_state,
                        wandb_run,
                        metrics_path,
                        output_dir,
                        best_val_mean_probe_f1,
                        best_val_mean_probe_f1_step,
                        best_probe_scores,
                        step,
                    )
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
            if (step + 1) % train_cfg["eval_every"] == 0:
                val = evaluate(root_model, val_loader, cfg, device, world_size)
                run_probe = cfg["probe"]["enabled"] and (step + 1) % cfg["probe"]["every"] == 0
                if val["proxy"] < best_val_proxy:
                    best_val_proxy = val["proxy"]
                last_eval_step = step + 1
                if rank == 0:
                    with metrics_path.open("a") as f:
                        f.write(json.dumps({"step": step, "val_pred": val["pred"], "val_sig": val["sig"], "val_latent": val["latent"], "val_lejepa_proxy": val["proxy"]}) + "\n")
                    wandb_run.log({"val/pred": val["pred"], "val/sig": val["sig"], "val/latent": val["latent"], "val/lejepa_proxy": val["proxy"]}, step=step)
                    if run_probe and probe_state is not None:
                        queue_probe_job(cfg, probe_state, probe_checkpoint_payload(step + 1), step + 1)
                    if probe_state is not None:
                        best_val_mean_probe_f1, best_val_mean_probe_f1_step, best_probe_scores = collect_probe_results(
                            probe_state,
                            wandb_run,
                            metrics_path,
                            output_dir,
                            best_val_mean_probe_f1,
                            best_val_mean_probe_f1_step,
                            best_probe_scores,
                            step,
                        )
            if rank == 0 and (step + 1) % train_cfg["save_every"] == 0:
                torch.save(checkpoint_payload(step + 1), output_dir / f"step_{step + 1:07d}.pt")
                last_saved_step = step + 1
            cbs_trigger = step + 1 in train_cfg["cbs_doubling_steps"]
            if cbs_trigger:
                batch_size *= 2
                lr *= 2 ** 0.5
                train_loader = make_loader(train_ds, True, batch_size)
                val_loader = make_loader(val_ds, False, batch_size)
            step += 1
            train_loop_wall_seconds = time.monotonic() - train_loop_started_at
            if train_loop_wall_seconds >= max_wall_seconds:
                stop_reason = "max_wall_seconds"
                cooldown_started_step = step
                cooldown_started_wall_seconds = train_loop_wall_seconds
                break
            if cbs_trigger:
                break
        if stop_reason is not None:
            break
    train_loop_wall_seconds = time.monotonic() - train_loop_started_at
    if distributed:
        dist.barrier()
    if rank == 0 and stop_reason is not None:
        with metrics_path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "event": "cooldown_start",
                        "step": cooldown_started_step,
                        "stop_reason": stop_reason,
                        "train_loop_wall_seconds": cooldown_started_wall_seconds,
                    }
                )
                + "\n"
            )
        wandb_run.log(
            {
                "train/cooldown_start_wall_seconds": cooldown_started_wall_seconds,
                "train/cooldown_start_fraction": min(1.0, cooldown_started_wall_seconds / max_wall_seconds),
            },
            step=cooldown_started_step,
        )
    if step > 0 and last_eval_step != step:
        val = evaluate(root_model, val_loader, cfg, device, world_size)
        if val["proxy"] < best_val_proxy:
            best_val_proxy = val["proxy"]
        last_eval_step = step
        if rank == 0:
            with metrics_path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "event": "cooldown_eval",
                            "step": step,
                            "val_pred": val["pred"],
                            "val_sig": val["sig"],
                            "val_latent": val["latent"],
                            "val_lejepa_proxy": val["proxy"],
                        }
                    )
                    + "\n"
                )
            wandb_run.log(
                {
                    "val/pred": val["pred"],
                    "val/sig": val["sig"],
                    "val/latent": val["latent"],
                    "val/lejepa_proxy": val["proxy"],
                },
                step=step,
            )
    if not math.isfinite(best_val_proxy):
        raise ValueError("run finished without a finite best_val_proxy; check max_wall_seconds, eval_every, validation data, and loss stability")
    if rank == 0:
        if probe_state is not None:
            best_val_mean_probe_f1, best_val_mean_probe_f1_step, best_probe_scores = collect_probe_results(
                probe_state,
                wandb_run,
                metrics_path,
                output_dir,
                best_val_mean_probe_f1,
                best_val_mean_probe_f1_step,
                best_probe_scores,
                step,
            )
        if step > 0 and step != last_saved_step:
            torch.save(checkpoint_payload(step), output_dir / f"step_{step:07d}.pt")
            last_saved_step = step
        summary_path.write_text(
            json.dumps(
                {
                    "project": cfg["project"]["name"],
                    "family": cfg["project"]["family"],
                    "recipe_id": cfg["project"]["recipe_id"],
                    "config_path": cfg["config_path"],
                    "model_params": model_params,
                    "world_size": world_size,
                    "batch_size_per_rank": batch_size,
                    "global_batch_size": batch_size * world_size,
                    "max_wall_seconds": max_wall_seconds,
                    "train_loop_wall_seconds": train_loop_wall_seconds,
                    "stop_reason": stop_reason,
                    "cooldown_started_step": cooldown_started_step,
                    "cooldown_started_wall_seconds": cooldown_started_wall_seconds,
                    "steps_completed": step,
                    "best_val_lejepa_proxy": best_val_proxy,
                    "best_val_mean_probe_f1": None if not math.isfinite(best_val_mean_probe_f1) else best_val_mean_probe_f1,
                    "best_val_mean_probe_f1_step": best_val_mean_probe_f1_step,
                    "tile_presentations": examples_seen,
                    "visible_patch_presentations": visible_patch_presentations,
                    "masked_target_presentations": masked_target_presentations,
                    "train_flops": train_flops,
                    "last_gns_simple": last_gns_simple,
                    "last_gns_batch_ratio": last_gns_batch_ratio,
                    "thunder_probe_active_job_id": None if probe_state is None or probe_state["data"]["active"] is None else probe_state["data"]["active"]["job_id"],
                    "thunder_probe_active_step": None if probe_state is None or probe_state["data"]["active"] is None else probe_state["data"]["active"]["train_step"],
                    "thunder_probe_queued_step": None if probe_state is None or probe_state["data"]["queued"] is None else probe_state["data"]["queued"]["train_step"],
                    **best_probe_scores,
                },
                indent=2,
            )
        )
        wandb_run.summary["max_wall_seconds"] = max_wall_seconds
        wandb_run.summary["train_loop_wall_seconds"] = train_loop_wall_seconds
        wandb_run.summary["stop_reason"] = stop_reason
        wandb_run.summary["cooldown_started_step"] = cooldown_started_step
        wandb_run.summary["cooldown_started_wall_seconds"] = cooldown_started_wall_seconds
        wandb_run.summary["best_val_lejepa_proxy"] = best_val_proxy
        if math.isfinite(best_val_mean_probe_f1):
            wandb_run.summary["best_val_mean_probe_f1"] = best_val_mean_probe_f1
            wandb_run.summary["best_val_mean_probe_f1_step"] = best_val_mean_probe_f1_step
        if last_gns_simple is not None:
            wandb_run.summary["last_gns_simple"] = last_gns_simple
            wandb_run.summary["last_gns_batch_ratio"] = last_gns_batch_ratio
        if probe_state is not None and probe_state["data"]["active"] is not None:
            wandb_run.summary["thunder_probe_active_job_id"] = probe_state["data"]["active"]["job_id"]
            wandb_run.summary["thunder_probe_active_step"] = probe_state["data"]["active"]["train_step"]
        if probe_state is not None and probe_state["data"]["queued"] is not None:
            wandb_run.summary["thunder_probe_queued_step"] = probe_state["data"]["queued"]["train_step"]
        for key, value in best_probe_scores.items():
            wandb_run.summary[key] = value
        wandb_run.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
