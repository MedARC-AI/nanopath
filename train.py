# NanoPath pretraining loop: YAML config, DDP, logging, checkpointing, and async probes.

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
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dataloader import RandomTCGADataset, SAMPLE_LIST_PATCH_SIZE, prepare_sample_list_offsets
from model import SIGReg, NanoPathFM
from probe import collect_probe_results, prepare_probe_state, probe_enabled, queue_probe_job


GNS_EVERY = 100
EVAL_KEYS = ("jepa_pred", "sig", "total", "jepa_proxy")
TRAIN_KEYS = ("jepa_pred", "sig", "total", "jepa_proxy", "proj_std")


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
    # total_loss is the objective we backprop through, and jepa_proxy tracks that same
    # JEPA branch on the historical normalized scale used for model selection.
    proj_sigreg = proj if proj_sigreg is None else proj_sigreg
    jepa_pred_loss = (proj - proj.mean(dim=1, keepdim=True)).square().mean()
    sig_loss = sigreg(proj_sigreg.transpose(0, 1), generator=sigreg_generator)
    total_loss = train_cfg["lambda_jepa_pred"] * jepa_pred_loss + train_cfg["lambda_sig"] * sig_loss
    jepa_proxy = total_loss / (train_cfg["lambda_sig"] ** 0.4) if train_cfg["lambda_sig"] > 0 else total_loss
    return jepa_pred_loss, sig_loss, total_loss, jepa_proxy


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
                jepa_pred_loss, sig_loss, total_loss, jepa_proxy = loss_terms(
                    sigreg,
                    proj,
                    train_cfg,
                    eval_generator,
                    torch.cat(dist_nn.all_gather(proj), dim=0) if world_size > 1 else proj,
                )
            for i, value in enumerate((jepa_pred_loss, sig_loss, total_loss, jepa_proxy)):
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
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
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
    batch_size = int(train_cfg["batch_size"])
    lr = train_cfg["lr_ref"] * math.sqrt((batch_size * world_size) / train_cfg["batch_ref"])
    examples_seen = 0
    visible_patch_presentations = 0
    train_flops = 0
    best_val_jepa_proxy = float("inf")
    best_val_mean_probe_f1 = float("-inf")
    best_val_mean_probe_f1_step = -1
    best_probe_scores = {}
    output_dir = Path(cfg["project"]["output_dir"])
    wandb_dir = Path("/data/nanopath/wandb")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_stdout_path = os.environ.get("NANOPATH_SLURM_STDOUT")
    slurm_stderr_path = os.environ.get("NANOPATH_SLURM_STDERR")
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
    wandb_meta = None
    resume_path = train_cfg["resume"]
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["opt"])
        (
            step,
            batch_size,
            lr,
            best_val_jepa_proxy,
            examples_seen,
            visible_patch_presentations,
            train_flops,
            best_val_mean_probe_f1,
            best_val_mean_probe_f1_step,
        ) = (
            int(checkpoint["step"]),
            int(checkpoint["batch_size"]),
            float(checkpoint["lr"]),
            float(checkpoint["best_val_jepa_proxy"]),
            int(checkpoint["examples_seen"]),
            int(checkpoint["visible_patch_presentations"]),
            int(checkpoint["train_flops"]),
            float(checkpoint["best_val_mean_probe_f1"]),
            int(checkpoint["best_val_mean_probe_f1_step"]),
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
                console="redirect",
                console_multipart=True,
                console_chunk_max_seconds=15,
            ),
        }
        if wandb_meta is not None:
            wandb_init["id"] = wandb_meta["id"]
            wandb_init["resume"] = "must"
        wandb_run = wandb.init(**wandb_init)
        wandb_run.define_metric("probe/step", hidden=True)
        wandb_run.define_metric("probe/*", step_metric="probe/step")
        source_artifact = wandb.Artifact(f"nanopath-source-{wandb_run.id}", type="code")
        repo_dir = Path(__file__).resolve().parent
        for path in sorted({*repo_dir.rglob("*"), *repo_dir.glob(".*")}):
            if not path.is_file() or any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
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
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    root_model = model.module if distributed else model
    global_patches = (train_cfg["global_size"] // cfg["model"]["patch_size"]) ** 2
    local_patches = (train_cfg["local_size"] // cfg["model"]["patch_size"]) ** 2
    last_time = time.time()
    last_examples = examples_seen
    last_visible_patch_presentations = visible_patch_presentations
    last_train_flops = train_flops
    last_gns_simple = None
    last_gns_batch_ratio = None
    unique_tile_patch_count = (SAMPLE_LIST_PATCH_SIZE // cfg["model"]["patch_size"]) ** 2
    seen_sample_ids = set()
    seen_slide_ids = set()
    seen_patient_ids = set()
    pending_sample_ids = set()
    pending_slide_ids = set()
    pending_patient_ids = set()

    def checkpoint_payload(next_step):
        return {
            "model": root_model.state_dict(),
            "opt": opt.state_dict(),
            "step": next_step,
            "best_val_jepa_proxy": best_val_jepa_proxy,
            "best_val_mean_probe_f1": best_val_mean_probe_f1,
            "best_val_mean_probe_f1_step": best_val_mean_probe_f1_step,
            "examples_seen": examples_seen,
            "visible_patch_presentations": visible_patch_presentations,
            "train_flops": train_flops,
            "batch_size": batch_size,
            "best_probe_scores": best_probe_scores,
            "lr": lr,
            "wandb": wandb_meta,
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

    def log_probe_results(log_step):
        nonlocal best_val_mean_probe_f1, best_val_mean_probe_f1_step, best_probe_scores
        if probe_state is None:
            return
        best_val_mean_probe_f1, best_val_mean_probe_f1_step, best_probe_scores = collect_probe_results(
            probe_state, wandb_run, metrics_path, output_dir, best_val_mean_probe_f1, best_val_mean_probe_f1_step, best_probe_scores, log_step
        )

    def log_val(step_value, val, event=None):
        payload = {"step": step_value, **{f"val_{key}": val[key] for key in EVAL_KEYS}}
        if event is not None:
            payload["event"] = event
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")
        wandb_run.log({f"val/{key}": val[key] for key in EVAL_KEYS}, step=step_value)

    if rank == 0 and probe_state is not None:
        log_probe_results(step)
    if distributed:
        dist.barrier()
    max_train_flops = int(train_cfg["max_train_flops"])
    warmup_flop_fraction = float(train_cfg["warmup_flop_fraction"])
    if warmup_flop_fraction <= 0.0 or warmup_flop_fraction > 1.0:
        raise ValueError(f"warmup_flop_fraction must be in (0, 1], got {warmup_flop_fraction}")
    warmup_train_flops = math.ceil(max_train_flops * warmup_flop_fraction)
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
            next_probe_idx = min(
                len(probe_targets),
                len(probe_state["data"]["logged_results"])
                + int(probe_state["data"]["active"] is not None)
                + int(probe_state["data"]["queued"] is not None),
            )
    train_loop_started_at = time.monotonic()
    train_loop_wall_seconds = 0.0
    stop_reason = "max_train_flops" if train_flops >= max_train_flops else None
    last_eval_step = step if math.isfinite(best_val_jepa_proxy) else -1
    last_saved_step = step if resume_path is not None else 0

    while stop_reason is None:
        if distributed:
            train_loader.sampler.set_epoch(step + train_cfg["seed"])
        for batch in train_loader:
            model.train()
            pending_sample_ids.update(int(x) for x in batch["sample_idx"].tolist())
            pending_slide_ids.update(int(x) for x in batch["slide_id"].tolist())
            pending_patient_ids.update(int(x) for x in batch["patient_id"].tolist())
            global_views, local_views = [batch[key].to(device, non_blocking=True) for key in ("global_views", "local_views")]
            current_batch = global_views.shape[0]
            global_batch = current_batch * world_size
            visible_now = global_batch * (train_cfg["global_views"] * global_patches + train_cfg["local_views"] * local_patches)
            step_train_flops = int(6 * model_params * visible_now)
            finite_mpp = batch["sampled_mpp"].float()
            finite_mpp = finite_mpp[torch.isfinite(finite_mpp)]
            sampled_mpp_mean = float(finite_mpp.mean().item()) if finite_mpp.numel() > 0 else float("nan")
            warmup = min(1.0, float(train_flops + step_train_flops) / float(warmup_train_flops))
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
                            proj = model(global_views, local_views, train_cfg)
                            jepa_pred_loss, sig_loss, total_loss, _ = loss_terms(
                                root_model.sigreg,
                                proj,
                                train_cfg,
                                sigreg_generator,
                                torch.cat(dist_nn.all_gather(proj), dim=0),
                            )
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
                            proj = model(
                                global_views[start : start + half],
                                local_views[start : start + half],
                                train_cfg,
                            )
                            jepa_pred_loss, sig_loss, total_loss, _ = loss_terms(
                                root_model.sigreg,
                                proj,
                                train_cfg,
                                sigreg_generator,
                            )
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
                proj = model(global_views, local_views, train_cfg)
                jepa_pred_loss, sig_loss, total_loss, jepa_proxy = loss_terms(
                    root_model.sigreg,
                    proj,
                    train_cfg,
                    sigreg_generator,
                    torch.cat(dist_nn.all_gather(proj), dim=0) if distributed else proj,
                )
                proj_std = proj.float().reshape(-1, proj.shape[-1]).std(dim=0).mean()
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
            gns_batch_ratio = None
            if gns_grad_sq_small is not None:
                noise = max(0.0, (gns_batch_small * gns_batch_big * (gns_grad_sq_small - grad_sq)) / (gns_batch_big - gns_batch_small))
                signal = max(0.0, (gns_batch_big * grad_sq - gns_batch_small * gns_grad_sq_small) / (gns_batch_big - gns_batch_small))
                if signal > 0.0:
                    gns_simple = noise / signal
                    last_gns_simple = gns_simple
                    if gns_simple > 0.0:
                        gns_batch_ratio = gns_batch_big / gns_simple
                        last_gns_batch_ratio = gns_batch_ratio
            if train_cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(root_model.parameters(), train_cfg["grad_clip"])
            opt.step()
            completed_step = step + 1
            examples_seen += global_batch
            visible_patch_presentations += visible_now
            train_flops += step_train_flops
            if distributed:
                reduced = torch.tensor(
                    [
                        jepa_pred_loss.item(),
                        sig_loss.item(),
                        total_loss.item(),
                        jepa_proxy.item(),
                        proj_std.item(),
                    ],
                    device=device,
                )
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                reduced = (reduced / world_size).tolist()
            else:
                reduced = [jepa_pred_loss.item(), sig_loss.item(), total_loss.item(), jepa_proxy.item(), proj_std.item()]
            reduced = dict(zip(TRAIN_KEYS, reduced))
            should_log = completed_step == 1 or completed_step % train_cfg["log_every"] == 0
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
                gpu_mem_gb = torch.cuda.memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
                gpu_peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
                train_log = {
                    "step": completed_step,
                    **reduced,
                    "sampled_mpp_mean": sampled_mpp_mean,
                    "items_per_sec": items_per_sec,
                    "visible_patches_per_sec": visible_patches_per_sec,
                    "flops_per_sec": flops_per_sec,
                    "wall_seconds": train_loop_wall_seconds,
                    "flop_fraction": min(1.0, float(train_flops) / float(max_train_flops)),
                    "lr": opt.param_groups[0]["lr"],
                    "batch_size": batch_size,
                    "global_batch_size": global_batch,
                    "examples_seen": examples_seen,
                    "visible_patch_presentations": visible_patch_presentations,
                    "train_flops": train_flops,
                    "grad_norm": grad_norm,
                    "param_norm": param_norm,
                    "grad_param_ratio": grad_param_ratio,
                    "grad_clip_scale": grad_clip_scale,
                    "gpu_mem_gb": gpu_mem_gb,
                    "gpu_peak_mem_gb": gpu_peak_mem_gb,
                }
                train_log.update(unique_counts)
                if gns_simple is not None:
                    train_log["gns_simple"] = gns_simple
                if gns_batch_ratio is not None:
                    train_log["gns_batch_ratio"] = gns_batch_ratio
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
                    log_probe_results(completed_step)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
            if completed_step % train_cfg["eval_every"] == 0:
                val = evaluate(model, root_model.sigreg, val_loader, cfg, device, world_size)
                run_probe = probe_state is not None and next_probe_idx < len(probe_targets) and train_flops >= probe_targets[next_probe_idx]
                if val["jepa_proxy"] < best_val_jepa_proxy:
                    best_val_jepa_proxy = val["jepa_proxy"]
                last_eval_step = completed_step
                if rank == 0:
                    log_val(completed_step, val)
                    if run_probe and probe_state is not None:
                        queue_probe_job(
                            cfg,
                            probe_state,
                            {"model": root_model.state_dict(), "step": completed_step, "config": cfg},
                            completed_step,
                            next_probe_idx + 1,
                            probe_targets[next_probe_idx],
                            probe_targets[next_probe_idx] / max_train_flops,
                        )
                        next_probe_idx += 1
                    if probe_state is not None:
                        log_probe_results(completed_step)
            if rank == 0 and completed_step % train_cfg["save_every"] == 0:
                torch.save(checkpoint_payload(completed_step), output_dir / f"step_{completed_step:07d}.pt")
                last_saved_step = completed_step
            cbs_trigger = completed_step in train_cfg["cbs_doubling_steps"]
            if cbs_trigger:
                batch_size *= 2
                lr *= 2 ** 0.5
                train_loader = make_loader(train_ds, True, batch_size)
                val_loader = make_loader(val_ds, False, batch_size)
            step = completed_step
            train_loop_wall_seconds = time.monotonic() - train_loop_started_at
            if train_flops >= max_train_flops:
                stop_reason = "max_train_flops"
                break
            if cbs_trigger:
                break
            if stop_reason is not None:
                break
    train_loop_wall_seconds = time.monotonic() - train_loop_started_at
    final_unique_counts = flush_unique_counts()
    if distributed:
        dist.barrier()
    if step > 0 and last_eval_step != step:
        val = evaluate(model, root_model.sigreg, val_loader, cfg, device, world_size)
        if val["jepa_proxy"] < best_val_jepa_proxy:
            best_val_jepa_proxy = val["jepa_proxy"]
        last_eval_step = step
        if rank == 0:
            log_val(step, val, "final_eval")
            if probe_state is not None and next_probe_idx < len(probe_targets) and train_flops >= probe_targets[next_probe_idx]:
                queue_probe_job(
                    cfg,
                    probe_state,
                    {"model": root_model.state_dict(), "step": step, "config": cfg},
                    step,
                    next_probe_idx + 1,
                    probe_targets[next_probe_idx],
                    probe_targets[next_probe_idx] / max_train_flops,
                )
                next_probe_idx += 1
            if probe_state is not None:
                log_probe_results(step)
    if not math.isfinite(best_val_jepa_proxy):
        raise ValueError("run finished without a finite best_val_jepa_proxy; check max_train_flops, eval_every, validation data, and loss stability")
    if rank == 0:
        if probe_state is not None:
            log_probe_results(step)
        if step > 0 and step != last_saved_step:
            torch.save(checkpoint_payload(step), output_dir / f"step_{step:07d}.pt")
            last_saved_step = step
        active_probe = None if probe_state is None else probe_state["data"]["active"]
        queued_probe = None if probe_state is None else probe_state["data"]["queued"]
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
            "best_val_jepa_proxy": best_val_jepa_proxy,
            "best_val_mean_probe_f1": None if not math.isfinite(best_val_mean_probe_f1) else best_val_mean_probe_f1,
            "best_val_mean_probe_f1_step": best_val_mean_probe_f1_step,
            "tile_presentations": examples_seen,
            "visible_patch_presentations": visible_patch_presentations,
            **final_unique_counts,
            "train_flops": train_flops,
            "flop_fraction": min(1.0, float(train_flops) / float(max_train_flops)),
            "warmup_flop_fraction": warmup_flop_fraction,
            "warmup_train_flops": warmup_train_flops,
            "last_gns_simple": last_gns_simple,
            "last_gns_batch_ratio": last_gns_batch_ratio,
            "probe_target_flops": probe_targets,
            "probe_target_fractions": [None if max_train_flops == 0 else target / max_train_flops for target in probe_targets],
            "thunder_probe_active_job_id": None if active_probe is None else active_probe["job_id"],
            "thunder_probe_active_step": None if active_probe is None else active_probe["train_step"],
            "thunder_probe_queued_step": None if queued_probe is None else queued_probe["train_step"],
            **best_probe_scores,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        for key, value in summary.items():
            wandb_run.summary[key] = value
        wandb_run.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
