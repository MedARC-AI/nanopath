# Pretraining entry point. Runs JEPA + SIGReg loop end-to-end: parse a YAML
# config (e.g. configs/leader.yaml), build the TCGA dataloader, construct the
# NanoPathFM model and EMA copy, optimize across DDP ranks under torchrun, log
# to wandb + output_dir/metrics.jsonl, optionally write rolling latest.pt
# checkpoints, and queue downstream probes (probe.py).
# Researchers changing objectives should start at loss_terms() here and SIGReg
# in model.py; changing data preprocessing starts in dataloader.py; changing
# downstream comparisons starts in probe.py.
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


EVAL_KEYS = ("jepa_pred", "sig", "total")
TRAIN_KEYS = ("jepa_pred", "sig", "total", "proj_std")


# Prefix every console line with wall time and job/process id so SLURM logs are easy to scan.
def console_prefix(): return f"{time.strftime('%H:%M:%S')} {os.environ.get('SLURM_JOB_ID', str(os.getpid()))}"


# Read the YAML recipe and fail before any GPU work if the TCGA sample list is absent.
def load_config():
    if len(sys.argv) != 2:
        raise ValueError("usage: python train.py <config.yaml>")
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    cfg["config_path"] = str(Path(sys.argv[1]).resolve())
    sample_list = Path(cfg["data"]["sample_list"])
    if not sample_list.exists():
        raise FileNotFoundError(
            f"Pretraining sample list not found at {sample_list}. This is TCGA WSI metadata "
            f"and slide data that are not downloaded by train.py. Follow the TCGA data setup "
            f"in README.md, for example `bash download_TCGA.sh /data/TCGA 8`, then set "
            f"data.sample_list to /data/TCGA/sample_dataset_30.txt or place the file at "
            f"the configured path in {cfg['config_path']}."
        )
    return cfg


# Training objective: JEPA-style view consistency plus SIGReg anti-collapse regularization.
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


# Run a short validation pass with the same objective so optimization curves stay comparable.
def evaluate(model, sigreg, loader, cfg, device, world_size):
    model.eval()
    train_cfg = cfg["train"]
    losses = [0.0] * len(EVAL_KEYS)
    batches = 0
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" and train_cfg["bf16"] else contextlib.nullcontext()
    eval_generator = torch.Generator(device=device.type)
    eval_generator.manual_seed(train_cfg["seed"])
    # Validation uses deterministic SIGReg directions so validation loss noise is not seed drift.
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
    # Every rank validates its shard, then rank-reduced sums recover the global mean loss.
    if world_size > 1:
        tensor = torch.tensor([*losses, batches], device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        losses = tensor[:-1].tolist()
        batches = int(tensor[-1].item())
    return {key: value / batches for key, value in zip(EVAL_KEYS, losses)}


# Orchestrates one pretraining run: setup, train/eval/probe loop, checkpoint, summary.
def main():
    cfg = load_config()
    train_cfg = cfg["train"]
    ema_decay = float(train_cfg["ema_decay"])
    probe_model_weights = str(cfg["probe"]["model_weights"])
    save_every = train_cfg["save_every"]
    save_checkpoints = save_every is not None
    # torchrun sets WORLD_SIZE/RANK/LOCAL_RANK; absence of those variables means local single-process.
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
    # Rank-offset seeds keep DDP workers from sampling identical augmentation streams.
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
    # NanoPathFM is the encoder/projector; SIGReg is kept separate so objective edits stay local.
    model = NanoPathFM(cfg).to(device)
    sigreg = SIGReg().to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Excludes the projector head because downstream probes read pre-projector pooled registers,
    # so the projector is pretraining-only scaffolding and shouldn't count toward the leaderboard
    # size cap. The sum below is exact for dense models; MoE / sparse-routing contributors must
    # rewrite it to count per-token activated params.
    backbone_activated_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and not n.startswith("projector."))
    print(f"{console_prefix()} backbone_activated_params: {backbone_activated_params:,} / 150,000,000", flush=True)
    if backbone_activated_params > 150_000_000:
        raise ValueError(f"backbone_activated_params={backbone_activated_params:,} exceeds the 150M activated-parameter leaderboard cap; failing fast to avoid spending training compute on an ineligible backbone.")
    # Optimizer groups live in model.py so weight-decay policy follows model changes.
    opt = torch.optim.AdamW(model.param_groups(train_cfg["weight_decay"]), lr=1.0, betas=(0.9, 0.95))
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
    output_dir = Path(cfg["project"]["output_dir"])
    wandb_dir = Path(cfg["project"]["wandb_dir"])
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_stdout_path = os.environ.get("NANOPATH_SLURM_STDOUT")
    slurm_stderr_path = os.environ.get("NANOPATH_SLURM_STDERR")
    # Fresh launches fully replace the run directory so repeated use of a checked-in
    # output_dir like /data/nanopath/leader never trips over stale artifacts.
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
        # Resume restores training progress, optimizer state, and wandb identity.
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
        ) = (
            int(checkpoint["step"]),
            float(checkpoint["lr"]),
            float(checkpoint["best_val_total"]),
            int(checkpoint["examples_seen"]),
            int(checkpoint["visible_patch_presentations"]),
            int(checkpoint["train_flops"]),
        )
        wandb_meta = dict(checkpoint["wandb"])
    if rank == 0:
        # Rank 0 owns external side effects; nonzero ranks only participate in training/eval collectives.
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
        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = sorted(d for d in dirs if d not in {".git", ".venv", "__pycache__", ".claude"})
            for name in sorted(files):
                path = Path(root) / name
                source_artifact.add_file(str(path), name=str(path.relative_to(repo_dir)))
        wandb_run.log_artifact(source_artifact)
        wandb_meta = {"project": "nanopath", "id": wandb_run.id, "name": cfg["project"]["name"]}
    else:
        wandb_run = None
    if rank == 0:
        # Build byte-offset caches once so workers can jump directly to sample-list lines.
        prepare_sample_list_offsets(cfg)
    if distributed:
        dist.barrier()
    train_ds = RandomTCGADataset(cfg, cfg["data"]["train_split"])
    val_ds = RandomTCGADataset(cfg, cfg["data"]["val_split"])
    probe_state = prepare_probe_state(cfg, output_dir) if rank == 0 and probe_enabled(cfg) else None

    # Wrap datasets in DistributedSampler only when DDP is active; the recipe defines global batch size.
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
    model = torch.compile(model, dynamic=False)
    # EMA is a lightweight target snapshot for probes, not a separate forward path during training.
    model_state = root_model.state_dict()
    ema_source = checkpoint["model_ema"] if resume_path is not None else model_state
    ema_state = {k: ema_source[k].detach().to(device).clone() for k in model_state}
    global_patches = (train_cfg["global_size"] // cfg["model"]["patch_size"]) ** 2
    local_patches = (train_cfg["local_size"] // cfg["model"]["patch_size"]) ** 2
    last_time = time.time()
    last_examples = examples_seen
    last_visible_patch_presentations = visible_patch_presentations
    last_train_flops = train_flops
    unique_tile_patch_count = (SAMPLE_LIST_PATCH_SIZE // cfg["model"]["patch_size"]) ** 2
    seen_ids = {"sample": set(), "slide": set(), "patient": set()}
    pending_ids = {key: set() for key in seen_ids}

    # Full checkpoint payload used for latest.pt resume.
    def checkpoint_payload(next_step):
        return {
            "model": {k: v.detach().cpu().clone() for k, v in root_model.state_dict().items()},
            "model_ema": {k: v.detach().cpu().clone() for k, v in ema_state.items()},
            "opt": opt.state_dict(),
            "step": next_step,
            "best_val_total": best_val_total,
            "examples_seen": examples_seen,
            "visible_patch_presentations": visible_patch_presentations,
            "train_flops": train_flops,
            "lr": lr,
            "ema_decay": ema_decay,
            "probe_model_weights": probe_model_weights,
            "wandb": wandb_meta,
            "config": cfg,
        }

    # Smaller probe payload: probe workers only need weights and config, not optimizer state.
    def probe_checkpoint_payload(next_step):
        return {
            "model": {k: v.detach().cpu().clone() for k, v in root_model.state_dict().items()},
            "model_ema": {k: v.detach().cpu().clone() for k, v in ema_state.items()},
            "step": next_step,
            "ema_decay": ema_decay,
            "probe_model_weights": probe_model_weights,
            "config": cfg,
        }

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

    # Ingest completed probe result JSONs into metrics.jsonl and wandb.
    def log_probe_results():
        if probe_state is None:
            return
        collect_probe_results(probe_state, wandb_run, metrics_path)

    # Record validation losses with an optional event tag such as final_eval.
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

    # Queue the furthest crossed FLOP milestone so delayed validation does not run stale probes.
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

    if rank == 0:
        log_probe_results()
    if distributed:
        dist.barrier()
    max_train_flops = int(train_cfg["max_train_flops"])
    warmup_flop_fraction = float(train_cfg["warmup_flop_fraction"])
    warmup_train_flops = math.ceil(max_train_flops * warmup_flop_fraction)
    warmdown_train_flops = round(max_train_flops * float(train_cfg["warmdown_flop_fraction"]))
    final_lr_frac = float(train_cfg["final_lr_frac"])
    # Probe targets are FLOP milestones, not step milestones, so comparisons survive batch-size changes.
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
    stop_reason = "max_train_flops" if train_flops >= max_train_flops else None
    last_eval_step = step if math.isfinite(best_val_total) else -1
    last_saved_step = step if resume_path is not None else 0
    last_console_step = step
    last_console_monotonic = time.monotonic()
    data_wait_started_at = time.monotonic()

    while stop_reason is None:
        if distributed:
            train_loader.sampler.set_epoch(step + train_cfg["seed"])
        for batch in train_loader:
            batch_started_at = time.monotonic()
            data_seconds = batch_started_at - data_wait_started_at
            model.train()
            completed_step = step + 1
            should_log = completed_step == 1 or completed_step % train_cfg["log_every"] == 0
            # Data identifiers stay on CPU and feed coverage metrics; image tensors move below.
            for key, batch_key in (("sample", "sample_idx"), ("slide", "slide_id"), ("patient", "patient_id")):
                pending_ids[key].update(int(x) for x in batch[batch_key].tolist())
            global_views, local_views = [batch[key].to(device, non_blocking=True) for key in ("global_views", "local_views")]
            current_batch = global_views.shape[0]
            global_batch = current_batch * world_size
            # Training budget is estimated FLOPs from visible patch presentations and trainable params.
            visible_now = global_batch * (train_cfg["global_views"] * global_patches + train_cfg["local_views"] * local_patches)
            step_train_flops = int(6 * model_params * visible_now)
            # Nanochat-style schedule: linear warmup, flat middle, linear warmdown by FLOPs.
            lr_flops = min(max_train_flops, train_flops + step_train_flops)
            lr_multiplier = min(1.0, lr_flops / warmup_train_flops)
            lr_multiplier = min(lr_multiplier, final_lr_frac + (1.0 - final_lr_frac) * (max_train_flops - lr_flops) / warmdown_train_flops)
            for group in opt.param_groups:
                group["lr"] = lr * lr_multiplier
            autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" and train_cfg["bf16"] else contextlib.nullcontext()
            sigreg_generator = torch.Generator(device=device.type)
            sigreg_generator.manual_seed(train_cfg["seed"] + step)
            with autocast:
                # Model returns per-view projector features; loss_terms() defines the pretraining objective.
                proj = model(global_views, local_views, train_cfg)
                jepa_pred_loss, sig_loss, total_loss = loss_terms(
                    sigreg,
                    proj,
                    train_cfg,
                    sigreg_generator,
                    torch.cat(dist_nn.all_gather(proj), dim=0) if distributed else proj,
                )
                proj_std = proj.float().reshape(-1, proj.shape[-1]).std(dim=0).mean() if should_log else None
            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            # Gradient norm is both a stability diagnostic and the value used for optional clipping.
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
            # EMA mirrors floating-point model tensors and directly copies non-floating buffers.
            ema_floating, model_floating = [], []
            for key, value in root_model.state_dict().items():
                source = value.detach()
                if torch.is_floating_point(ema_state[key]):
                    ema_floating.append(ema_state[key])
                    model_floating.append(source)
                else:
                    ema_state[key].copy_(source)
            torch._foreach_mul_(ema_floating, ema_decay)
            torch._foreach_add_(ema_floating, model_floating, alpha=1.0 - ema_decay)
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
                current_lr = opt.param_groups[0]["lr"]
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
                    "grad_norm": grad_norm,
                    "param_norm": param_norm,
                    "grad_param_ratio": grad_param_ratio,
                    "grad_clip_scale": grad_clip_scale,
                }
                train_log.update(unique_counts)
                print(
                    f"{console_prefix()} Training  "
                    f"[{completed_step}/{total_steps_estimate}]  eta: {eta_string}  gap: {console_gap_ms:.2f} ms  "
                    f"lr: {current_lr:.6f}  lrm: {lr_multiplier:.4f}  "
                    f"current_batch_size: {batch_size}  "
                    f"total_loss: {reduced['total']:.4f}  "
                    f"jepa_pred_loss: {reduced['jepa_pred']:.4f}  "
                    f"sig_loss: {reduced['sig']:.4f}  "
                    f"proj_std: {reduced['proj_std']:.4f}  "
                    f"grad_norm: {grad_norm:.4f}  flops/s: {flops_per_sec:.3e}  gpu: {gpu_util_pct:.0f}  "
                    f"time: {step_seconds:.6f}  data: {data_seconds:.6f}  "
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
                log_probe_results()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
            if completed_step % train_cfg["eval_every"] == 0:
                # Validation is also the synchronization point for probe milestones.
                if rank == 0:
                    print(f"{console_prefix()} Validation  [{completed_step}]  start", flush=True)
                val = evaluate(model, sigreg, val_loader, cfg, device, world_size)
                if val["total"] < best_val_total:
                    best_val_total = val["total"]
                last_eval_step = completed_step
                if rank == 0:
                    log_val(completed_step, val)
                    maybe_run_probe(completed_step)
                    log_probe_results()
                if distributed:
                    dist.barrier()
            if rank == 0 and save_checkpoints and completed_step % save_every == 0:
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
    train_loop_wall_seconds = time.monotonic() - train_loop_started_at
    final_unique_counts = flush_unique_counts()
    if distributed:
        dist.barrier()
    if step > 0 and last_eval_step != step:
        # Always end on a validation/probe opportunity so summary.json reflects final weights.
        if rank == 0:
            print(f"{console_prefix()} Validation  [{step}]  start final_eval", flush=True)
        val = evaluate(model, sigreg, val_loader, cfg, device, world_size)
        if val["total"] < best_val_total:
            best_val_total = val["total"]
        last_eval_step = step
        # Final probes have their own readers; close pretraining workers before they compete for CPU and IO.
        if train_cfg["num_workers"] > 0:
            for loader in (train_loader, val_loader):
                if loader._iterator is not None:
                    loader._iterator._shutdown_workers()
                    loader._iterator = None
        if rank == 0:
            log_val(step, val, "final_eval")
            maybe_run_probe(step)
            log_probe_results()
        if distributed:
            dist.barrier()
    if not math.isfinite(best_val_total):
        raise ValueError("run finished without a finite best_val_total; check max_train_flops, eval_every, validation data, and loss stability")
    if rank == 0:
        log_probe_results()
        if save_checkpoints and step > 0 and step != last_saved_step:
            print(f"{console_prefix()} Checkpoint  [{step}]  save: latest.pt", flush=True)
            torch.save(checkpoint_payload(step), latest_checkpoint_path)
            for stale_checkpoint_path in output_dir.glob("step_*.pt"):
                stale_checkpoint_path.unlink()
            last_saved_step = step
        # Summary is the small, stable artifact downstream scripts and humans compare across runs.
        summary = {
            "project": cfg["project"]["name"],
            "family": cfg["project"]["family"],
            "recipe_id": cfg["project"]["recipe_id"],
            "config_path": cfg["config_path"],
            "slurm_job_id": slurm_job_id,
            "slurm_stdout_path": slurm_stdout_path,
            "slurm_stderr_path": slurm_stderr_path,
            "model_params": model_params,
            "backbone_activated_params": backbone_activated_params,
            "world_size": world_size,
            "batch_size_per_rank": batch_size,
            "global_batch_size": batch_size * world_size,
            "max_train_flops": max_train_flops,
            "train_loop_wall_seconds": train_loop_wall_seconds,
            "stop_reason": stop_reason,
            "steps_completed": step,
            "best_val_total": best_val_total,
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
        for key in summary.keys() - {"best_val_total"}:
            wandb_run.summary[key] = summary[key]
        wandb_run.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
