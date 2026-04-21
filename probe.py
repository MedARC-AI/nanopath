import json
import math
import os
import random
import shutil
import subprocess
import sys
import fcntl
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import wandb


THUNDER_REPO = Path("/admin/home/paul/thunder")
THUNDER_PYTHON = THUNDER_REPO / ".venv" / "bin" / "python"
THUNDER_BIN = THUNDER_REPO / ".venv" / "bin" / "thunder"
THUNDER_MODEL = Path(__file__).resolve().with_name("thunder_adapter.py")
SLURM_PARTITION = "n"
SLURM_ACCOUNT = "sophont"
SLURM_QOS = "normal"
SLURM_GRES = "gpu:nvidia_h100_80gb_hbm3:1"
SLURM_TIME_LIMIT = "08:00:00"
LINEAR_PROBE_PARALLEL_DATASETS = 5
PATCH_CAMELYON_SUBSET_SEED = 1337
PATCH_CAMELYON_SUBSET_SIZES = {"train": 3072, "valid": 768, "test": 768}
DATASET_ROOTS = {
    "bach": Path("/block/eva-data/bach"),
    "bracs": Path("/block/eva-data/bracs"),
    "break_his": Path("/block/eva-data/breakhis"),
    "mhist": Path("/block/eva-data/mhist"),
    "pcam": Path("/block/eva-data/patch_camelyon"),
}
THUNDER_DATASET_NAMES = {
    "bach": "bach",
    "bracs": "bracs",
    "break_his": "break_his",
    "mhist": "mhist",
    "pcam": "patch_camelyon",
}


def probe_paths(output_dir):
    output_dir = Path(output_dir)
    probe_dir = output_dir / "thunder"
    return {
        "probe_dir": probe_dir,
        "state_path": probe_dir / "state.json",
        "lock_path": probe_dir / "state.lock",
        "results_dir": probe_dir / "results",
        "slurm_dir": probe_dir / "slurm",
        "scratch_root": Path("/tmp/nanopath-thunder") / output_dir.name,
    }


def probe_enabled(cfg):
    return bool(cfg["probe"]["enabled"]) and len(cfg["probe"]["datasets"]) > 0


def write_probe_state(state):
    state["paths"]["state_path"].write_text(json.dumps(state["data"], indent=2) + "\n")


def prepare_probe_state(cfg, output_dir):
    paths = probe_paths(output_dir)
    for path in [paths["probe_dir"], paths["results_dir"], paths["slurm_dir"], paths["scratch_root"]]:
        path.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 5,
        "family": str(cfg["project"]["family"]),
        "datasets": [str(x) for x in cfg["probe"]["datasets"]],
        "count": int(cfg["probe"]["count"]),
        "active": None,
        "queued": None,
        "logged_results": [],
    }
    if paths["state_path"].exists():
        previous = json.loads(paths["state_path"].read_text())
        if previous["version"] != data["version"]:
            raise ValueError(f"unsupported Thunder probe state version: {previous['version']}")
        if previous["family"] != data["family"]:
            raise ValueError(f"Thunder probe family changed from {previous['family']} to {data['family']}")
        if previous["datasets"] != data["datasets"]:
            raise ValueError(f"Thunder probe datasets changed from {previous['datasets']} to {data['datasets']}")
        if "count" in previous and previous["count"] != data["count"]:
            raise ValueError(f"Thunder probe count changed from {previous['count']} to {data['count']}")
        data["active"] = previous["active"]
        data["queued"] = previous["queued"]
        data["logged_results"] = previous["logged_results"]
    for dataset in data["datasets"]:
        if dataset not in DATASET_ROOTS:
            raise ValueError(f"unsupported Thunder probe dataset: {dataset}")
    state = {"paths": paths, "data": data}
    write_probe_state(state)
    return state


def checkpoint_request(state, checkpoint_step, probe_ordinal, target_flops, target_fraction):
    step_tag = f"step_{checkpoint_step:07d}"
    return {
        "checkpoint_step": int(checkpoint_step),
        "train_step": int(checkpoint_step),
        "probe_ordinal": int(probe_ordinal),
        "probe_count": int(state["data"]["count"]),
        "target_flops": int(target_flops),
        "target_fraction": float(target_fraction),
        "checkpoint_path": str(state["paths"]["probe_dir"] / f"{step_tag}.pt"),
        "request_path": str(state["paths"]["probe_dir"] / f"{step_tag}.request.json"),
        "result_path": str(state["paths"]["results_dir"] / f"{step_tag}.json"),
        "job_script": str(state["paths"]["slurm_dir"] / f"probe-{step_tag}.sbatch"),
        "model_name": f"{state['data']['family']}_{step_tag}",
        "datasets": list(state["data"]["datasets"]),
        "job_id": None,
        "submitted_at_utc": None,
    }


def queue_probe_job(cfg, state, checkpoint_payload, checkpoint_step, probe_ordinal, target_flops, target_fraction):
    if not probe_enabled(cfg):
        return state
    with state["paths"]["lock_path"].open("w") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        state["data"] = json.loads(state["paths"]["state_path"].read_text())
        request = checkpoint_request(state, checkpoint_step, probe_ordinal, target_flops, target_fraction)
        if not Path(request["checkpoint_path"]).exists():
            torch.save(checkpoint_payload, request["checkpoint_path"])
        if state["data"]["active"] is None:
            submit_probe_job(state, request)
            return state
        if state["data"]["queued"] is not None and Path(state["data"]["queued"]["checkpoint_path"]).exists():
            Path(state["data"]["queued"]["checkpoint_path"]).unlink()
        state["data"]["queued"] = request
        write_probe_state(state)
    return state


def submit_probe_job(state, request):
    for dataset in request["datasets"]:
        if not DATASET_ROOTS[dataset].exists():
            raise FileNotFoundError(f"missing Thunder dataset root for {dataset}: {DATASET_ROOTS[dataset]}")
    Path(request["request_path"]).write_text(json.dumps(request, indent=2) + "\n")
    runtime_dir = f"{state['paths']['scratch_root']}/step_{request['checkpoint_step']:07d}-$SLURM_JOB_ID"
    script = f"""#!/usr/bin/env bash
set -euo pipefail
unset WANDB_SERVICE
export NANOPATH_THUNDER_RUNTIME_DIR="{runtime_dir}"
trap 'rm -rf "$NANOPATH_THUNDER_RUNTIME_DIR"' EXIT
"{THUNDER_PYTHON}" "{Path(__file__).resolve()}" "{request['request_path']}"
"""
    Path(request["job_script"]).write_text(script)
    submit = subprocess.run(
        [
            "sbatch",
            "--parsable",
            "--partition",
            SLURM_PARTITION,
            "--account",
            SLURM_ACCOUNT,
            "--qos",
            SLURM_QOS,
            "--nodes",
            "1",
            "--ntasks",
            "1",
            "--time",
            SLURM_TIME_LIMIT,
            "--gres",
            SLURM_GRES,
            "--job-name",
            "nanopath-thunder",
            "--output",
            str(state["paths"]["slurm_dir"] / "nanopath-thunder-%j.out"),
            "--error",
            str(state["paths"]["slurm_dir"] / "nanopath-thunder-%j.err"),
            request["job_script"],
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    request["job_id"] = submit.stdout.strip()
    request["submitted_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    Path(request["request_path"]).write_text(json.dumps(request, indent=2) + "\n")
    state["data"]["active"] = request
    state["data"]["queued"] = None
    write_probe_state(state)


def prepare_patch_camelyon(runtime_dir):
    import h5py
    import numpy as np
    import yaml

    dst_root = runtime_dir / "datasets" / "patch_camelyon"
    dst_root.mkdir(parents=True, exist_ok=True)
    custom_dir = runtime_dir / "custom_datasets"
    custom_dir.mkdir(parents=True, exist_ok=True)
    for split_idx, split in enumerate(["train", "valid", "test"]):
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
        "test": {
            "images": "camelyonpatch_level_2_split_test_x.h5",
            "labels": "camelyonpatch_level_2_split_test_y.h5",
        },
        "train_few_shot": {str(k): {"images": [], "labels": []} for k in [1, 2, 4, 8, 16]},
    }
    with h5py.File(dst_root / data_splits["train"]["labels"], "r") as handle:
        labels = np.array(handle[next(iter(handle.keys()))]).reshape(-1)
    label_to_images = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_images[int(label)].append(idx)
    few_shot_rng = random.Random(PATCH_CAMELYON_SUBSET_SEED)
    for shots in data_splits["train_few_shot"]:
        for _ in range(1000):
            images = []
            targets = []
            for label in sorted(label_to_images):
                picks = few_shot_rng.sample(label_to_images[label], int(shots))
                images.extend(picks)
                targets.extend([label] * int(shots))
            data_splits["train_few_shot"][shots]["images"].append(images)
            data_splits["train_few_shot"][shots]["labels"].append(targets)
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
                    "adversarial_attack",
                    "alignment_scoring",
                    "image_retrieval",
                    "knn",
                    "linear_probing",
                    "pre_computing_embeddings",
                    "simple_shot",
                    "transformation_invariance",
                    "zero_shot_vlm",
                ],
                "nb_train_samples": PATCH_CAMELYON_SUBSET_SIZES["train"],
                "nb_val_samples": PATCH_CAMELYON_SUBSET_SIZES["valid"],
                "nb_test_samples": PATCH_CAMELYON_SUBSET_SIZES["test"],
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


def run_probe_job(request_path):
    request = json.loads(Path(request_path).read_text())
    runtime_dir = Path(os.environ["NANOPATH_THUNDER_RUNTIME_DIR"])
    datasets_dir = runtime_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["THUNDER_BASE_DATA_FOLDER"] = str(runtime_dir)
    env["NANOPATH_THUNDER_CKPT"] = request["checkpoint_path"]
    env["NANOPATH_THUNDER_MODEL_NAME"] = request["model_name"]
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent)
    env["THUNDER_WANDB_MODE"] = "disabled"
    env["WANDB_MODE"] = "disabled"
    env["WANDB_SILENT"] = "true"
    targets = {}
    builtin = []
    for dataset in request["datasets"]:
        thunder_name = THUNDER_DATASET_NAMES[dataset]
        if dataset == "pcam":
            targets[dataset] = prepare_patch_camelyon(runtime_dir)
        else:
            (datasets_dir / thunder_name).symlink_to(DATASET_ROOTS[dataset])
            builtin.append(thunder_name)
            targets[dataset] = thunder_name
    if len(builtin) > 0:
        subprocess.run([str(THUNDER_BIN), "generate-data-splits", *builtin], cwd=THUNDER_REPO, env=env, check=True)
    for dataset in request["datasets"]:
        subprocess.run(
            [str(THUNDER_BIN), "benchmark", f"custom:{THUNDER_MODEL}", targets[dataset], "pre_computing_embeddings"],
            cwd=THUNDER_REPO,
            env=env,
            check=True,
        )
    for start in range(0, len(request["datasets"]), LINEAR_PROBE_PARALLEL_DATASETS):
        procs = []
        for dataset in request["datasets"][start : start + LINEAR_PROBE_PARALLEL_DATASETS]:
            procs.append(
                (
                    dataset,
                    subprocess.Popen(
                        [
                            str(THUNDER_BIN),
                            "benchmark",
                            f"custom:{THUNDER_MODEL}",
                            targets[dataset],
                            "linear_probing",
                            "--loading-mode",
                            "embedding_pre_loading",
                        ],
                        cwd=THUNDER_REPO,
                        env=env,
                    ),
                )
            )
        for dataset, proc in procs:
            if proc.wait() != 0:
                raise RuntimeError(f"Thunder linear probing failed for {dataset}")
    metrics = {}
    results = {}
    for dataset in request["datasets"]:
        thunder_name = THUNDER_DATASET_NAMES[dataset]
        base = runtime_dir / "outputs"
        best_ckpt_path = base / "ckpts" / thunder_name / request["model_name"] / "frozen" / "best_model.pth"
        val_metrics = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)["val_metrics"]
        metrics[f"probe_{dataset}_val_f1"] = float(val_metrics["f1"]["metric_score"])
        results[dataset] = {"val_metrics": val_metrics}
    f1_keys = [key for key in metrics if key.endswith("_val_f1")]
    metrics["mean_probe_f1"] = sum(metrics[key] for key in f1_keys) / len(f1_keys)
    Path(request["result_path"]).write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "checkpoint_step": request["checkpoint_step"],
                "train_step": request["train_step"],
                "probe_ordinal": request["probe_ordinal"],
                "probe_count": request["probe_count"],
                "target_flops": request["target_flops"],
                "target_fraction": request["target_fraction"],
                "checkpoint_path": request["checkpoint_path"],
                "model_name": request["model_name"],
                "datasets": request["datasets"],
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
        if state["data"]["active"] is not None and str(state["data"]["active"]["job_id"]) == os.environ["SLURM_JOB_ID"]:
            state["data"]["active"] = None
        if state["data"]["active"] is None and state["data"]["queued"] is not None:
            submit_probe_job(state, state["data"]["queued"])
        else:
            write_probe_state(state)
    output_dir = Path(request["checkpoint_path"]).parents[1]
    if (output_dir / "summary.json").exists():
        collect_finished_probe_results(output_dir)


def collect_probe_results(state, wandb_run, metrics_path, output_dir, best_val_mean_probe_f1, best_val_mean_probe_f1_step, best_probe_scores, log_step):
    if wandb_run is None:
        return best_val_mean_probe_f1, best_val_mean_probe_f1_step, best_probe_scores
    with state["paths"]["lock_path"].open("w") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        state["data"] = json.loads(state["paths"]["state_path"].read_text())
        active = state["data"]["active"]
        if active is not None and not Path(active["result_path"]).exists():
            if subprocess.run(["squeue", "--noheader", "--jobs", str(active["job_id"]), "--format", "%T"], check=True, capture_output=True, text=True).stdout.strip() == "":
                slurm_state = "UNKNOWN"
                for line in subprocess.run(
                    ["sacct", "-X", "-n", "-j", str(active["job_id"]), "--format", "State"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.splitlines():
                    line = line.strip()
                    if line != "":
                        slurm_state = line.split()[0]
                        break
                Path(active["result_path"]).write_text(
                    json.dumps(
                        {
                            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "slurm_job_id": active["job_id"],
                            "checkpoint_step": active["checkpoint_step"],
                            "train_step": active["train_step"],
                            "probe_ordinal": active["probe_ordinal"],
                            "probe_count": active["probe_count"],
                            "target_flops": active["target_flops"],
                            "target_fraction": active["target_fraction"],
                            "checkpoint_path": active["checkpoint_path"],
                            "model_name": active["model_name"],
                            "datasets": state["data"]["datasets"],
                            "status": "missing_result" if slurm_state == "COMPLETED" else "failed",
                            "slurm_state": slurm_state,
                            "metrics": {},
                            "results": {},
                        },
                        indent=2,
                    )
                    + "\n"
                )
        logged = set(state["data"]["logged_results"])
        for result_path in sorted(state["paths"]["results_dir"].glob("step_*.json")):
            result_path_str = str(result_path)
            if result_path_str in logged:
                continue
            result = json.loads(result_path.read_text())
            metrics = {key: float(value) for key, value in result["metrics"].items()}
            with metrics_path.open("a") as handle:
                handle.write(json.dumps({"event": "thunder_probe", "step": result["train_step"], "status": result["status"], **metrics}) + "\n")
            if len(metrics) > 0:
                wandb_run.log(
                    {
                        "probe/step": result["train_step"],
                        **{f"probe/{key}": value for key, value in metrics.items()},
                    },
                    step=log_step,
                )
            if result["status"] == "ok" and "mean_probe_f1" in metrics:
                for key, value in metrics.items():
                    best_probe_scores[key] = max(best_probe_scores.get(key, float("-inf")), value)
                if metrics["mean_probe_f1"] > best_val_mean_probe_f1:
                    best_val_mean_probe_f1 = metrics["mean_probe_f1"]
                    best_val_mean_probe_f1_step = int(result["train_step"])
                    shutil.copy2(result["checkpoint_path"], output_dir / "best_mean_probe_f1.pt")
            checkpoint_path = Path(result["checkpoint_path"])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            logged.add(result_path_str)
        state["data"]["logged_results"] = sorted(logged)
        active = state["data"]["active"]
        if active is not None and Path(active["result_path"]).exists():
            state["data"]["active"] = None
        if state["data"]["active"] is None and state["data"]["queued"] is not None:
            submit_probe_job(state, state["data"]["queued"])
        else:
            write_probe_state(state)
    return best_val_mean_probe_f1, best_val_mean_probe_f1_step, best_probe_scores


def collect_finished_probe_results(output_dir):
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return
    state = {"paths": probe_paths(output_dir), "data": json.loads((output_dir / "thunder" / "state.json").read_text())}
    training_checkpoints = sorted(output_dir.glob("step_*.pt"))
    if len(training_checkpoints) == 0:
        raise FileNotFoundError(f"missing training checkpoints in {output_dir}")
    checkpoint = torch.load(training_checkpoints[-1], map_location="cpu", weights_only=False)
    wandb_meta = checkpoint["wandb"]
    if wandb_meta is None:
        raise ValueError(f"missing wandb metadata in {training_checkpoints[-1]}")
    os.environ.pop("WANDB_SERVICE", None)
    wandb_run = wandb.init(
        project=wandb_meta["project"],
        name=wandb_meta["name"],
        id=wandb_meta["id"],
        resume="must",
        dir="/data/nanopath/wandb",
        config=checkpoint["config"],
        settings=wandb.Settings(
            console="redirect",
            console_multipart=True,
            console_chunk_max_seconds=15,
        ),
    )
    wandb_run.define_metric("probe/step", hidden=True)
    wandb_run.define_metric("probe/*", step_metric="probe/step")
    summary = json.loads(summary_path.read_text())
    best_val_mean_probe_f1 = float("-inf") if summary["best_val_mean_probe_f1"] is None else float(summary["best_val_mean_probe_f1"])
    best_val_mean_probe_f1_step = int(summary["best_val_mean_probe_f1_step"])
    best_probe_scores = {}
    for key, value in summary.items():
        if key == "mean_probe_f1" or (key.startswith("probe_") and key.endswith("_val_f1")):
            best_probe_scores[key] = float(value)
    best_val_mean_probe_f1, best_val_mean_probe_f1_step, best_probe_scores = collect_probe_results(
        state,
        wandb_run,
        output_dir / "metrics.jsonl",
        output_dir,
        best_val_mean_probe_f1,
        best_val_mean_probe_f1_step,
        best_probe_scores,
        int(summary["steps_completed"]) + len(state["data"]["logged_results"]) + 1,
    )
    state["data"] = json.loads(state["paths"]["state_path"].read_text())
    summary["best_val_mean_probe_f1"] = None if not math.isfinite(best_val_mean_probe_f1) else best_val_mean_probe_f1
    summary["best_val_mean_probe_f1_step"] = best_val_mean_probe_f1_step
    summary["thunder_probe_active_job_id"] = None if state["data"]["active"] is None else state["data"]["active"]["job_id"]
    summary["thunder_probe_active_step"] = None if state["data"]["active"] is None else state["data"]["active"]["train_step"]
    summary["thunder_probe_queued_step"] = None if state["data"]["queued"] is None else state["data"]["queued"]["train_step"]
    summary.update(best_probe_scores)
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
