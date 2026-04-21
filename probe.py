import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


THUNDER_REPO = Path("/admin/home/paul/thunder")
THUNDER_VENV = THUNDER_REPO / ".venv" / "bin" / "activate"
THUNDER_MODEL = Path(__file__).resolve().with_name("thunder_adapter.py")
SLURM_PARTITION = "main"
SLURM_ACCOUNT = "sophont"
SLURM_QOS = "normal"
SLURM_GRES = "gpu:nvidia_h100_80gb_hbm3:1"
SLURM_CPUS_PER_TASK = "16"
SLURM_TIME_LIMIT = "08:00:00"
LINEAR_PROBE_PARALLEL_DATASETS = 3
DATASET_ROOTS = {
    "bach": Path("/block/eva-data/bach"),
    "break_his": Path("/block/eva-data/breakhis"),
    "mhist": Path("/block/eva-data/mhist"),
}


def probe_enabled(cfg):
    probe_cfg = cfg["probe"]
    return bool(probe_cfg["enabled"]) and len(probe_cfg["datasets"]) > 0


def prepare_probe_state(cfg, output_dir):
    probe_dir = output_dir / "thunder"
    paths = {
        "probe_dir": probe_dir,
        "state_path": probe_dir / "state.json",
        "results_dir": probe_dir / "results",
        "slurm_dir": probe_dir / "slurm",
        "scratch_root": Path("/tmp/nanopath-thunder") / str(cfg["project"]["name"]),
    }
    probe_dir.mkdir(parents=True, exist_ok=True)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    paths["slurm_dir"].mkdir(parents=True, exist_ok=True)
    paths["scratch_root"].mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "family": str(cfg["project"]["family"]),
        "datasets": [str(x) for x in cfg["probe"]["datasets"]],
        "active": None,
        "queued": None,
        "logged_results": [],
    }
    if paths["state_path"].exists():
        data.update(json.loads(paths["state_path"].read_text()))
    for dataset in data["datasets"]:
        if dataset not in DATASET_ROOTS:
            raise ValueError(f"unsupported Thunder probe dataset: {dataset}")
    state = {"paths": paths, "data": data}
    write_probe_state(state)
    return state


def write_probe_state(state):
    state["paths"]["state_path"].write_text(json.dumps(state["data"], indent=2) + "\n")


def checkpoint_request(state, checkpoint_step):
    checkpoint_path = state["paths"]["probe_dir"] / f"step_{checkpoint_step:07d}.pt"
    result_path = state["paths"]["results_dir"] / f"step_{checkpoint_step:07d}.json"
    job_script = state["paths"]["slurm_dir"] / f"probe-step-{checkpoint_step:07d}.sbatch"
    return {
        "checkpoint_step": int(checkpoint_step),
        "train_step": int(checkpoint_step) - 1,
        "checkpoint_path": str(checkpoint_path),
        "result_path": str(result_path),
        "job_script": str(job_script),
        "job_id": None,
        "submitted_at_utc": None,
    }


def queue_probe_job(cfg, state, checkpoint_payload, checkpoint_step):
    if not probe_enabled(cfg):
        return state
    request = checkpoint_request(state, checkpoint_step)
    checkpoint_path = Path(request["checkpoint_path"])
    if not checkpoint_path.exists():
        torch.save(checkpoint_payload, checkpoint_path)
    active = state["data"]["active"]
    if active is None:
        submit_probe_job(state, request)
        return state
    queued = state["data"]["queued"]
    if queued is not None:
        queued_checkpoint = Path(queued["checkpoint_path"])
        if queued_checkpoint.exists():
            queued_checkpoint.unlink()
    state["data"]["queued"] = request
    write_probe_state(state)
    return state


def submit_probe_job(state, request):
    datasets = [str(x) for x in state["data"]["datasets"]]
    parallel_datasets = min(LINEAR_PROBE_PARALLEL_DATASETS, max(1, len(datasets)))
    for dataset in datasets:
        if not DATASET_ROOTS[dataset].exists():
            raise FileNotFoundError(f"missing Thunder dataset root for {dataset}: {DATASET_ROOTS[dataset]}")
    model_name = f"{state['data']['family']}_step_{request['checkpoint_step']:07d}"
    script = f"""#!/usr/bin/env bash
set -uo pipefail
set -x
runtime_dir="{state['paths']['scratch_root']}/step_{request['checkpoint_step']:07d}-${{SLURM_JOB_ID:-none}}"
result_path="{request['result_path']}"
checkpoint_path="{request['checkpoint_path']}"
mkdir -p "$runtime_dir/datasets"
export THUNDER_BASE_DATA_FOLDER="$runtime_dir"
export NANOPATH_THUNDER_CKPT="$checkpoint_path"
export NANOPATH_THUNDER_MODEL_NAME="{model_name}"
export PYTHONPATH="{Path(__file__).resolve().parent}:${{PYTHONPATH:-}}"
export THUNDER_WANDB_MODE=disabled
export WANDB_MODE=disabled
export WANDB_SILENT=true
source "{THUNDER_VENV}"
cd "{THUNDER_REPO}"
"""
    for dataset in datasets:
        script += f'ln -sfn "{DATASET_ROOTS[dataset]}" "$runtime_dir/datasets/{dataset}"\n'
    script += "status=0\n"
    script += f"thunder generate-data-splits {' '.join(datasets)} || status=$?\n"
    for dataset in datasets:
        script += f'if [[ "$status" -eq 0 ]]; then thunder benchmark custom:{THUNDER_MODEL} {dataset} pre_computing_embeddings || status=$?; fi\n'
    if parallel_datasets > 1 and len(datasets) > 1:
        script += 'parallel_log_dir="$runtime_dir/parallel-logs"\n'
        script += 'mkdir -p "$parallel_log_dir"\n'
        script += "pids=()\n"
        script += "labels=()\n"
        script += """wait_group() {
  local i rc=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "Completed ${labels[$i]}"
    else
      rc=$?
      echo "Failed ${labels[$i]} with exit code $rc" >&2
      for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
      done
      wait || true
      status=$rc
      break
    fi
  done
  pids=()
  labels=()
  return "$rc"
}
"""
        for start in range(0, len(datasets), parallel_datasets):
            batch = datasets[start : start + parallel_datasets]
            script += 'if [[ "$status" -eq 0 ]]; then\n'
            for dataset in batch:
                label = f"{dataset}-linear_probing"
                script += "(\n"
                script += "  set -euo pipefail\n"
                script += "  set -o pipefail\n"
                script += (
                    f"  thunder benchmark custom:{THUNDER_MODEL} {dataset} linear_probing --loading-mode embedding_pre_loading "
                    f"2>&1 | sed -u 's/^/[{label}] /' | tee \"$parallel_log_dir/{label}.log\"\n"
                )
                script += ") &\n"
                script += 'pids+=("$!")\n'
                script += f'labels+=("{label}")\n'
            script += "wait_group || true\n"
            script += "fi\n"
    else:
        for dataset in datasets:
            script += f'if [[ "$status" -eq 0 ]]; then thunder benchmark custom:{THUNDER_MODEL} {dataset} linear_probing --loading-mode embedding_pre_loading || status=$?; fi\n'
    script += 'export NANOPATH_THUNDER_STATUS="$status"\n'
    script += f"""python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

runtime_dir = Path(os.environ["THUNDER_BASE_DATA_FOLDER"])
result_path = Path("{request['result_path']}")
datasets = {datasets!r}
model_name = os.environ["NANOPATH_THUNDER_MODEL_NAME"]
checkpoint_path = Path("{request['checkpoint_path']}")
status_code = int(os.environ.get("NANOPATH_THUNDER_STATUS", "0"))
metrics = {{}}
results = {{}}
for dataset in datasets:
    outputs_path = runtime_dir / "outputs" / "res" / dataset / model_name / "linear_probing" / "frozen" / "outputs.json"
    best_ckpt_path = runtime_dir / "outputs" / "ckpts" / dataset / model_name / "frozen" / "best_model.pth"
    if not outputs_path.exists():
        raise FileNotFoundError(f"Thunder outputs for {{dataset}} were not written: {{outputs_path}}")
    if not best_ckpt_path.exists():
        raise FileNotFoundError(f"Thunder best checkpoint for {{dataset}} was not written: {{best_ckpt_path}}")
    entry = {{"outputs_path": str(outputs_path), "best_ckpt_path": str(best_ckpt_path), "exists": True}}
    test_output = json.loads(outputs_path.read_text())
    best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    if "val_metrics" not in best_ckpt:
        raise KeyError(f"Thunder best checkpoint for {{dataset}} did not include val_metrics")
    val_output = best_ckpt["val_metrics"]
    entry["test_metrics"] = test_output
    entry["val_metrics"] = val_output
    if "f1" in val_output and val_output["f1"] is not None:
        metrics[f"probe_{{dataset}}_val_f1"] = float(val_output["f1"]["metric_score"])
    else:
        raise KeyError(f"Thunder outputs for {{dataset}} did not include f1")
    if "balanced_accuracy" in val_output and val_output["balanced_accuracy"] is not None:
        metrics[f"probe_{{dataset}}_val_acc"] = float(val_output["balanced_accuracy"]["metric_score"])
    else:
        raise KeyError(f"Thunder outputs for {{dataset}} did not include balanced_accuracy")
    results[dataset] = entry
f1_keys = [f"probe_{{dataset}}_val_f1" for dataset in datasets if f"probe_{{dataset}}_val_f1" in metrics]
acc_keys = [f"probe_{{dataset}}_val_acc" for dataset in datasets if f"probe_{{dataset}}_val_acc" in metrics]
metrics["mean_probe_f1"] = sum(metrics[key] for key in f1_keys) / len(f1_keys)
metrics["mean_probe_acc"] = sum(metrics[key] for key in acc_keys) / len(acc_keys)
payload = {{
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "checkpoint_step": {request['checkpoint_step']},
    "train_step": {request['train_step']},
    "checkpoint_path": str(checkpoint_path),
    "model_name": model_name,
    "datasets": datasets,
    "status": "ok" if status_code == 0 else "failed",
    "status_code": status_code,
    "metrics": metrics,
    "results": results,
}}
result_path.write_text(json.dumps(payload, indent=2) + "\\n")
print(f"Wrote Thunder probe result to {{result_path}}")
PY
exit "$status"
"""
    Path(request["job_script"]).write_text(script)
    log_out = state["paths"]["slurm_dir"] / "nanopath-thunder-%j.out"
    log_err = state["paths"]["slurm_dir"] / "nanopath-thunder-%j.err"
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
            "--cpus-per-task",
            SLURM_CPUS_PER_TASK,
            "--time",
            SLURM_TIME_LIMIT,
            "--gres",
            SLURM_GRES,
            "--job-name",
            "nanopath-thunder",
            "--output",
            str(log_out),
            "--error",
            str(log_err),
            request["job_script"],
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    request["job_id"] = submit.stdout.strip()
    request["submitted_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    state["data"]["active"] = request
    state["data"]["queued"] = None
    write_probe_state(state)


def collect_probe_results(state, wandb_run, metrics_path, output_dir, best_val_mean_probe_f1, best_val_mean_probe_f1_step, best_probe_scores, log_step):
    if wandb_run is None:
        return best_val_mean_probe_f1, best_val_mean_probe_f1_step, best_probe_scores
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
            wandb_run.log({"val_thunder/step": result["train_step"], **{f"val_thunder/{key}": value for key, value in metrics.items()}}, step=log_step)
        if result["status"] == "ok":
            for key, value in metrics.items():
                best_probe_scores[key] = max(best_probe_scores.get(key, float("-inf")), float(value))
            current_mean = metrics["mean_probe_f1"]
            if float(current_mean) > best_val_mean_probe_f1:
                best_val_mean_probe_f1 = float(current_mean)
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
    write_probe_state(state)
    return best_val_mean_probe_f1, best_val_mean_probe_f1_step, best_probe_scores
