# Run the full frozen-probe suite on untouched Meta DINOv2-small/14-reg weights.
# Writes train.py-compatible metrics.jsonl + summary.json under output_dir.

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import torch
import yaml

from model import ViT, load_dinov2_pretrained
from probe import (
    TASK_FIELDS,
    completed_probe_summary,
    prepare_probe_state,
    queue_probe_job,
)


# Shared runner for baselines whose adapter exposes load_probe_model(checkpoint_path, device).
def run_frozen_baseline(script, project, recipe_id, model_type, checkpoint_default, output_default, activated_params, transform_policy, mean=None, std=None):
    script = Path(script)
    usage = f"usage: python baselines/{script.name} [config.yaml] [checkpoint_path=/path] [output_dir=/path]"
    config_path = REPO_DIR / "configs" / "main.yaml"
    checkpoint_path = Path(os.path.expandvars(checkpoint_default))
    output_dir = Path(os.path.expandvars(output_default))
    for arg in sys.argv[1:]:
        if arg.endswith((".yaml", ".yml")):
            config_path = Path(arg)
        else:
            key, _, value = arg.partition("=")
            if key == "checkpoint_path":
                checkpoint_path = Path(os.path.expandvars(value))
            elif key == "output_dir":
                output_dir = Path(os.path.expandvars(value))
            else:
                raise SystemExit(usage)

    cfg = yaml.safe_load(os.path.expandvars(config_path.read_text()))
    cfg["config_path"] = str(config_path.resolve())
    cfg["project"].update({"name": project, "family": "baseline", "recipe_id": recipe_id, "output_dir": str(output_dir)})
    cfg["model"]["type"] = model_type
    if mean is not None:
        cfg["data"]["mean"], cfg["data"]["std"] = mean, std
    cfg["probe"].update({
        "enabled": True,
        "model_weights": "ema",
        "count": 1,
        "model_loader": f"baselines.{script.stem}:load_probe_model",
        "transform_policy": transform_policy,
    })

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    started_at = time.monotonic()
    state = prepare_probe_state(cfg, output_dir)
    request = {
        "checkpoint_step": 0,
        "train_step": 0,
        "target_flops": 0,
        "target_fraction": 1.0,
        "checkpoint_path": str(checkpoint_path),
        "request_path": str(state["paths"]["probe_dir"] / "step_0000000.request.json"),
        "result_path": str(state["paths"]["results_dir"] / "step_0000000.json"),
        "job_id": f"{os.environ.get('SLURM_JOB_ID', 'local')}-{project.removeprefix('baseline-')}",
        "config": cfg,
        **{key: list(state["data"][key]) for key in TASK_FIELDS},
    }
    Path(request["request_path"]).write_text(json.dumps(request, indent=2) + "\n")
    env = os.environ.copy()
    env.pop("WANDB_SERVICE", None)
    env["PYTHONPATH"] = str(REPO_DIR)
    subprocess.run([sys.executable, str(REPO_DIR / "probe.py"), request["request_path"]], env=env, check=True)

    result = json.loads(Path(request["result_path"]).read_text())
    event = {"event": "probe", "step": 0, "target_flops": 0, "target_fraction": 1.0, "probe_wall_seconds": float(result["wall_seconds"]), **{key: float(value) for key, value in result["metrics"].items()}}
    (output_dir / "metrics.jsonl").write_text(json.dumps(event) + "\n")
    summary = {
        "project": project,
        "family": "baseline",
        "recipe_id": recipe_id,
        "config_path": cfg["config_path"],
        "checkpoint_path": str(checkpoint_path),
        "backbone_activated_params": activated_params,
        "steps_completed": 0,
        "train_flops": 0,
        "total_wall_seconds": time.monotonic() - started_at,
        **completed_probe_summary(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"baseline metrics: {output_dir / 'metrics.jsonl'}")
    print(f"final_score: {event['final_score']:.6f}")


def run_dinov2_baseline(script, project, recipe_id, variant, output_default, pretrained=True, seed=0):
    usage = f"usage: python baselines/{script} [config.yaml] [output_dir=/path]"
    config_path = REPO_DIR / "configs" / "main.yaml"
    output_dir = Path(os.path.expandvars(output_default))
    for arg in sys.argv[1:]:
        if arg.endswith((".yaml", ".yml")):
            config_path = Path(arg)
        else:
            key, _, value = arg.partition("=")
            if key != "output_dir":
                raise SystemExit(usage)
            output_dir = Path(os.path.expandvars(value))

    cfg = yaml.safe_load(os.path.expandvars(config_path.read_text()))
    cfg["config_path"] = str(config_path.resolve())
    cfg["project"]["name"] = project
    cfg["project"]["family"] = "baseline"
    cfg["project"]["recipe_id"] = recipe_id
    cfg["project"]["output_dir"] = str(output_dir)
    cfg["model"]["type"] = variant
    cfg["probe"]["enabled"] = True
    cfg["probe"]["model_weights"] = "ema"
    cfg["probe"]["count"] = 1
    cfg["probe"]["transform_policy"] = "resize_crop_224"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    started_at = time.monotonic()
    torch.manual_seed(seed)
    model = ViT(variant=variant)
    if pretrained:
        model = load_dinov2_pretrained(model)
    else:
        # The loader normally overwrites these token/pos parameters; initialize
        # them here so the random baseline is not partly zero by construction.
        for p in (model.cls_token, model.register_tokens, model.pos_embed, model.mask_token):
            torch.nn.init.trunc_normal_(p, std=0.02)
    weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    state = prepare_probe_state(cfg, output_dir)
    queue_probe_job(state, {"model": weights, "model_ema": weights, "step": 0, "config": cfg}, 0, 0, 1.0)

    result_path = state["paths"]["results_dir"] / "step_0000000.json"
    result = json.loads(result_path.read_text())
    event = {
        "event": "probe",
        "step": 0,
        "target_flops": 0,
        "target_fraction": 1.0,
        "probe_wall_seconds": float(result["wall_seconds"]),
        **{key: float(value) for key, value in result["metrics"].items()},
    }
    (output_dir / "metrics.jsonl").write_text(json.dumps(event) + "\n")
    summary = {
        "project": cfg["project"]["name"],
        "family": cfg["project"]["family"],
        "recipe_id": cfg["project"]["recipe_id"],
        "config_path": cfg["config_path"],
        "backbone_activated_params": sum(p.numel() for p in model.parameters()),
        "steps_completed": 0,
        "train_flops": 0,
        "total_wall_seconds": time.monotonic() - started_at,
        **completed_probe_summary(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    checkpoint_path = Path(result["checkpoint_path"])
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"baseline metrics: {output_dir / 'metrics.jsonl'}")
    print(f"final_score: {event['final_score']:.6f}")


def main():
    run_dinov2_baseline(
        "dinov2_small_baseline.py",
        "baseline-dinov2-small",
        "dinov2-vits14-reg-no-continued-pretraining",
        "dinov2_vits14_reg",
        "/data/$USER/nanopath/baselines/dinov2-small",
        pretrained=True,
    )


if __name__ == "__main__":
    main()
