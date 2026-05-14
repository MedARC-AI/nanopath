# Run the full frozen-probe suite on the untouched OpenMidnight ViT-G checkpoint.
# Defaults to the MedARC cluster checkpoint path; pass checkpoint_path=/path off-cluster.

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

from model import DinoV2ViT
from probe import TASK_FIELDS, completed_probe_summary, prepare_probe_state

OPENMIDNIGHT_VITG14_REG = (1536, 40, 24, 16, "swiglu", True, None)


def load_probe_model(checkpoint_path, device):
    model = DinoV2ViT("openmidnight_vitg14_reg", variant_cfg=OPENMIDNIGHT_VITG14_REG)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = raw["teacher"] if "teacher" in raw else raw
    state = {}
    for key, value in raw.items():
        if "dino" in key or "ibot" in key:
            continue
        key = key.removeprefix("backbone.")
        if key.startswith("blocks."):
            parts = key.split(".", 3)
            if parts[2].isdigit():
                key = f"blocks.{parts[2]}.{parts[3]}"
        state[key] = value
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    usage = "usage: python baselines/openmidnight_baseline.py [config.yaml] [checkpoint_path=/path] [output_dir=/path]"
    config_path = REPO_DIR / "configs" / "leader.yaml"
    checkpoint_path = Path("/data/OpenMidnight_ckpts/openmidnight_checkpoint.pth")
    output_dir = Path(os.path.expandvars("/data/$USER/nanopath/baselines/openmidnight"))
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
    print(f"checkpoint_path={checkpoint_path} (override with checkpoint_path=/path if not using MedARC defaults)", flush=True)

    cfg = yaml.safe_load(os.path.expandvars(config_path.read_text()))
    cfg["config_path"] = str(config_path.resolve())
    cfg["project"]["name"] = "baseline-openmidnight"
    cfg["project"]["family"] = "baseline"
    cfg["project"]["recipe_id"] = "openmidnight-vitg14-reg-untouched"
    cfg["project"]["output_dir"] = str(output_dir)
    cfg["model"]["type"] = "openmidnight_vitg14_reg"
    cfg["probe"]["enabled"] = True
    cfg["probe"]["model_weights"] = "ema"
    cfg["probe"]["count"] = 1
    cfg["probe"]["model_loader"] = "baselines.openmidnight_baseline:load_probe_model"
    cfg["probe"]["transform_policy"] = "square_224"

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
        "job_id": f"{os.environ.get('SLURM_JOB_ID', 'local')}-openmidnight",
        "config": cfg,
        **{key: list(state["data"][key]) for key in TASK_FIELDS},
    }
    Path(request["request_path"]).write_text(json.dumps(request, indent=2) + "\n")
    env = os.environ.copy()
    env.pop("WANDB_SERVICE", None)
    env["PYTHONPATH"] = str(REPO_DIR)
    subprocess.run([sys.executable, str(REPO_DIR / "probe.py"), request["request_path"]], env=env, check=True)

    result = json.loads(Path(request["result_path"]).read_text())
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
        "checkpoint_path": str(checkpoint_path),
        "backbone_activated_params": 1_134_777_344,
        "steps_completed": 0,
        "train_flops": 0,
        "total_wall_seconds": time.monotonic() - started_at,
        **completed_probe_summary(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"baseline metrics: {output_dir / 'metrics.jsonl'}")
    print(f"mean_probe_score: {event['mean_probe_score']:.6f}")


if __name__ == "__main__":
    main()
