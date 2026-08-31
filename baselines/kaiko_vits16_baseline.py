# Run the full frozen-probe suite on the untouched Kaiko pathology ViT-S/16.

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import timm
import torch
import torch.nn as nn
import yaml

from probe import TASK_FIELDS, completed_probe_summary, prepare_probe_state


class KaikoModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone, self.registers = backbone, 0

    def forward(self, x):
        tokens = self.backbone.forward_features(x)
        return {"x_norm_clstoken": tokens[:, 0], "x_norm_patchtokens": tokens[:, 1:]}

    def encode_image(self, x):
        return self.forward(x)["x_norm_patchtokens"]

    def probe_features(self, x):
        return self.backbone(x)


def load_probe_model(checkpoint_path, device):
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True), strict=True)
    return KaikoModel(model).to(device).eval()


def main():
    usage = "usage: python baselines/kaiko_vits16_baseline.py [config.yaml] [checkpoint_path=/path] [output_dir=/path]"
    config_path = REPO_DIR / "configs" / "main.yaml"
    checkpoint_path = Path("/data/Kaiko-ViTS16/vits16.pth")
    output_dir = Path("/data/paul/nanopath/probe-v2-study/expanded-models/kaiko-vits16")
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
    cfg["project"].update({
        "name": "baseline-kaiko-vits16",
        "family": "baseline",
        "recipe_id": "kaiko-pathology-vits16-untouched",
        "output_dir": str(output_dir),
    })
    cfg["data"]["mean"] = [0.5, 0.5, 0.5]
    cfg["data"]["std"] = [0.5, 0.5, 0.5]
    cfg["model"]["type"] = "kaiko_vits16"
    cfg["probe"].update({
        "enabled": True,
        "model_weights": "ema",
        "count": 1,
        "model_loader": "baselines.kaiko_vits16_baseline:load_probe_model",
        "transform_policy": "resize_crop_224",
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
        "job_id": f"{os.environ.get('SLURM_JOB_ID', 'local')}-kaiko-vits16",
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
        "project": cfg["project"]["name"],
        "family": cfg["project"]["family"],
        "recipe_id": cfg["project"]["recipe_id"],
        "config_path": cfg["config_path"],
        "checkpoint_path": str(checkpoint_path),
        "backbone_activated_params": 21_665_664,
        "steps_completed": 0,
        "train_flops": 0,
        "total_wall_seconds": time.monotonic() - started_at,
        **completed_probe_summary(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"mean_probe_score: {event['mean_probe_score']:.6f}")


if __name__ == "__main__":
    main()
