# Run the full frozen-probe suite on the EXAONE-Path-2.5 patch encoder.
# Local weights at /data/exaone_path_2.5 (download via snapshot_download if missing).

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
import torch.nn as nn
import torch.nn.functional as F
import yaml

from probe import TASK_FIELDS, completed_probe_summary, prepare_probe_state

CHECKPOINT_DIR = Path("/data/exaone_path_2.5")


def load_probe_model(checkpoint_path, device):
    # Load EXAONE-Path-2.5 patch encoder (ViT-B/14, 768-d) via transformers.
    # Use local weights to avoid repeated HF requests.
    from transformers import AutoModel

    local_dir = checkpoint_path if checkpoint_path else str(CHECKPOINT_DIR)
    model = AutoModel.from_pretrained(
        local_dir,
        component="patch",
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()

    # The wrapper forward() returns CLS-only. The raw ViT backbone has
    # get_intermediate_layers which returns the full [B, 257, 768] sequence.
    vit = model.patch_encoder.backbone

    class _ExaonePathWrapper(nn.Module):
        def __init__(self, v):
            super().__init__()
            self.vit = v
            self.registers = 0  # EXAONE has no register tokens

        def forward(self, x):
            seq = self.vit.get_intermediate_layers(x, n=1)[0]
            # DINOv2 returns L2-normalized tokens. EXAONE returns raw outputs, so normalize
            # here to match the probe interface and keep features bounded (needed for
            # Coxnet and cosine-similarity probes).
            cls = seq[:, 0]
            patches = seq[:, 1:]
            cls = F.normalize(cls, dim=-1)
            patches = F.normalize(patches, dim=-1)
            return {
                "x_norm_clstoken": cls,
                "x_norm_patchtokens": patches,
            }

        def encode_image(self, x):
            return self(x)["x_norm_patchtokens"]

        def probe_features(self, x):
            return self(x)["x_norm_clstoken"]

    return _ExaonePathWrapper(vit).to(device).eval()


def main():
    config_path = REPO_DIR / "configs" / "main.yaml"
    output_dir = Path(os.path.expandvars("/data/$USER/nanopath/baselines/exaone_path_2.5"))
    for arg in sys.argv[1:]:
        if arg.endswith((".yaml", ".yml")):
            config_path = Path(arg)
        else:
            key, _, value = arg.partition("=")
            if key == "output_dir":
                output_dir = Path(os.path.expandvars(value))
            else:
                raise SystemExit(f"usage: python baselines/exaone_path_2.5_baseline.py [config.yaml] [output_dir=/path]")

    cfg = yaml.safe_load(os.path.expandvars(config_path.read_text()))
    cfg["config_path"] = str(config_path.resolve())
    cfg["project"]["name"] = "baseline-exaone-path-2.5"
    cfg["project"]["family"] = "baseline"
    cfg["project"]["recipe_id"] = "exaone-path-2.5-vitb14-patch-encoder-untouched"
    cfg["project"]["output_dir"] = str(output_dir)
    cfg["data"]["mean"] = [0.485, 0.456, 0.406]
    cfg["data"]["std"] = [0.229, 0.224, 0.225]
    cfg["model"]["type"] = "exaone_path_2.5"
    cfg["probe"]["enabled"] = True
    cfg["probe"]["model_weights"] = "ema"
    cfg["probe"]["count"] = 1
    cfg["probe"]["model_loader"] = "baselines.exaone_path_25_baseline:load_probe_model"
    cfg["probe"]["parallel_segmentation"] = False

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
        "checkpoint_path": str(CHECKPOINT_DIR),
        "request_path": str(state["paths"]["probe_dir"] / "step_0000000.request.json"),
        "result_path": str(state["paths"]["results_dir"] / "step_0000000.json"),
        "job_id": f"{os.environ.get('SLURM_JOB_ID', 'local')}-exaone-path-2.5",
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
        "checkpoint_path": str(CHECKPOINT_DIR),
        "backbone_activated_params": 86_156_544,  # ViT-B/14
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
