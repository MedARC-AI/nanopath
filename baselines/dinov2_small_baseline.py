# Run the full frozen-probe suite on untouched Meta DINOv2-S/14-reg weights.
# Writes train.py-compatible metrics.jsonl + summary.json under output_dir.

import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import torch
import yaml

from model import DinoV2ViT, load_dinov2_pretrained
from probe import completed_probe_summary, prepare_probe_state, queue_probe_job


def main():
    usage = "usage: python baselines/dinov2_small_baseline.py [config.yaml] [output_dir=/path]"
    config_path = REPO_DIR / "configs" / "leader.yaml"
    output_dir = Path(os.path.expandvars("/data/$USER/nanopath/baselines/dinov2-small"))
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
    cfg["project"]["name"] = "baseline-dinov2-small"
    cfg["project"]["family"] = "baseline"
    cfg["project"]["recipe_id"] = "dinov2-vits14-reg-no-continued-pretraining"
    cfg["project"]["output_dir"] = str(output_dir)
    cfg["model"]["type"] = "dinov2_vits14_reg"
    cfg["probe"]["enabled"] = True
    cfg["probe"]["model_weights"] = "ema"
    cfg["probe"]["count"] = 1

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    started_at = time.monotonic()
    model = load_dinov2_pretrained(DinoV2ViT(variant="dinov2_vits14_reg"))
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
    print(f"mean_probe_score: {event['mean_probe_score']:.6f}")


if __name__ == "__main__":
    main()
