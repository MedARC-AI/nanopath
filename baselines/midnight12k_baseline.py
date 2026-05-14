# Run the full frozen-probe suite on the untouched Kaiko Midnight-12K checkpoint.
# checkpoint_path points at the HF repo directory containing model.safetensors.

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
from safetensors.torch import load_file

from model import DinoV2ViT
from probe import TASK_FIELDS, completed_probe_summary, prepare_probe_state

MIDNIGHT12K_VITG14 = (1536, 40, 24, 37, "swiglu", True, None, 0)


class Midnight12KViT(DinoV2ViT):
    def probe_features(self, x):
        out = self(x)
        return torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(1)], dim=-1)


def load_probe_model(checkpoint_path, device):
    model = Midnight12KViT("midnight12k_vitg14", variant_cfg=MIDNIGHT12K_VITG14)
    raw = load_file(str(Path(checkpoint_path) / "model.safetensors"))
    state = {
        "cls_token": raw["embeddings.cls_token"],
        "register_tokens": model.register_tokens.detach().cpu().clone(),
        "pos_embed": raw["embeddings.position_embeddings"],
        "mask_token": raw["embeddings.mask_token"],
        "patch_embed.proj.weight": raw["embeddings.patch_embeddings.projection.weight"],
        "patch_embed.proj.bias": raw["embeddings.patch_embeddings.projection.bias"],
        "norm.weight": raw["layernorm.weight"],
        "norm.bias": raw["layernorm.bias"],
    }
    # HF Dinov2 stores q/k/v separately and has no register tokens; nanopath keeps qkv fused.
    for i in range(40):
        src, dst = f"encoder.layer.{i}", f"blocks.{i}"
        state[f"{dst}.attn.qkv.weight"] = torch.cat([raw[f"{src}.attention.attention.{x}.weight"] for x in ("query", "key", "value")])
        state[f"{dst}.attn.qkv.bias"] = torch.cat([raw[f"{src}.attention.attention.{x}.bias"] for x in ("query", "key", "value")])
        state[f"{dst}.attn.proj.weight"] = raw[f"{src}.attention.output.dense.weight"]
        state[f"{dst}.attn.proj.bias"] = raw[f"{src}.attention.output.dense.bias"]
        state[f"{dst}.ls1.gamma"] = raw[f"{src}.layer_scale1.lambda1"]
        state[f"{dst}.ls2.gamma"] = raw[f"{src}.layer_scale2.lambda1"]
        state[f"{dst}.norm1.weight"] = raw[f"{src}.norm1.weight"]
        state[f"{dst}.norm1.bias"] = raw[f"{src}.norm1.bias"]
        state[f"{dst}.norm2.weight"] = raw[f"{src}.norm2.weight"]
        state[f"{dst}.norm2.bias"] = raw[f"{src}.norm2.bias"]
        state[f"{dst}.mlp.w12.weight"] = raw[f"{src}.mlp.weights_in.weight"]
        state[f"{dst}.mlp.w12.bias"] = raw[f"{src}.mlp.weights_in.bias"]
        state[f"{dst}.mlp.w3.weight"] = raw[f"{src}.mlp.weights_out.weight"]
        state[f"{dst}.mlp.w3.bias"] = raw[f"{src}.mlp.weights_out.bias"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    usage = "usage: python baselines/midnight12k_baseline.py [config.yaml] [checkpoint_path=/path] [output_dir=/path]"
    config_path = REPO_DIR / "configs" / "leader.yaml"
    checkpoint_path = Path("/data/Midnight-12K")
    output_dir = Path(os.path.expandvars("/data/$USER/nanopath/baselines/midnight-12k"))
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
    cfg["project"]["name"] = "baseline-midnight-12k"
    cfg["project"]["family"] = "baseline"
    cfg["project"]["recipe_id"] = "midnight-12k-vitg14-untouched"
    cfg["project"]["output_dir"] = str(output_dir)
    cfg["data"]["mean"] = [0.5, 0.5, 0.5]
    cfg["data"]["std"] = [0.5, 0.5, 0.5]
    cfg["model"]["type"] = "midnight12k_vitg14"
    cfg["probe"]["enabled"] = True
    cfg["probe"]["model_weights"] = "ema"
    cfg["probe"]["count"] = 1
    cfg["probe"]["model_loader"] = "baselines.midnight12k_baseline:load_probe_model"
    cfg["probe"]["transform_policy"] = "resize_crop_224"

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
        "job_id": f"{os.environ.get('SLURM_JOB_ID', 'local')}-midnight-12k",
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
        "backbone_activated_params": 1_136_480_768,
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
