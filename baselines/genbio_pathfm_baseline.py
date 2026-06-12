# Run the full frozen-probe suite on the untouched GenBio-PathFM ViT-G checkpoint.
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
import torch.nn as nn
import yaml

from probe import TASK_FIELDS, completed_probe_summary, prepare_probe_state


def load_probe_model(checkpoint_path, device):
    import importlib.util, types
    path = str(Path(checkpoint_path))
    if "transformers" not in sys.modules:
        tf = types.ModuleType("transformers")
        class _Pre(nn.Module):
            def __init__(self, config=None): super().__init__(); self.config = config
            def post_init(self): pass
        tf.PreTrainedModel = _Pre
        tf.PretrainedConfig = type("PretrainedConfig", (), {"__init__": lambda self, **k: None})
        sys.modules["transformers"] = tf
    if "_genbio" not in sys.modules:
        # Synthetic package so GenBio's relative imports resolve without installing transformers.
        pkg = types.ModuleType("_genbio"); pkg.__path__ = [path]; sys.modules["_genbio"] = pkg
        for n in ("configuration_genbio_pathfm", "modeling_genbio_pathfm"):
            spec = importlib.util.spec_from_file_location(f"_genbio.{n}", str(Path(path, f"{n}.py")))
            mod = importlib.util.module_from_spec(spec); sys.modules[f"_genbio.{n}"] = mod
            spec.loader.exec_module(mod)
    VisionTransformer = sys.modules["_genbio.modeling_genbio_pathfm"].VisionTransformer
    backbone = VisionTransformer(**json.loads(Path(path, "config.json").read_text()))
    backbone.load_state_dict(torch.load(str(Path(path, "model.pth")), map_location="cpu", weights_only=False), strict=True)
    class _GenBioPathFM(nn.Module):
        def __init__(self, b): super().__init__(); self.backbone, self.registers = b, 0
        def _encode(self, x):
            tokens, (h, w) = self.backbone.prepare_tokens(x)
            rope = self.backbone.rope_embed(H=h, W=w)
            for blk in self.backbone.blocks:
                tokens = blk(tokens, rope)
            tokens = self.backbone.norm(tokens)
            return tokens[:, 0], tokens[:, 1 + self.backbone.n_storage_tokens:]
        def _stack(self, x, patches=False):
            b, _, h, w = x.shape
            cls, patch = self._encode(x.reshape(b * 3, 1, h, w))
            out = (patch if patches else cls).unflatten(0, (b, 3))
            return torch.cat([out[:, 0], out[:, 1], out[:, 2]], dim=-1)
        def forward(self, x):
            b, _, h, w = x.shape
            cls, patch = self._encode(x.reshape(b * 3, 1, h, w))
            return {"x_norm_clstoken": torch.cat([cls.unflatten(0, (b, 3))[:, i] for i in range(3)], dim=-1), "x_norm_patchtokens": torch.cat([patch.unflatten(0, (b, 3))[:, i] for i in range(3)], dim=-1)}
        def encode_image(self, x): return self._stack(x, patches=True)
        def probe_features(self, x): return self._stack(x)
    return _GenBioPathFM(backbone).to(device).eval()


def main():
    usage = "usage: python baselines/genbio_pathfm_baseline.py [config.yaml] [checkpoint_path=/path] [output_dir=/path]"
    config_path = REPO_DIR / "configs" / "main.yaml"
    # checkpoint_path points at the HF repo directory: load_genbio_pathfm reads
    # config.json + modeling_genbio_pathfm.py + model.pth from it.
    checkpoint_path = Path("/data/genbio-pathfm")
    output_dir = Path(os.path.expandvars("/data/$USER/nanopath/baselines/genbio_pathfm"))
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
    cfg["project"]["name"] = "baseline-genbio-pathfm"
    cfg["project"]["family"] = "baseline"
    cfg["project"]["recipe_id"] = "genbio-pathfm-vitg16-rope-untouched"
    cfg["project"]["output_dir"] = str(output_dir)
    # Per-channel ImageNet-style stats from /data/genbio-pathfm/config.json (image_mean/image_std);
    # GenBio normalizes RGB then internally splits channels into 3 single-channel inputs.
    cfg["data"]["mean"] = [0.697, 0.575, 0.728]
    cfg["data"]["std"] = [0.188, 0.240, 0.187]
    cfg["model"]["type"] = "genbio_pathfm"
    cfg["probe"]["enabled"] = True
    cfg["probe"]["model_weights"] = "ema"
    cfg["probe"]["count"] = 1
    cfg["probe"]["model_loader"] = "baselines.genbio_pathfm_baseline:load_probe_model"
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
        "job_id": f"{os.environ.get('SLURM_JOB_ID', 'local')}-genbio-pathfm",
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
        "backbone_activated_params": 1_133_686_784,
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
