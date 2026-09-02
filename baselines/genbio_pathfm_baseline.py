# Run the full frozen-probe suite on the untouched GenBio-PathFM ViT-G checkpoint.
# Defaults to the MedARC cluster checkpoint path; pass checkpoint_path=/path off-cluster.

import json
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import torch
import torch.nn as nn

from baselines.dinov2_small_baseline import run_frozen_baseline


def load_probe_model(checkpoint_path, device):
    import importlib.util
    import types
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
            return {"cls": torch.cat([cls.unflatten(0, (b, 3))[:, i] for i in range(3)], dim=-1), "patches": torch.cat([patch.unflatten(0, (b, 3))[:, i] for i in range(3)], dim=-1)}
        def encode_image(self, x): return self._stack(x, patches=True)
        def probe_features(self, x): return self._stack(x)
    return _GenBioPathFM(backbone).to(device).eval()


if __name__ == "__main__":
    # Stats follow GenBio's config; its adapter internally splits normalized RGB into three channels.
    run_frozen_baseline(
        __file__, "baseline-genbio-pathfm", "genbio-pathfm-vitg16-rope-untouched",
        "genbio_pathfm", "/data/genbio-pathfm", "/data/$USER/nanopath/baselines/genbio_pathfm",
        1_133_686_784, "resize_crop_224", [0.697, 0.575, 0.728], [0.188, 0.240, 0.187],
    )
