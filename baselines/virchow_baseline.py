# Run the full frozen-probe suite on the untouched Paige/Microsoft Virchow checkpoint.
# checkpoint_path is the HF/timm cache dir; pass checkpoint_path=/path off-cluster.

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import torch
import torch.nn as nn

from baselines.dinov2_small_baseline import run_frozen_baseline


class VirchowModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone, self.registers = backbone, 0

    def forward(self, x):
        # Virchow's model card defines model(x) as [cls, patch tokens].
        tokens = self.backbone(x)
        return {"cls": tokens[:, 0], "patches": tokens[:, 1:]}

    def encode_image(self, x):
        return self.forward(x)["patches"]

    def probe_features(self, x):
        out = self.forward(x)
        return torch.cat([out["cls"], out["patches"].mean(1)], dim=-1)


def load_probe_model(checkpoint_path, device):
    import timm
    from timm.layers import SwiGLUPacked

    model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, cache_dir=str(checkpoint_path), mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    return VirchowModel(model).to(device).eval()


if __name__ == "__main__":
    run_frozen_baseline(
        __file__, "baseline-virchow", "virchow-vith14-untouched",
        "virchow", "/data/Virchow", "/data/$USER/nanopath/baselines/virchow",
        632_000_000, "bicubic224_crop224", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
    )
