# Run the full frozen-probe suite on the untouched Kaiko pathology ViT-S/16.

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import timm
import torch
import torch.nn as nn

from baselines.dinov2_small_baseline import run_frozen_baseline


class KaikoModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone, self.registers = backbone, 0

    def forward(self, x):
        tokens = self.backbone.forward_features(x)
        return {"cls": tokens[:, 0], "patches": tokens[:, 1:]}

    def encode_image(self, x):
        return self.forward(x)["patches"]

    def probe_features(self, x):
        return self.backbone(x)


def load_probe_model(checkpoint_path, device):
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True), strict=True)
    return KaikoModel(model).to(device).eval()


if __name__ == "__main__":
    run_frozen_baseline(
        __file__, "baseline-kaiko-vits16", "kaiko-pathology-vits16-untouched",
        "kaiko_vits16", "/data/Kaiko-ViTS16/vits16.pth", "/data/$USER/nanopath/baselines/kaiko-vits16",
        21_665_664, "resize_crop_224", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
    )
