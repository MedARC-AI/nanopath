# Run the full frozen-probe suite on the untouched Prov-GigaPath-Flash tile encoder.

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import torch
import torch.nn as nn

from baselines.dinov2_small_baseline import run_frozen_baseline


class GigaPathFlashModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        tokens = self.backbone.forward_features(x)
        return {"cls": tokens[:, 0], "patches": tokens[:, 1:]}

    def encode_image(self, x):
        return self.forward(x)["patches"]

    def probe_features(self, x):
        return self.backbone(x)


def load_probe_model(checkpoint_path, device):
    from timm.layers import SwiGLUPacked
    from timm.models.vision_transformer import VisionTransformer

    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=2048 / 384.0, mlp_layer=SwiGLUPacked, act_layer=nn.SiLU,
        init_values=1e-5, num_classes=0, global_pool="token", class_token=True,
        reg_tokens=0,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True), strict=True)
    return GigaPathFlashModel(model).to(device).eval()


if __name__ == "__main__":
    run_frozen_baseline(
        __file__, "baseline-gigapath-flash", "prov-gigapath-flash-tile-encoder-untouched",
        "gigapath_flash", "/data/GigaPath-Flash/pytorch_model.bin", "/data/$USER/nanopath/baselines/gigapath-flash",
        21_681_024, "bicubic256_crop224", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
    )
