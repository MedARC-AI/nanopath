# Run the full frozen-probe suite on the untouched Prov-GigaPath tile encoder.
# checkpoint_path is the HF/timm cache dir; pass checkpoint_path=/path off-cluster.

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import torch.nn as nn

from baselines.dinov2_small_baseline import run_frozen_baseline


class GigaPathModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        tokens = self.backbone.forward_features(x)
        return {"cls": tokens[:, 0], "patches": tokens[:, 1:]}

    def encode_image(self, x):
        return self.forward(x)["patches"]

    def probe_features(self, x):
        # GigaPath exposes tile_encoder(x) as the deployed 1536-d tile embedding.
        return self.backbone(x)


def load_probe_model(checkpoint_path, device):
    import timm

    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, cache_dir=str(checkpoint_path))
    return GigaPathModel(model).to(device).eval()


if __name__ == "__main__":
    run_frozen_baseline(
        __file__, "baseline-gigapath", "prov-gigapath-tile-encoder-untouched",
        "gigapath", "/data/GigaPath", "/data/$USER/nanopath/baselines/gigapath",
        1_134_953_984, "bicubic224_crop224", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
    )
