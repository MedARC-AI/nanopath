# Run the full frozen-probe suite on the untouched Bioptimus H0-mini checkpoint.
# Defaults to the canonical MedARC snapshot; pass checkpoint_path=/path off-cluster.
# Validated at Hugging Face revision 5b5cc0505d19ae558270045eb0df8c34df4d9609.

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import torch
import torch.nn as nn
from safetensors.torch import load_file

from baselines.dinov2_small_baseline import run_frozen_baseline


class H0MiniModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        # encode_image() already removes the class and four register tokens.
        self.backbone, self.registers = backbone, 0

    def forward(self, x):
        tokens = self.backbone(x)
        return {
            "x_norm_clstoken": tokens[:, 0],
            "x_norm_patchtokens": tokens[:, self.backbone.num_prefix_tokens :],
        }

    def encode_image(self, x):
        return self.forward(x)["x_norm_patchtokens"]

    def probe_features(self, x):
        # Match H0-mini's official THUNDER frozen representation.
        out = self.forward(x)
        return torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(1)], dim=-1)


def load_probe_model(checkpoint_path, device):
    import timm
    from timm.layers import SwiGLUPacked

    model = timm.create_model(
        "vit_base_patch14_reg4_dinov2", pretrained=False, mlp_layer=SwiGLUPacked,
        act_layer=nn.SiLU, img_size=224, init_values=1e-5, num_classes=0,
        reg_tokens=4, mlp_ratio=5.33334, global_pool="", dynamic_img_size=True,
    )
    model.load_state_dict(load_file(str(Path(checkpoint_path) / "model.safetensors")), strict=True)
    return H0MiniModel(model).to(device).eval()


if __name__ == "__main__":
    run_frozen_baseline(
        __file__, "baseline-h0-mini", "h0-mini-vitb14-reg-distilled-untouched",
        "h0mini_vitb14_reg", "/data/H0-mini", "/data/$USER/nanopath/baselines/h0-mini",
        85_739_520, "bicubic224_crop224", [0.707223, 0.578729, 0.703617], [0.211883, 0.230117, 0.177517],
    )
