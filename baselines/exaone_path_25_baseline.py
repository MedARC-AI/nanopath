# Run the full frozen-probe suite on the untouched EXAONE-Path-2.5 patch encoder.
# Defaults to the MedARC cluster checkpoint path; pass checkpoint_path=/path off-cluster.
# The local repo at checkpoint_path is expected to ship LG AI Research's `exaonepath/`
# package so we can build the ViT-B/14 backbone without depending on transformers.

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import torch.nn as nn
from safetensors.torch import load_file

from baselines.dinov2_small_baseline import run_frozen_baseline


# Probe contract: `forward` returns DINOv2-style x_norm_* dict, `encode_image` and
# `probe_features` mirror DinoV2ViT. EXAONE has no register tokens.
class ExaonePathPatchEncoder(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit, self.registers = vit, 0

    def forward(self, x):
        # get_intermediate_layers(n=1) returns the final-LayerNorm [B, 1+256, 768] tokens, which
        # is exactly what EXAONE's own model(x) emits as the cls feature; use it raw like every baseline.
        seq = self.vit.get_intermediate_layers(x, n=1)[0]
        return {"x_norm_clstoken": seq[:, 0], "x_norm_patchtokens": seq[:, 1:]}

    def encode_image(self, x):
        return self(x)["x_norm_patchtokens"]

    def probe_features(self, x):
        return self(x)["x_norm_clstoken"]


def load_probe_model(checkpoint_path, device):
    # Build vit_base via the upstream `exaonepath/` package shipped in the repo dir, then
    # strict-load patch-encoder/model.safetensors (keys are `backbone.*` from the PatchEncoder wrapper).
    repo = Path(checkpoint_path)
    sys.path.insert(0, str(repo))
    from exaonepath.models.patch_transformer import vit_base
    backbone = vit_base(patch_size=14, img_size=[224])
    state = load_file(str(repo / "patch-encoder" / "model.safetensors"))
    backbone.load_state_dict({k.removeprefix("backbone."): v for k, v in state.items()}, strict=True)
    return ExaonePathPatchEncoder(backbone).to(device).eval()


if __name__ == "__main__":
    run_frozen_baseline(
        __file__, "baseline-exaone-path-2.5", "exaone-path-2.5-vitb14-untouched",
        "exaone_path_2.5", "/data/exaone_path_2.5", "/data/$USER/nanopath/baselines/exaone-path-2.5",
        85_706_496, "resize_crop_224", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
    )
