# Run the full frozen-probe suite on the untouched MahmoodLab UNI2-h checkpoint.
# checkpoint_path points at the HF repo directory containing pytorch_model.bin.

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import torch

from baselines.dinov2_small_baseline import run_frozen_baseline
from model import DinoV2ViT

UNI2H_VITH14 = (1536, 24, 24, 16, "swiglu", False, None, 8)


def load_probe_model(checkpoint_path, device):
    model = DinoV2ViT("uni2h_vith14", variant_cfg=UNI2H_VITH14)
    raw = torch.load(Path(checkpoint_path) / "pytorch_model.bin", map_location="cpu", weights_only=True)
    state = {}
    for key, value in raw.items():
        key = key.replace("reg_token", "register_tokens").replace("mlp.fc1", "mlp.w12").replace("mlp.fc2", "mlp.w3")
        state[key] = value
    state["mask_token"] = model.mask_token.detach().cpu().clone()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


if __name__ == "__main__":
    run_frozen_baseline(
        __file__, "baseline-uni2-h", "uni2-h-vith14-untouched",
        "uni2h_vith14", "/data/UNI2-h", "/data/$USER/nanopath/baselines/uni2-h",
        681_395_712, "resize_crop_224",
    )
