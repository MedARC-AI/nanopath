# Run the full frozen-probe suite on the untouched OpenMidnight ViT-G checkpoint.
# Defaults to the MedARC cluster checkpoint path; pass checkpoint_path=/path off-cluster.

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import torch

from baselines.dinov2_small_baseline import run_frozen_baseline
from model import ViT

OPENMIDNIGHT_VITG14_REG = (1536, 40, 24, 16, 14, "swiglu", True, None)


def load_probe_model(checkpoint_path, device):
    model = ViT(variant_cfg=OPENMIDNIGHT_VITG14_REG)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = raw["teacher"] if "teacher" in raw else raw
    state = {}
    for key, value in raw.items():
        if "dino" in key or "ibot" in key:
            continue
        key = key.removeprefix("backbone.")
        if key.startswith("blocks."):
            parts = key.split(".", 3)
            if parts[2].isdigit():
                key = f"blocks.{parts[2]}.{parts[3]}"
        state[key] = value
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


if __name__ == "__main__":
    run_frozen_baseline(
        __file__, "baseline-openmidnight", "openmidnight-vitg14-reg-untouched",
        "openmidnight_vitg14_reg", "/data/OpenMidnight_ckpts/openmidnight_checkpoint.pth",
        "/data/$USER/nanopath/baselines/openmidnight", 1_134_777_344, "square_224",
    )
