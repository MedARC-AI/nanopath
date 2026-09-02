# Run the full frozen-probe suite on the untouched H-optimus-0 ViT-G checkpoint.
# Defaults to the MedARC cluster checkpoint path; pass checkpoint_path=/path off-cluster.

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import torch

from baselines.dinov2_small_baseline import run_frozen_baseline
from model import ViT

HOPTIMUS0_VITG14_REG = (1536, 40, 24, 16, "swiglu", False, None)


def load_probe_model(checkpoint_path, device):
    model = ViT("hoptimus0_vitg14_reg", variant_cfg=HOPTIMUS0_VITG14_REG)
    state = {}
    for key, value in torch.load(checkpoint_path, map_location="cpu", weights_only=False).items():
        key = key.replace("reg_token", "register_tokens").replace("mlp.fc1", "mlp.w12").replace("mlp.fc2", "mlp.w3")
        state[key] = value
    state["mask_token"] = model.mask_token.detach().cpu().clone()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


if __name__ == "__main__":
    run_frozen_baseline(
        __file__, "baseline-hoptimus0", "hoptimus0-vitg14-reg-untouched",
        "hoptimus0_vitg14_reg", "/data/H-optimus-0/pytorch_model.bin", "/data/$USER/nanopath/baselines/hoptimus0",
        1_134_775_808, "bicubic256_crop224", [0.707223, 0.578729, 0.703617], [0.211883, 0.230117, 0.177517],
    )
