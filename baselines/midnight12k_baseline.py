# Run the full frozen-probe suite on the untouched Kaiko Midnight-12K checkpoint.
# checkpoint_path points at the HF repo directory containing model.safetensors.

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import torch
from safetensors.torch import load_file

from baselines.dinov2_small_baseline import run_frozen_baseline
from model import DinoV2ViT

MIDNIGHT12K_VITG14 = (1536, 40, 24, 37, "swiglu", True, None, 0)


class Midnight12KViT(DinoV2ViT):
    # Hugging Face DINOv2 interpolates this non-register checkpoint without antialiasing.
    pos_interpolation_antialias = False

    def probe_features(self, x):
        out = self(x)
        return torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(1)], dim=-1)


def load_probe_model(checkpoint_path, device):
    model = Midnight12KViT("midnight12k_vitg14", variant_cfg=MIDNIGHT12K_VITG14)
    raw = load_file(str(Path(checkpoint_path) / "model.safetensors"))
    state = {
        "cls_token": raw["embeddings.cls_token"],
        "register_tokens": model.register_tokens.detach().cpu().clone(),
        "pos_embed": raw["embeddings.position_embeddings"],
        "mask_token": raw["embeddings.mask_token"],
        "patch_embed.proj.weight": raw["embeddings.patch_embeddings.projection.weight"],
        "patch_embed.proj.bias": raw["embeddings.patch_embeddings.projection.bias"],
        "norm.weight": raw["layernorm.weight"],
        "norm.bias": raw["layernorm.bias"],
    }
    # HF Dinov2 stores q/k/v separately and has no register tokens; nanopath keeps qkv fused.
    for i in range(40):
        src, dst = f"encoder.layer.{i}", f"blocks.{i}"
        state[f"{dst}.attn.qkv.weight"] = torch.cat([raw[f"{src}.attention.attention.{x}.weight"] for x in ("query", "key", "value")])
        state[f"{dst}.attn.qkv.bias"] = torch.cat([raw[f"{src}.attention.attention.{x}.bias"] for x in ("query", "key", "value")])
        state[f"{dst}.attn.proj.weight"] = raw[f"{src}.attention.output.dense.weight"]
        state[f"{dst}.attn.proj.bias"] = raw[f"{src}.attention.output.dense.bias"]
        state[f"{dst}.ls1.gamma"] = raw[f"{src}.layer_scale1.lambda1"]
        state[f"{dst}.ls2.gamma"] = raw[f"{src}.layer_scale2.lambda1"]
        state[f"{dst}.norm1.weight"] = raw[f"{src}.norm1.weight"]
        state[f"{dst}.norm1.bias"] = raw[f"{src}.norm1.bias"]
        state[f"{dst}.norm2.weight"] = raw[f"{src}.norm2.weight"]
        state[f"{dst}.norm2.bias"] = raw[f"{src}.norm2.bias"]
        state[f"{dst}.mlp.w12.weight"] = raw[f"{src}.mlp.weights_in.weight"]
        state[f"{dst}.mlp.w12.bias"] = raw[f"{src}.mlp.weights_in.bias"]
        state[f"{dst}.mlp.w3.weight"] = raw[f"{src}.mlp.weights_out.weight"]
        state[f"{dst}.mlp.w3.bias"] = raw[f"{src}.mlp.weights_out.bias"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


if __name__ == "__main__":
    run_frozen_baseline(
        __file__, "baseline-midnight-12k", "midnight-12k-vitg14-untouched",
        "midnight12k_vitg14", "/data/Midnight-12K", "/data/$USER/nanopath/baselines/midnight-12k",
        1_136_480_768, "resize_crop_224", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
    )
