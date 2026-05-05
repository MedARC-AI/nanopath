# DinoV2ViT: clean ViT + 4 register tokens that loads Meta's `dinov2_vit{s,b}14_reg`
# pretrained weights via state_dict (no xformers, no dinov2 codebase imports).
# Attention runs on `F.scaled_dot_product_attention` so we get FlashAttention-2
# on H100 bf16 with no third-party kernel dependency. Module names below match
# Meta's checkpoint key layout exactly, so `load_dinov2_pretrained(model)` does
# a strict load.
#
# DINOHead is the small MLP + weight-normed classifier used by train.py for the
# DINO CLS / iBOT patch self-distillation losses. It is intentionally trivial
# (~15 lines) so we have zero runtime dependency on the dinov2 codebase.

import torch
import torch.nn as nn
import torch.nn.functional as F


# (dim, depth, heads, pretrain_grid, ffn, pos_has_cls, weight URL) for each supported variant.
DINOV2_VARIANTS = {
    "dinov2_vits14_reg": (384, 12, 6, 37, "mlp", True, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"),
    "dinov2_vitb14_reg": (768, 12, 12, 37, "mlp", True, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"),
    "openmidnight_vitg14_reg": (1536, 40, 24, 16, "swiglu", True, None),
    "hoptimus0_vitg14_reg": (1536, 40, 24, 16, "swiglu", False, None),
}


# Stochastic depth: keep_prob bernoulli on the residual branch, scaled to preserve mean.
class DropPath(nn.Module):
    def __init__(self, p): super().__init__(); self.p = float(p)
    def forward(self, x):
        if self.p == 0.0 or not self.training: return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep)
        return x * mask / keep


# Per-channel learnable scale on residual branches; matches Meta's `ls1.gamma`/`ls2.gamma`.
class LayerScale(nn.Module):
    def __init__(self, dim): super().__init__(); self.gamma = nn.Parameter(torch.ones(dim))
    def forward(self, x): return x * self.gamma


# Attention with single qkv Linear + F.scaled_dot_product_attention (Flash-2 backend on H100 bf16).
class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        hidden = (int(hidden * 2 / 3) + 7) // 8 * 8
        self.w12 = nn.Linear(dim, 2 * hidden, bias=True)
        self.w3 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


# Standard pre-LN block: attn + ls1 + drop_path, then mlp + ls2 + drop_path.
class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, drop_path_p, ffn="mlp"):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.ls1 = LayerScale(dim)
        self.drop_path1 = DropPath(drop_path_p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, hidden) if ffn == "swiglu" else nn.Sequential()
        if ffn == "mlp":
            self.mlp.fc1 = nn.Linear(dim, hidden, bias=True)
            self.mlp.fc2 = nn.Linear(hidden, dim, bias=True)
        self.ls2 = LayerScale(dim)
        self.drop_path2 = DropPath(drop_path_p)

    def _ff(self, x): return self.mlp(x) if isinstance(self.mlp, SwiGLU) else self.mlp.fc2(F.gelu(self.mlp.fc1(x)))

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self._ff(self.norm2(x))))
        return x


# ViT-S/B-14 with 4 register tokens; key layout matches Meta's DINOv2 register checkpoints
# (cls_token, register_tokens, pos_embed (1, 1+37^2, dim), mask_token (1, dim), patch_embed.proj,
# blocks.{i}.{norm1,norm2,attn.qkv,attn.proj,ls1,ls2,mlp.fc1,mlp.fc2}, norm).
# Pos embed is bicubically interpolated at runtime to the current patch grid.
# Meta DINOv2 includes a cls pos and uses 37x37 patches; ViT-G baselines use 16x16.
class DinoV2ViT(nn.Module):
    def __init__(self, variant="dinov2_vits14_reg", drop_path_rate=0.0):
        super().__init__()
        dim, depth, heads, pretrain_grid, ffn, pos_has_cls, _ = DINOV2_VARIANTS[variant]
        mlp_ratio, patch, registers = 4.0, 14, 4
        self.variant = variant
        self.patch_size, self.registers, self.embed_dim = patch, registers, dim
        self._pretrain_grid, self._pos_has_cls = pretrain_grid, pos_has_cls
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, registers, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, int(self._pos_has_cls) + self._pretrain_grid**2, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        rates = [drop_path_rate * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio, p, ffn=ffn) for p in rates)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    # Bicubic resample of the checkpoint patch-pos grid to the current (h, w) grid.
    def _interpolate_pos_embed(self, h, w):
        cls_pos = self.pos_embed[:, :1] if self._pos_has_cls else None
        g = self._pretrain_grid
        patch_pos = self.pos_embed[:, int(self._pos_has_cls):].reshape(1, g, g, -1).permute(0, 3, 1, 2).float()
        # antialias=True matches Meta's default for DINOv2 `_reg` variants.
        patch_pos = F.interpolate(patch_pos, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1).to(self.pos_embed.dtype)
        return torch.cat([cls_pos, patch_pos], dim=1) if cls_pos is not None else patch_pos

    # Build [cls, registers, patches] tokens; iBOT swaps the masked patch positions for mask_token.
    def _prepare_tokens(self, x, masks=None):
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).expand_as(x), x)
        cls = self.cls_token.expand(B, -1, -1)
        regs = self.register_tokens.expand(B, -1, -1)
        if self._pos_has_cls:
            x = torch.cat([cls, x], dim=1) + self._interpolate_pos_embed(h, w)
            return torch.cat([x[:, :1], regs, x[:, 1:]], dim=1)
        return torch.cat([cls, regs, x + self._interpolate_pos_embed(h, w)], dim=1)

    # Returns the dict shape Meta's `forward_features` returns; used by train.py and probe.py.
    # `checkpoint=True` re-runs each block under torch.utils.checkpoint to trade compute for memory;
    # useful when the 1-GPU batch of 128 (2 globals + 8 locals) does not fit in 80 GB.
    def forward(self, x, masks=None, checkpoint=False):
        x = self._prepare_tokens(x, masks)
        for blk in self.blocks:
            if checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.norm(x)
        return {
            "x_norm_clstoken": x[:, 0],
            "x_norm_regtokens": x[:, 1 : 1 + self.registers],
            "x_norm_patchtokens": x[:, 1 + self.registers :],
        }

    # Probe contract: encode_image returns [registers || patches] for the seg head;
    # probe_features returns the cls token for classification probes.
    def encode_image(self, x, checkpoint=False):
        out = self(x, checkpoint=checkpoint)
        return torch.cat([out["x_norm_regtokens"], out["x_norm_patchtokens"]], dim=1)

    def probe_features(self, x):
        return self(x)["x_norm_clstoken"]


# Strict-load Meta's pretrained weights for the model's declared variant.
# Strict matches our key layout against Meta's; any drift fails loudly per AGENTS.md.
def load_dinov2_pretrained(model):
    *_, url = DINOV2_VARIANTS[model.variant]
    state = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model


def load_openmidnight_checkpoint(model, path="/data/OpenMidnight_ckpts/openmidnight_checkpoint.pth"):
    raw = torch.load(path, map_location="cpu", weights_only=False)["teacher"]
    state = {}
    for key, value in raw.items():
        if "dino" in key or "ibot" in key:
            continue
        key = key.removeprefix("backbone.")
        if key.startswith("blocks."):
            _, _, block, rest = key.split(".", 3)
            key = f"blocks.{block}.{rest}"
        state[key] = value
    model.load_state_dict(state, strict=True)
    return model


def load_hoptimus0_checkpoint(model, path="/data/H-optimus-0/pytorch_model.bin"):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = {}
    for key, value in raw.items():
        key = key.replace("reg_token", "register_tokens").replace("mlp.fc1", "mlp.w12").replace("mlp.fc2", "mlp.w3")
        state[key] = value
    state["mask_token"] = model.mask_token.detach().cpu().clone()
    model.load_state_dict(state, strict=True)
    return model


# DINO/iBOT projection head: 3-layer MLP (in -> hidden -> hidden -> bottleneck) + L2 norm +
# weight-normed Linear(bottleneck -> n_prototypes) with weight_g frozen at 1, matching the
# behaviour of dinov2.layers.DINOHead. Standalone reimplementation (no xformers, no fvcore).
class DINOHead(nn.Module):
    def __init__(self, in_dim, n_prototypes, hidden_dim=2048, bottleneck_dim=384, nlayers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(nlayers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, n_prototypes, bias=False))
        # weight-norm under torch.nn.utils.parametrizations exposes `parametrizations.weight.original0/1`;
        # original0 is the magnitude vector (size n_prototypes). Freeze it at 1 to match dinov2's recipe.
        with torch.no_grad():
            self.last_layer.parametrizations.weight.original0.fill_(1.0)
        self.last_layer.parametrizations.weight.original0.requires_grad_(False)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)
