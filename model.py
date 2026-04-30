# DinoV2ViT: clean ViT-S/14 + 4 register tokens that loads Meta's
# `dinov2_vits14_reg` pretrained weights via state_dict (no xformers, no dinov2
# codebase imports). Attention runs on `F.scaled_dot_product_attention` so we
# get FlashAttention-2 on H100 bf16 with the same backend the prior LeJEPA
# model used. Module names below match Meta's checkpoint key layout exactly,
# so `load_dinov2_pretrained(model)` does a strict load.
#
# DINOHead is the small MLP + weight-normed classifier used by train.py for the
# DINO CLS / iBOT patch self-distillation losses. It is intentionally trivial
# (~15 lines) so we have zero runtime dependency on the dinov2 codebase.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


PRETRAIN_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"


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


# Standard pre-LN block: attn + ls1 + drop_path, then mlp + ls2 + drop_path.
class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, drop_path_p):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.ls1 = LayerScale(dim)
        self.drop_path1 = DropPath(drop_path_p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential()
        self.mlp.fc1 = nn.Linear(dim, hidden, bias=True)
        self.mlp.fc2 = nn.Linear(hidden, dim, bias=True)
        self.ls2 = LayerScale(dim)
        self.drop_path2 = DropPath(drop_path_p)

    def _ff(self, x): return self.mlp.fc2(F.gelu(self.mlp.fc1(x)))

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self._ff(self.norm2(x))))
        return x


# ViT-S/14 with 4 register tokens; key layout matches Meta's dinov2_vits14_reg checkpoint
# (cls_token, register_tokens, pos_embed (1, 1+37^2, dim), mask_token (1, dim), patch_embed.proj,
# blocks.{i}.{norm1,norm2,attn.qkv,attn.proj,ls1,ls2,mlp.fc1,mlp.fc2}, norm).
# Pos embed in the checkpoint is for a 37x37 patch grid (Meta's 518x518 pretraining); we bicubically
# interpolate at runtime to whatever (h,w) the current crop produces (16x16 for 224, 7x7 for 98).
class DinoV2ViT(nn.Module):
    def __init__(self, drop_path_rate=0.0):
        super().__init__()
        dim, heads, depth, mlp_ratio, patch, registers = 384, 6, 12, 4.0, 14, 4
        self.patch_size, self.registers, self.embed_dim = patch, registers, dim
        self._pretrain_grid = 37
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, registers, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self._pretrain_grid**2, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        rates = [drop_path_rate * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio, p) for p in rates)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    # Bicubic resample of the 37x37 patch pos grid to the current (h, w) grid.
    def _interpolate_pos_embed(self, h, w):
        cls_pos = self.pos_embed[:, :1]
        g = self._pretrain_grid
        patch_pos = self.pos_embed[:, 1:].reshape(1, g, g, -1).permute(0, 3, 1, 2).float()
        # antialias=True matches Meta's `dinov2_vits14_reg` factory (their default for `_reg` variants).
        patch_pos = F.interpolate(patch_pos, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1).to(cls_pos.dtype)
        return torch.cat([cls_pos, patch_pos], dim=1)

    # Build [cls, registers, patches] tokens; iBOT swaps the masked patch positions for mask_token.
    def _prepare_tokens(self, x, masks=None):
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).expand_as(x), x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self._interpolate_pos_embed(h, w)
        regs = self.register_tokens.expand(B, -1, -1)
        return torch.cat([x[:, :1], regs, x[:, 1:]], dim=1)

    # Returns the dict shape Meta's `forward_features` returns; used by train.py and probe.py.
    # `checkpoint=True` re-runs each block under torch.utils.checkpoint to trade compute for memory;
    # the 1-GPU recipe flips this on so a per-rank batch of 128 (2 globals + 8 locals) fits in 80 GB.
    def forward_features(self, x, masks=None, checkpoint=False):
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
        out = self.forward_features(x, checkpoint=checkpoint)
        return torch.cat([out["x_norm_regtokens"], out["x_norm_patchtokens"]], dim=1)

    def probe_features(self, x):
        return self.forward_features(x)["x_norm_clstoken"]


# Strict-load Meta's pretrained ViT-S/14-reg weights from the public URL.
# Strict matches our key layout against Meta's; any drift fails loudly per AGENTS.md.
def load_dinov2_pretrained(model):
    state = torch.hub.load_state_dict_from_url(PRETRAIN_URL, progress=False, map_location="cpu")
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
