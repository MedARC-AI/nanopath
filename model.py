# ViT is a compact register-token vision transformer whose module names match
# DINOv2 checkpoints; the current variants can load Meta's pretrained weights.
# Attention runs on `F.scaled_dot_product_attention` so we get FlashAttention-2
# on H100 bf16 with no third-party kernel dependency. Module names below match
# Meta's checkpoint key layout exactly, so `load_pretrained(model)` does
# a strict load.
#
# DINOHead is the small MLP + weight-normed classifier used by train.py for the
# DINO CLS / iBOT patch self-distillation losses. It is intentionally trivial
# (~15 lines) so we have zero runtime dependency on the dinov2 codebase.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# (dim, depth, heads, pretrain_grid, patch, ffn, pos_has_cls, weight URL[, registers]) per variant.
VIT_VARIANTS = {
    "dinov2_vits14_reg": (384, 12, 6, 37, 14, "mlp", True, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"),
    "dinov2_vitb14_reg": (768, 12, 12, 37, 14, "mlp", True, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"),
    "dinov2_vitl14_reg": (1024, 24, 16, 37, 14, "mlp", True, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"),
    "dinov2_vitg14_reg": (1536, 40, 24, 37, 14, "swiglu", True, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth"),
}
VIT_VARIANTS["robust_norm_dinov2_vits14_reg"] = VIT_VARIANTS["dinov2_vits14_reg"]


def probe_transforms():
    # Default for Nanopath-trained checkpoints; baseline scripts override this in their request config.
    transform = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor()])
    # Keep the two return slots because probe.py separates tile-image and slide/patch-bag probes.
    return transform, transform


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


# Identity forward whose signed scale steers FINO gradients at the backbone boundary.
class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale): ctx.scale = scale; return x
    @staticmethod
    def backward(ctx, grad): return grad * ctx.scale, None


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


# Width, depth, patch/grid size, and register count are configurable; the default key layout matches
# Meta's DINOv2 register checkpoints
# (cls_token, register_tokens, pos_embed (1, 1+37^2, dim), mask_token (1, dim), patch_embed.proj,
# blocks.{i}.{norm1,norm2,attn.qkv,attn.proj,ls1,ls2,mlp.fc1,mlp.fc2}, norm).
# Pos embed is bicubically interpolated at runtime to the current patch grid.
# Meta DINOv2 includes a cls pos and uses 37x37 patches; variant_cfg can override this for other ViTs.
class ViT(nn.Module):
    pos_interpolation_antialias = True

    def __init__(self, variant="dinov2_vits14_reg", drop_path_rate=0.0, variant_cfg=None):
        super().__init__()
        cfg = variant_cfg or VIT_VARIANTS[variant]
        dim, depth, heads, pretrain_grid, patch, ffn, pos_has_cls, self.pretrained_url = cfg[:8]
        mlp_ratio, registers = 4.0, cfg[8] if len(cfg) > 8 else 4
        self.robust_norm = variant == "robust_norm_dinov2_vits14_reg" and variant_cfg is None
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
        # Keep external baseline state dicts unchanged; ordinary Nanopath models checkpoint these statistics.
        if self.robust_norm:
            self.register_buffer("rn_fitted", torch.zeros((), dtype=torch.bool))
            self.register_buffer("rn_mu", torch.zeros(2, dim))
            self.register_buffer("rn_v", torch.zeros(2, 32, dim))
            self.register_buffer("pf_fitted", torch.zeros((), dtype=torch.bool))
            self.register_buffer("pf_mu", torch.zeros(5, dim))
            self.register_buffer("pf_v", torch.zeros(5, 1, dim))
        else:
            self.rn_fitted = self.pf_fitted = False

    # Bicubic resample of the checkpoint patch-pos grid to the current (h, w) grid.
    def _interpolate_pos_embed(self, h, w):
        cls_pos = self.pos_embed[:, :1] if self._pos_has_cls else None
        g = self._pretrain_grid
        patch_pos = self.pos_embed[:, int(self._pos_has_cls):].reshape(1, g, g, -1).permute(0, 3, 1, 2).float()
        # antialias=True matches Meta's default for DINOv2 `_reg` variants.
        patch_pos = F.interpolate(
            patch_pos,
            size=(h, w),
            mode="bicubic",
            align_corners=False,
            antialias=self.pos_interpolation_antialias,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1).to(self.pos_embed.dtype)
        return torch.cat([cls_pos, patch_pos], dim=1) if cls_pos is not None else patch_pos

    # Build [cls, registers, patches]; masked objectives swap selected patches for mask_token.
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

    # Remove fitted scanner-response directions while retaining the original feature mean.
    def _suppress(self, x, mean, directions):
        centered = x.float() - mean
        return (centered - (centered @ directions.T) @ directions + mean).to(x.dtype)

    # Return semantic token groups used by train.py and probe.py.
    # `checkpoint=True` re-runs each block under torch.utils.checkpoint to trade compute for memory;
    # useful when a configured 1-GPU batch does not fit in 80 GB.
    def forward(self, x, masks=None, checkpoint=False):
        x = self._prepare_tokens(x, masks)
        for blk in self.blocks:
            if checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.norm(x)
        cls, patches = x[:, 0], x[:, 1 + self.registers :]
        if not self.training and self.rn_fitted:
            cls = self._suppress(cls, self.rn_mu[0], self.rn_v[0])
            patch_mean = patches.mean(1)
            patches = patches + (self._suppress(patch_mean, self.rn_mu[1], self.rn_v[1]) - patch_mean).unsqueeze(1)
        return {
            "cls": cls,
            "registers": x[:, 1 : 1 + self.registers],
            "patches": patches,
        }

    # Robust-norm's segmentation readout fuses the last four blocks, edge-guides a 32x32
    # spatial grid, and leaves v2 probe.py to pool that grid to the native decoder size.
    def encode_image(self, x, checkpoint=False):
        if not self.robust_norm:
            return self(x, checkpoint=checkpoint)["patches"]
        batch, _, height, width = x.shape
        h, w, grid = height // self.patch_size, width // self.patch_size, 32
        guide = x.mean(1, keepdim=True)
        guide = (guide - guide.amin((2, 3), keepdim=True)) / (guide.amax((2, 3), keepdim=True) - guide.amin((2, 3), keepdim=True) + 1e-6)
        tokens, features = self._prepare_tokens(x), []
        for i, block in enumerate(self.blocks):
            tokens = torch.utils.checkpoint.checkpoint(block, tokens, use_reentrant=False) if checkpoint and self.training else block(tokens)
            if i >= len(self.blocks) - 4:
                features.append(self.norm(tokens)[:, 1 + self.registers :])
        patches = torch.cat(features, -1)
        up = F.interpolate(patches.transpose(1, 2).reshape(batch, patches.shape[-1], h, w).float(), size=(grid, grid), mode="bilinear", align_corners=False)
        guide_lr = F.interpolate(guide, size=(h, w), mode="area")
        guide_hr = F.interpolate(guide, size=(grid, grid), mode="area")
        edge_weight = torch.exp(-((guide_hr - F.interpolate(guide_lr, size=(grid, grid), mode="nearest")).abs() ** 2) / 0.02)
        blur = F.avg_pool2d(F.pad(up, (1, 1, 1, 1), mode="replicate"), 3, 1)
        return (up + (1 - edge_weight) * (up - blur)).flatten(2).transpose(1, 2).to(patches.dtype)

    # Pooled probes use five strided-depth CLS taps, each with its fitted rank-one suppression.
    def probe_features(self, x):
        if not self.robust_norm:
            return self(x)["cls"]
        tokens, features = self._prepare_tokens(x), []
        for i, block in enumerate(self.blocks):
            tokens = block(tokens)
            if i in (2, 4, 6, 8, 11):
                feature = self.norm(tokens)[:, 0]
                j = len(features)
                features.append(self._suppress(feature, self.pf_mu[j], self.pf_v[j]) if not self.training and self.pf_fitted else feature)
        return torch.cat(features, dim=-1)


# Strict-load the model's declared pretrained weights; incompatible layouts fail loudly.
def load_pretrained(model):
    state = torch.hub.load_state_dict_from_url(model.pretrained_url, progress=False, map_location="cpu")
    if model.robust_norm:
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and all(key.startswith(("rn_", "pf_")) for key in missing), (missing, unexpected)
    else:
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


# I-JEPA predicts EMA-teacher patch features from the student's block-masked tokens.
class JEPAPredictor(nn.Module):
    def __init__(self, dim, depth=4, width=0, heads=6):
        super().__init__()
        width = width or dim
        self.proj_in = nn.Linear(dim, width) if width != dim else nn.Identity()
        self.blocks = nn.ModuleList(Block(width, heads, 4.0, 0.0) for _ in range(depth))
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.proj = nn.Linear(width, dim, bias=True)

    def forward(self, patch_tokens):
        x = self.proj_in(patch_tokens)
        for block in self.blocks:
            x = block(x)
        return self.proj(self.norm(x))
