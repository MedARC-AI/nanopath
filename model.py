# This file defines the model pieces used during pretraining.
# The backbone is a compact ViT with register tokens and RoPE so one encoder can serve
# JEPA view encoding and downstream probes without any auxiliary masked-prediction branch.
# The whole stack stays in native PyTorch so the code is easy to reason about, scale with
# DDP, and change after small-run ablations.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * x).to(x_dtype)


def apply_rotary_1d(x, positions, inv_freq):
    valid = (positions >= 0).to(dtype=x.dtype)[:, None, :, None]
    positions = positions.clamp_min(0).to(dtype=x.dtype)
    angles = positions[:, None, :, None] * inv_freq[None, None, None, :].to(dtype=x.dtype, device=x.device)
    cos = angles.cos()
    sin = angles.sin()
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1).flatten(-2)
    return rotated * valid + x * (1.0 - valid)


def apply_rotary_2d(x, positions, inv_freq):
    half = x.shape[-1] // 2
    return torch.cat([apply_rotary_1d(x[..., :half], positions[..., 0], inv_freq), apply_rotary_1d(x[..., half:], positions[..., 1], inv_freq)], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, heads, rope_base, qkv_bias, qk_norm):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        if self.head_dim % 4 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be divisible by 4 for 2D RoPE")
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        axis_dim = self.head_dim // 2
        self.register_buffer("inv_freq", 1.0 / (rope_base ** (torch.arange(0, axis_dim, 2).float() / axis_dim)), persistent=False)

    def forward(self, x, positions):
        b, n, _ = x.shape
        q = self.q(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv(x).view(b, n, 2, self.heads, self.head_dim)
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rotary_2d(q, positions, self.inv_freq)
        k = apply_rotary_2d(k, positions, self.inv_freq)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(x.transpose(1, 2).reshape(b, n, self.dim))


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, rope_base, qkv_bias, qk_norm):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads, rope_base, qkv_bias, qk_norm)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x, positions):
        x = x + self.attn(self.norm1(x), positions)
        return x + self.mlp(self.norm2(x))


class SIGReg(nn.Module):
    def __init__(self, knots=17, t_max=5.0, num_slices=1024):
        super().__init__()
        t = torch.linspace(0.0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-0.5 * t.square())
        self.num_slices = num_slices
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi", phi, persistent=False)
        self.register_buffer("weights", weights * phi, persistent=False)

    def forward(self, proj, generator=None):
        direction = torch.randn(proj.shape[-1], self.num_slices, device=proj.device, generator=generator)
        direction = direction / direction.norm(dim=0, keepdim=True)
        x_t = (proj @ direction).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(dim=-3) - self.phi).square() + x_t.sin().mean(dim=-3).square()
        statistic = (err @ self.weights) * proj.shape[-2]
        return statistic.mean()


class NanoPathFM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model = cfg["model"]
        self.patch_size = int(model["patch_size"])
        self.dim = int(model["dim"])
        self.registers = int(model["registers"])
        if self.registers < 1:
            raise ValueError("registers must be at least 1")
        if model.get("abs_pos", False):
            raise ValueError("abs_pos is not supported; NanoPath uses RoPE only")
        self.patch_embed = nn.Conv2d(3, self.dim, kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        self.register_tokens = nn.Parameter(torch.zeros(1, self.registers, self.dim))
        self.encoder = nn.ModuleList([Block(self.dim, model["heads"], model["mlp_ratio"], model["rope_base"], model["qkv_bias"], model["qk_norm"]) for _ in range(model["depth"])])
        self.encoder_norm = RMSNorm(self.dim)
        self.projector = nn.Sequential(
            RMSNorm(self.dim),
            nn.Linear(self.dim, model["projector_dim"], bias=True),
            nn.GELU(),
            nn.Linear(model["projector_dim"], model["projector_dim"], bias=True),
        )
        nn.init.trunc_normal_(self.register_tokens, std=0.02)

    def patch_positions(self, batch, h, w, device):
        gy, gx = h // self.patch_size, w // self.patch_size
        y, x = torch.meshgrid(torch.arange(gy, device=device), torch.arange(gx, device=device), indexing="ij")
        pos = torch.stack([y.reshape(-1), x.reshape(-1)], dim=-1).float()
        return pos.unsqueeze(0).expand(batch, -1, -1)

    def register_positions(self, batch, patch_positions):
        lo = patch_positions.min(dim=1, keepdim=True).values
        hi = patch_positions.max(dim=1, keepdim=True).values
        if self.registers == 1:
            base = torch.full((1, 2), 0.5, device=patch_positions.device, dtype=patch_positions.dtype)
        else:
            side = math.ceil(math.sqrt(self.registers))
            coords = torch.linspace(0.0, 1.0, side, device=patch_positions.device, dtype=patch_positions.dtype)
            y, x = torch.meshgrid(coords, coords, indexing="ij")
            base = torch.stack([y.reshape(-1), x.reshape(-1)], dim=-1)[: self.registers]
        return lo + base.unsqueeze(0) * (hi - lo)

    def patch_tokens(self, x):
        return self.patch_embed(x).flatten(2).transpose(1, 2)

    def encode_image(self, x, checkpoint=False):
        batch, _, h, w = x.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(f"image size {(h, w)} must be divisible by patch_size={self.patch_size}")
        tokens = self.patch_tokens(x)
        positions = self.patch_positions(batch, h, w, x.device)
        reg = self.register_tokens.expand(batch, -1, -1)
        reg_pos = self.register_positions(batch, positions)
        tokens = torch.cat([reg, tokens], dim=1)
        positions = torch.cat([reg_pos, positions], dim=1)
        for blk in self.encoder:
            if checkpoint and self.training:
                tokens = torch.utils.checkpoint.checkpoint(blk, tokens, positions, use_reentrant=False)
            else:
                tokens = blk(tokens, positions)
        return self.encoder_norm(tokens)

    def encode_views(self, global_views, local_views, checkpoint):
        batch = global_views.shape[0]
        global_tokens = self.encode_image(global_views.flatten(0, 1), checkpoint=checkpoint)
        local_tokens = self.encode_image(local_views.flatten(0, 1), checkpoint=checkpoint)
        global_reg = global_tokens[:, : self.registers].mean(dim=1).view(batch, global_views.shape[1], -1)
        local_reg = local_tokens[:, : self.registers].mean(dim=1).view(batch, local_views.shape[1], -1)
        reg = torch.cat([global_reg, local_reg], dim=1)
        return self.projector(reg)

    def probe_features(self, x):
        tokens = self.encode_image(x, checkpoint=False)
        return tokens[:, : self.registers].mean(dim=1)

    def forward(self, global_views, local_views, train_cfg):
        return self.encode_views(global_views, local_views, bool(train_cfg["activation_checkpointing"]))
