# NanoPathFM: the encoder used both for pretraining and for downstream probes.
# Compact ViT with register tokens and 2D RoPE (no abs-pos, no CLS), a small MLP 
# projector head, and SIGReg — LeJEPA's regularizer applied to the projected register 
# tokens to discourage representational collapse. encode_views() drives the JEPA 
# objective in train.py; probe_features() returns the same register pooling for probe.py.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Lightweight RMSNorm: one learned scale, fp32 normalization, bf16-friendly output.
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


# Rotate adjacent channel pairs for one RoPE axis using per-token grid positions.
def apply_rotary_1d(x, positions, inv_freq):
    positions = positions.to(dtype=x.dtype)
    angles = positions[:, None, :, None] * inv_freq[None, None, None, :].to(dtype=x.dtype, device=x.device)
    cos = angles.cos()
    sin = angles.sin()
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1).flatten(-2)


# Split each head into y/x halves so attention gets 2D relative position.
def apply_rotary_2d(x, positions, inv_freq):
    half = x.shape[-1] // 2
    y = apply_rotary_1d(x[..., :half], positions[..., 0], inv_freq)
    x = apply_rotary_1d(x[..., half:], positions[..., 1], inv_freq)
    return torch.cat([y, x], dim=-1)


# ViT attention over register + patch tokens, with RoPE applied to q/k only.
class Attention(nn.Module):
    def __init__(self, dim, heads, rope_base, qkv_bias, qk_norm):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim % 4 == 0
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


# Feed-forward branch inside each transformer block.
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


# Pre-norm transformer block with attention and MLP residual paths.
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


# SIGReg is the anti-collapse objective term used by train.py:loss_terms().
class SIGReg(nn.Module):
    # Sliced characteristic-function regularizer. The statistic is computed across
    # the rows of `proj`, so its gradient depends on the size of the batch passed in.
    # train.py all-gathers across ranks before calling this, which is why
    # `global_batch_size` is a recipe-level constant rather than a per-GPU knob.
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

    # Match projected features to a Gaussian characteristic function along random slices.
    def forward(self, proj, generator):
        direction = torch.randn(proj.shape[-1], self.num_slices, device=proj.device, generator=generator)
        direction = direction / direction.norm(dim=0, keepdim=True)
        x_t = (proj @ direction).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(dim=-3) - self.phi).square() + x_t.sin().mean(dim=-3).square()
        statistic = (err @ self.weights) * proj.shape[-2]
        return statistic.mean()


# Shared backbone/projector used by pretraining, classification probes, and seg probes.
class NanoPathFM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model = cfg["model"]
        self.patch_size = int(model["patch_size"])
        self.dim = int(model["dim"])
        self.registers = int(model["registers"])
        assert self.registers > 0
        self.patch_embed = nn.Conv2d(3, self.dim, kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        self.register_tokens = nn.Parameter(torch.zeros(1, self.registers, self.dim))
        block_args = (self.dim, model["heads"], model["mlp_ratio"], model["rope_base"], model["qkv_bias"], model["qk_norm"])
        self.encoder = nn.ModuleList([Block(*block_args) for _ in range(model["depth"])])
        self.encoder_norm = RMSNorm(self.dim)
        self.projector = nn.Sequential(
            RMSNorm(self.dim),
            nn.Linear(self.dim, model["projector_dim"], bias=True),
            nn.GELU(),
            nn.Linear(model["projector_dim"], model["projector_dim"], bias=True),
        )
        nn.init.trunc_normal_(self.register_tokens, std=0.02)

    # Encode one image batch into register + patch tokens; probe.py also consumes patch tokens directly.
    def encode_image(self, x, checkpoint=False):
        batch, _, h, w = x.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0
        gy, gx = h // self.patch_size, w // self.patch_size
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        y, x_pos = torch.meshgrid(torch.arange(gy, device=x.device), torch.arange(gx, device=x.device), indexing="ij")
        patch_pos = torch.stack([y.reshape(-1), x_pos.reshape(-1)], dim=-1).float()
        # Registers are prepended and later pooled; placing them on the grid lets RoPE give them spatial context.
        if self.registers == 1:
            reg_pos = patch_pos.new_tensor([[0.5 * (gy - 1), 0.5 * (gx - 1)]])
        else:
            side = math.ceil(math.sqrt(self.registers))
            coords = torch.linspace(0.0, 1.0, side, device=x.device)
            reg_y, reg_x = torch.meshgrid(coords, coords, indexing="ij")
            reg_pos = torch.stack([reg_y.reshape(-1) * (gy - 1), reg_x.reshape(-1) * (gx - 1)], dim=-1)[: self.registers]
        reg = self.register_tokens.expand(batch, -1, -1)
        tokens = torch.cat([reg, tokens], dim=1)
        positions = torch.cat([reg_pos.unsqueeze(0).expand(batch, -1, -1), patch_pos.unsqueeze(0).expand(batch, -1, -1)], dim=1)
        for blk in self.encoder:
            # Checkpointing is the only memory/compute tradeoff knob in the model path.
            if checkpoint and self.training:
                tokens = torch.utils.checkpoint.checkpoint(blk, tokens, positions, use_reentrant=False)
            else:
                tokens = blk(tokens, positions)
        return self.encoder_norm(tokens)

    # Pretraining path: encode global/local crops, pool registers, then project per view.
    def encode_views(self, global_views, local_views, checkpoint):
        batch = global_views.shape[0]
        global_tokens = self.encode_image(global_views.flatten(0, 1), checkpoint=checkpoint)
        local_tokens = self.encode_image(local_views.flatten(0, 1), checkpoint=checkpoint)
        global_reg = global_tokens[:, : self.registers].mean(dim=1).view(batch, global_views.shape[1], -1)
        local_reg = local_tokens[:, : self.registers].mean(dim=1).view(batch, local_views.shape[1], -1)
        reg = torch.cat([global_reg, local_reg], dim=1)
        return self.projector(reg)

    # Downstream classification probes use pooled backbone registers before the projector.
    def probe_features(self, x):
        tokens = self.encode_image(x, checkpoint=False)
        return tokens[:, : self.registers].mean(dim=1)

    # train.py passes train_cfg here so activation checkpointing stays a YAML recipe knob.
    def forward(self, global_views, local_views, train_cfg):
        return self.encode_views(global_views, local_views, bool(train_cfg["activation_checkpointing"]))

    # AdamW weight decay policy: no decay for biases, norms, and register tokens.
    def param_groups(self, weight_decay):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            no_decay_param = param.ndim < 2 or name.endswith("bias") or "norm" in name or "register_tokens" in name
            (no_decay if no_decay_param else decay).append(param)
        return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
