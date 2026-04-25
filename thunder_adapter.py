# Vendored Thunder pieces used by the inline pannuke segmentation probe.
# Sourced verbatim (with drop_path/dropout/einops dependencies inlined) from
# thunder/src/thunder/models/task_specific_models.py and thunder/src/thunder/utils/dice_loss.py.

import torch
import torch.nn as nn
import torch.nn.functional as F


def multiclass_dice_loss(pred, label, mask, smooth=1.0):
    pred = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    target = label.clone()
    target[~mask] = num_classes
    target = F.one_hot(target, num_classes=num_classes + 1)[..., :-1].permute(0, 3, 1, 2)
    mask = mask.unsqueeze(1)
    intersection = (pred * target * mask).sum(dim=(0, 2, 3))
    union = (pred * mask).sum(dim=(0, 2, 3)) + (target * mask).sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()


class _Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(b, n, c))


class _FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, heads)
        self.mlp = _FeedForward(dim, mlp_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class MaskTransformer(nn.Module):
    def __init__(self, n_cls, d_encoder, n_layers=2, n_heads=8, d_model=768, d_ff=3072):
        super().__init__()
        self.n_cls = n_cls
        self.scale = d_model ** -0.5
        self.proj_dec = nn.Linear(d_encoder, d_model)
        self.blocks = nn.ModuleList([_Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)
        self.apply(_init_weights)
        nn.init.trunc_normal_(self.cls_emb, std=0.02)

    def forward(self, x):
        b, n, _ = x.shape
        gs = int(n ** 0.5)
        x = self.proj_dec(x)
        x = torch.cat([x, self.cls_emb.expand(b, -1, -1)], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes
        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)
        masks = self.mask_norm(patches @ cls_seg_feat.transpose(1, 2))
        return masks.reshape(b, gs, gs, self.n_cls).permute(0, 3, 1, 2)
