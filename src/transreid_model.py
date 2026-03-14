"""
transreid_model.py — Inference-only TransReID ViT-B/16 backbone.

Minimal model definition extracted from the official TransReID repository
(MIT licensed, https://github.com/damo-cv/TransReID) with all training-only
modules removed:

  - SIE  (Side Information Embedding) — injects MSMT17 camera bias → removed
  - JPM  (Jigsaw Patch Module) — training-only shuffle augmentation → removed
  - Classification heads (classifier, bottleneck) → removed

The resulting model is a vanilla ViT-B/16 with overlapping patch embedding
(stride 12), producing a 768-d CLS token descriptor per input crop.

Checkpoint compatibility
------------------------
The official TransReID* (ViT) MSMT17 checkpoint stores backbone weights under
the ``base.*`` prefix.  Call :func:`load_transreid_weights` to strip that
prefix and filter out SIE/JPM/classifier keys before loading.
"""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path) — identity at eval time."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device).add_(keep).floor_()
        return x.div(keep) * mask


class PatchEmbed_overlap(nn.Module):
    """Image → patch tokens via Conv2d with overlapping stride."""
    def __init__(self, img_size: tuple = (256, 128), patch_size: int = 16,
                 stride_size: int = 12, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.num_y = (img_size[0] - patch_size) // stride_size + 1  # 21
        self.num_x = (img_size[1] - patch_size) // stride_size + 1  # 10
        self.num_patches = self.num_y * self.num_x                  # 210
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=stride_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) → (B, num_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# TransReID backbone (inference-only, no SIE / JPM / classifier)
# ─────────────────────────────────────────────────────────────────────────────

class TransReIDBackbone(nn.Module):
    """
    Vanilla ViT-B/16 backbone with overlapping patch embedding.

    Forward pass:
        image → PatchEmbed → prepend CLS → add pos_embed
              → 12× TransformerBlock → LayerNorm → CLS token (768-d)
    """
    def __init__(
        self,
        img_size: tuple = (256, 128),
        patch_size: int = 16,
        stride_size: int = 12,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Patch embedding (overlapping stride)
        self.patch_embed = PatchEmbed_overlap(
            img_size=img_size, patch_size=patch_size,
            stride_size=stride_size, in_chans=in_chans, embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches  # 210

        # Learnable tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer blocks with stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)                                    # (B, 210, 768)
        cls_tokens = self.cls_token.expand(B, -1, -1)              # (B, 1, 768)
        x = torch.cat((cls_tokens, x), dim=1)                     # (B, 211, 768)
        x = x + self.pos_embed                                     # (B, 211, 768)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]                                              # (B, 768)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def make_transreid_model() -> TransReIDBackbone:
    """Create a TransReID inference backbone — vanilla ViT-B/16, no SIE/JPM."""
    return TransReIDBackbone(
        img_size=(256, 128),
        patch_size=16,
        stride_size=12,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    )


_STRIP_PREFIXES = ("b1.", "b2.", "bottleneck", "classifier", "sie_embed")


def load_transreid_weights(model: TransReIDBackbone, checkpoint_path: str) -> None:
    """
    Load a TransReID checkpoint, stripping SIE/JPM/classifier keys and the
    ``base.`` prefix so it maps onto :class:`TransReIDBackbone`.
    """
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]

    cleaned: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        # Skip SIE, JPM branches, bottlenecks, classifiers, ImageNet fc head
        if any(k.startswith(p) for p in _STRIP_PREFIXES):
            continue
        if k.startswith("base.fc.") or k == "base.sie_embed":
            continue
        # Strip the `base.` prefix used by build_transformer_local
        if k.startswith("base."):
            k = k[len("base."):]
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if unexpected:
        import logging
        logging.getLogger(__name__).warning(
            "TransReID load: %d unexpected keys ignored: %s",
            len(unexpected), unexpected[:5],
        )
