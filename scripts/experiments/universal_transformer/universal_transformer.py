#!/usr/bin/env python
"""
universal_transformer.py — Universal Multi-Task 1D-Patched Transformer

Simultaneously predicts:
  1. System Type (Binary Classification): 0 = AGNR, 1 = Square Lattice
  2. System Width/Size (3-Class Classification): 0 = 7-AGNR, 1 = 9-AGNR, 2 = Square-10
  3. Impurity Concentration (Scalar Regression): c in [2, 98]

Architecture:
  Input: [B, 150] (Normalized Conductance Spectrum)
    -> 1D ConvStem (1 -> 16 -> 32)
    -> 1D Patch Embedding (32 -> 128, patch_size=10, 15 tokens)
    -> Prepend [CLS] token (16 tokens) + Positional Embeddings
    -> Pre-Norm Transformer Encoder Blocks (Multi-Head Attention + DropPath)
    -> LayerNorm
    -> Type Head (128 -> 32 -> 2)
    -> Width Head (128 -> 32 -> 3)
    -> Conditioned Concentration Head ([CLS] + Softmax(Type) + Softmax(Width) -> 133 -> 64 -> 1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Per-sample stochastic depth (drops entire residual branches during training)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device))
        return x * mask / keep


class ConvStem(nn.Module):
    """Lightweight 1D ConvStem extracting local derivative and slope transitions."""
    def __init__(self, in_channels: int = 1, mid_channels: int = 16, out_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=7, stride=1, padding=3),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, L] -> [B, out_channels, L]"""
        return self.net(x)


class PatchEmbedding1D(nn.Module):
    """
    Non-overlapping 1D patching with learned positional encodings injected
    BEFORE LayerNorm to preserve absolute energy subband coordinates.
    """
    def __init__(
        self,
        seq_len: int = 150,
        patch_size: int = 10,
        in_channels: int = 32,
        embed_dim: int = 128,
        pos_embed_std: float = 0.10,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size  # 150 // 10 = 15
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

        # 15 patches + 1 [CLS] token = 16 tokens
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * pos_embed_std)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_channels, L] -> [B, 1 + num_patches, embed_dim]"""
        x = self.proj(x).transpose(1, 2)  # [B, num_patches, embed_dim]
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)    # [B, 1 + num_patches, embed_dim]
        # Injected before LayerNorm (BUILD-06 improvement)
        return self.norm(x + self.pos_embed)


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block with Multi-Head Attention and Stochastic Depth."""
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.05,
        drop_path: float = 0.05,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm Self-Attention
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.drop_path1(attn_out)

        # Pre-Norm FFN
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class UniversalPatchedTransformer(nn.Module):
    """
    Universal Multi-Task Patched Transformer for:
      - System Type (AGNR vs Square)
      - System Width (7 vs 9 vs 10)
      - Impurity Concentration (c)
    """
    def __init__(
        self,
        seq_len: int = 150,
        patch_size: int = 10,
        stem_channels: int = 32,
        embed_dim: int = 128,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.05,
        drop_path_rate: float = 0.05,
        pos_embed_std: float = 0.10,
        condition_heads: bool = True,
    ):
        super().__init__()
        self.condition_heads = condition_heads

        # 1. Stem & Patch Embedding
        self.stem = ConvStem(in_channels=1, mid_channels=16, out_channels=stem_channels)
        self.patch_embed = PatchEmbedding1D(
            seq_len=seq_len,
            patch_size=patch_size,
            in_channels=stem_channels,
            embed_dim=embed_dim,
            pos_embed_std=pos_embed_std,
        )

        # 2. Transformer Encoder Blocks
        dpr = [v.item() for v in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 3. Task Heads
        # 3a. System Type: 0 = AGNR, 1 = Square
        self.type_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

        # 3b. System Width: 0 = 7-AGNR, 1 = 9-AGNR, 2 = Square-10
        self.width_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
        )

        # 3c. Conditioned Concentration Head
        conc_in_dim = embed_dim + (2 + 3 if condition_heads else 0)
        self.conc_head = nn.Sequential(
            nn.Linear(conc_in_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        x: [B, L] or [B, 1, L]
        Returns:
          type_logits:  [B, 2]
          width_logits: [B, 3]
          conc_pred:    [B, 1] (normalized space)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, L]

        # 1. Stem + Patch Embedding
        tokens = self.patch_embed(self.stem(x))  # [B, 16, embed_dim]

        # 2. Transformer Blocks
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        # 3. [CLS] Feature Vector
        cls = tokens[:, 0]  # [B, embed_dim]

        # 4. Classification
        type_logits = self.type_head(cls)    # [B, 2]
        width_logits = self.width_head(cls)  # [B, 3]

        # 5. Conditioned Concentration Regression
        if self.condition_heads:
            p_type = F.softmax(type_logits, dim=1)
            p_width = F.softmax(width_logits, dim=1)
            cls_cond = torch.cat([cls, p_type, p_width], dim=1)  # [B, 128 + 2 + 3]
        else:
            cls_cond = cls

        conc_pred = self.conc_head(cls_cond)  # [B, 1]

        return type_logits, width_logits, conc_pred
