"""Lattice-type classifier: a transmission spectrum -> which site types (C, H, K) the host has, one of 7 patterns.

The encoder is the inverse model's bandtop CNN (`make_encoder`, the band top added through its own Linear); a two-layer
head gives 7 logits. No 156-state head and no flow: the whole model is trained on the lattice type alone. `dropout`
(INV-PATTERN-2) drops units of the embedding and of the head's hidden layer in training only; it has no weights, so
checkpoints without it load unchanged.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tbribbon.parent_prior import PATTERNS

from .model import make_encoder, run_encoder
from .reward import pattern_name

N_PATTERNS = len(PATTERNS)
NAMES = tuple(pattern_name(p) for p in PATTERNS)
_INDEX = {p: i for i, p in enumerate(PATTERNS)}


def pattern_labels(on):
    """Pattern index (into PATTERNS / NAMES) of each row of a stored `on` array (int8 [N, 3], C/H/K)."""
    return np.array([_INDEX[tuple(bool(v) for v in r)] for r in np.asarray(on)], dtype=np.int64)


class PatternModel(nn.Module):
    """x: the bandtop input (parent_data.to_input_bandtop, 416 resampled values + the band top)."""

    def __init__(self, channels=417, embed=128, extra_conv=1, group_norm=True, pool="avg13", dropout=0.0):
        super().__init__()
        self.cfg = dict(channels=channels, embed=embed, extra_conv=extra_conv, group_norm=group_norm, pool=pool,
                        dropout=dropout)
        self.encoder = make_encoder(channels - 1, embed, group_norm=group_norm, extra_conv=extra_conv, encoder="cnn",
                                    pool=pool)
        self.band_top = nn.Linear(1, embed)
        self.head = nn.Sequential(nn.Linear(embed, 256), nn.GELU(), nn.Linear(256, N_PATTERNS))
        self.drop = nn.Dropout(dropout)

    def logits(self, x):
        z = run_encoder(self.encoder, "cnn", x[:, :-1]) + self.band_top(x[:, -1:])
        return self.head[2](self.drop(self.head[1](self.head[0](self.drop(z)))))

    def loss(self, x, y, label_smoothing=0.0):
        return F.cross_entropy(self.logits(x), y, label_smoothing=label_smoothing)

    @torch.no_grad()
    def predict_proba(self, x, batch=2000):
        self.eval()
        return torch.cat([F.softmax(self.logits(x[i:i + batch]), 1) for i in range(0, len(x), batch)])

    def save(self, path):
        torch.save({"cfg": self.cfg, "state_dict": self.state_dict()}, Path(path))

    @classmethod
    def load(cls, path):
        ck = torch.load(Path(path), map_location="cpu", weights_only=True)
        model = cls(**ck["cfg"])
        model.load_state_dict(ck["state_dict"])
        return model.eval()
