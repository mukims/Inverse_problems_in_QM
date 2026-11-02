"""Denoising spectrum autoencoder (INV-AE, plan docs/superpowers/plans/2026-10-29-mlp-encoder-and-autoencoder.md, Part 2).

Input: the band-top input (parent_data.to_input_bandtop: 416 resampled values + the band top). Target: the pristine
open-channel count on the same grid (parent_data.pristine_target_bandtop), so the 128-number latent must keep what
fixes the clean ribbon (host, edge, width) and may drop the disorder of the one impurity arrangement. The encoder is
built by model.make_encoder and applied exactly as InverseModel.encode with input="bandtop", so copy_encoder_into puts
it into an inverse model unchanged.
"""
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from shazam.inverse.model import make_encoder, run_encoder

CHANNELS = 416


class SpectrumAutoencoder(nn.Module):
    def __init__(self, encoder="cnn", extra_conv=0, mlp_hidden=(1024, 512), dropout=0.1, embed=128, group_norm=True):
        super().__init__()
        self.cfg = dict(encoder=encoder, extra_conv=extra_conv, mlp_hidden=list(mlp_hidden), dropout=dropout,
                        embed=embed, group_norm=group_norm)
        # cnn: the convs see the 416 values and the band top enters through its own Linear (as InverseModel.band_top);
        # mlp: all 417 values feed the first layer
        self.encoder = make_encoder(CHANNELS + (encoder == "mlp"), embed, group_norm, extra_conv, encoder, mlp_hidden,
                                    dropout)
        if encoder == "cnn":
            self.band_top = nn.Linear(1, embed)
        self.decoder = nn.Sequential(nn.Linear(embed, 512), nn.GELU(), nn.Linear(512, 1024), nn.GELU(),
                                     nn.Linear(1024, CHANNELS))

    def encode(self, x):
        if self.cfg["encoder"] == "cnn":
            return run_encoder(self.encoder, "cnn", x[:, :-1]) + self.band_top(x[:, -1:])
        return run_encoder(self.encoder, "mlp", x)

    def forward(self, x):
        return self.decoder(self.encode(x))

    def loss(self, x, target):
        """Mean squared error plus mean absolute error (a one-channel step costs the same at any channel count)."""
        out = self(x)
        return F.mse_loss(out, target) + F.l1_loss(out, target)

    def inverse_kwargs(self):
        """The InverseModel arguments whose encoder this autoencoder's encoder is."""
        c = self.cfg
        return dict(channels=CHANNELS + 1, input="bandtop", encoder=c["encoder"], extra_conv=c["extra_conv"],
                    mlp_hidden=tuple(c["mlp_hidden"]), dropout=c["dropout"], embed=c["embed"],
                    group_norm=c["group_norm"])

    def copy_encoder_into(self, model):
        """Load this encoder (and the cnn's band-top Linear) into an InverseModel built with inverse_kwargs()."""
        # only the settings that shape this encoder are compared (a cnn has no mlp widths or dropout; an mlp no GroupNorm)
        unused = {"cnn": ("mlp_hidden", "dropout"), "mlp": ("group_norm", "extra_conv")}[self.cfg["encoder"]]
        for key, want in self.inverse_kwargs().items():
            if key in unused:
                continue
            have = model.cfg.get(key)
            if (list(have) if isinstance(have, (list, tuple)) else have) != \
                    (list(want) if isinstance(want, (list, tuple)) else want):
                raise ValueError(f"inverse model has {key}={have!r}, the autoencoder's encoder needs {want!r}")
        model.encoder.load_state_dict(self.encoder.state_dict(), strict=True)
        if self.cfg["encoder"] == "cnn":
            model.band_top.load_state_dict(self.band_top.state_dict(), strict=True)

    def save(self, path):
        torch.save({"cfg": self.cfg, "state_dict": self.state_dict()}, Path(path))

    @classmethod
    def load(cls, path):
        ck = torch.load(Path(path), map_location="cpu", weights_only=True)
        model = cls(**ck["cfg"])
        model.load_state_dict(ck["state_dict"])
        return model.eval()
