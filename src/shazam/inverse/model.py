"""Inverse model: a 416-channel transmission spectrum -> p(discrete state, 14 continuous slots) (spec section 6).

A 1D-convolutional encoder summarises the spectrum; a classifier head gives the discrete state; a conditional
flow-matching velocity field gives the unbounded continuous values of the slots the state uses. Slots a state does not
use are held at exactly 0, in training (their noise, target and loss are masked) and in sampling.

Masks: the model needs the per-state slot mask to sample. Pass `state_masks` (bool [n_states, n_cont], e.g. built from
`encoding.active_slots`) to fix the table at construction; otherwise the table is recorded from the masks seen in
`loss` (a mask is a deterministic function of the state, so the record is exact) and `sample` only draws states whose
mask is known.
"""
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

_T_FREQS = torch.tensor([1.0, 2.0, 4.0, 8.0]) * torch.pi              # Fourier features of the flow time


ENCODERS = ("cnn", "mlp")
POOLS = ("avg13", "gap", "spectral")


class SpectralPool1d(nn.Module):
    """Downsample along the last axis by keeping the lowest Fourier frequencies (INV-POOL): rfft, keep the
    out_len // 2 + 1 lowest coefficients, inverse transform onto out_len points. Keeps the mean and smooth shape with
    no block edges (average pooling averages fixed blocks of 4 instead)."""

    def __init__(self, out_len):
        super().__init__()
        self.out_len = out_len

    def forward(self, x):
        X = torch.fft.rfft(x, dim=-1, norm="forward")
        return torch.fft.irfft(X[..., :self.out_len // 2 + 1], n=self.out_len, dim=-1, norm="forward")


def make_encoder(channels=416, embed=128, group_norm=True, extra_conv=0, encoder="cnn", mlp_hidden=(512, 256),
                 dropout=0.0, pool="avg13"):
    """The spectrum encoder, [batch, channels] -> [batch, embed] through `run_encoder` (shared by InverseModel and the
    autoencoder, so a pretrained encoder's state dict loads into either).

    cnn: three stride-2 convs (32, 64, 128 filters), each with GroupNorm (unless group_norm=False) and GELU, then
    `extra_conv` 128 -> 128 width-3 stride-1 blocks at the same 52 positions, then the pool and a Linear: avg13
    averages blocks of 4 positions into 13 buckets, gap averages all positions (no position kept), spectral keeps the 7
    lowest frequencies on 13 points (SpectralPool1d).
    mlp: Linear -> LayerNorm -> GELU (-> Dropout) for each width in mlp_hidden, then a Linear to embed; every energy
    feeds every unit from the first layer (no locality, no weight sharing across energies)."""
    if encoder not in ENCODERS:
        raise ValueError(f"unknown encoder {encoder!r}; use one of {ENCODERS}")
    if pool not in POOLS:
        raise ValueError(f"unknown pool {pool!r}; use one of {POOLS}")
    if encoder == "mlp":
        if extra_conv:
            raise ValueError("extra_conv adds conv blocks, which the mlp encoder does not have")
        layers, width = [], channels
        for h in mlp_hidden:
            layers += [nn.Linear(width, h), nn.LayerNorm(h), nn.GELU()] + ([nn.Dropout(dropout)] if dropout else [])
            width = h
        return nn.Sequential(*layers, nn.Linear(width, embed))
    # group_norm=False drops the per-spectrum GroupNorm after each conv, which rescales each group of feature rows over
    # all energies and so may remove their overall level (plateau height, i.e. the channel count)
    norm = (lambda c: [nn.GroupNorm(8, c)]) if group_norm else (lambda c: [])
    extra = [m for _ in range(extra_conv) for m in (nn.Conv1d(128, 128, 3, 1, 1), *norm(128), nn.GELU())]
    return nn.Sequential(
        nn.Conv1d(1, 32, 7, 2, 3), *norm(32), nn.GELU(),
        nn.Conv1d(32, 64, 5, 2, 2), *norm(64), nn.GELU(),
        nn.Conv1d(64, 128, 3, 2, 1), *norm(128), nn.GELU(),
        *extra,
        {"avg13": nn.AdaptiveAvgPool1d(13), "gap": nn.AdaptiveAvgPool1d(1), "spectral": SpectralPool1d(13)}[pool],
        nn.Flatten(), nn.Linear(128 * (1 if pool == "gap" else 13), embed))


def run_encoder(module, kind, x):
    """Apply an encoder built by make_encoder(encoder=kind) to x of shape [batch, channels]."""
    return module(x if kind == "mlp" else x.unsqueeze(1))


class InverseModel(nn.Module):
    def __init__(self, n_states, n_cont=14, channels=416, embed=128, state_masks=None, state_embed=32, hidden=256,
                 group_norm=True, input="log", extra_conv=0, encoder="cnn", mlp_hidden=(512, 256), dropout=0.0,
                 pool="avg13"):
        super().__init__()
        self.cfg = dict(n_states=n_states, n_cont=n_cont, channels=channels, embed=embed, state_embed=state_embed,
                        hidden=hidden, group_norm=group_norm, input=input, extra_conv=extra_conv, encoder=encoder,
                        mlp_hidden=list(mlp_hidden), dropout=dropout, pool=pool, masks_fixed=state_masks is not None)
        self.n_states, self.n_cont = n_states, n_cont
        # `input` only records which spectrum transform the model was trained on ("log": parent_data.to_input,
        # "linear": parent_data.to_input_linear); callers must feed the same one
        # a CNN on the bandtop input (parent_data.to_input_bandtop: 416 resampled values + the band top) convolves only
        # the spectrum and adds the band top to the fingerprint through its own Linear; an MLP reads all values directly
        self._split_top = input == "bandtop" and encoder == "cnn"
        self.encoder = make_encoder(channels - self._split_top, embed, group_norm, extra_conv, encoder, mlp_hidden,
                                    dropout, pool)
        if self._split_top:
            self.band_top = nn.Linear(1, embed)
        self.discrete = nn.Sequential(nn.Linear(embed, 256), nn.GELU(), nn.Linear(256, n_states))
        self.state_emb = nn.Embedding(n_states, state_embed)
        d_in = n_cont + 2 * len(_T_FREQS) + 1 + embed + state_embed + n_cont                  # z_t, t, h, state embedding, mask
        self.velocity = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                      nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_cont))
        if state_masks is None:
            masks, known = torch.zeros(n_states, n_cont, dtype=torch.bool), torch.zeros(n_states, dtype=torch.bool)
        else:
            masks = torch.as_tensor(state_masks, dtype=torch.bool)
            if masks.shape != (n_states, n_cont):
                raise ValueError(f"state_masks must have shape {(n_states, n_cont)}, got {tuple(masks.shape)}")
            known = torch.ones(n_states, dtype=torch.bool)
        self.register_buffer("state_masks", masks.clone())
        self.register_buffer("mask_known", known)

    def encode(self, x):
        if self._split_top:
            return run_encoder(self.encoder, "cnn", x[:, :-1]) + self.band_top(x[:, -1:])
        return run_encoder(self.encoder, self.cfg["encoder"], x)

    def _velocity(self, zt, t, h, state, mask):
        m = mask.to(zt.dtype)
        tf = t * _T_FREQS.to(t.dtype).to(t.device)
        inp = torch.cat([zt * m, t, tf.sin(), tf.cos(), h, self.state_emb(state), m], 1)
        return self.velocity(inp) * m

    def loss(self, x, state, z, mask):
        """Cross-entropy of the state plus flow-matching MSE on the active slots; returns (loss, parts)."""
        if self.cfg["masks_fixed"]:
            if not torch.equal(self.state_masks[state], mask.to(torch.bool)):
                raise ValueError("mask differs from the model's fixed state_masks for the given states")
        else:
            with torch.no_grad():
                self.state_masks[state] = mask
                self.mask_known[state] = True
        h = self.encode(x)
        ce = F.cross_entropy(self.discrete(h), state)
        m = mask.to(z.dtype)
        z0 = torch.randn_like(z) * m
        t = torch.rand(z.shape[0], 1, dtype=z.dtype, device=z.device)
        zt = (1 - t) * z0 + t * z * m
        v = self._velocity(zt, t, h, state, mask)
        flow = (((v - (z * m - z0)) ** 2) * m).sum() / m.sum().clamp_min(1.0)
        total = ce + flow
        return total, {"ce": ce.item(), "flow": flow.item(), "loss": total.item()}

    @torch.no_grad()
    def sample(self, x, n, steps=64, state=None):
        """n (state, z) draws for one spectrum x of shape [1, channels]; CPU tensors, inactive slots exactly 0."""
        was_training = self.training
        self.eval()
        try:
            dev = self.state_masks.device
            x = torch.as_tensor(x, dtype=torch.float32, device=dev)
            if x.shape != (1, self.cfg["channels"]):
                raise ValueError(f"x must have shape (1, {self.cfg['channels']}), got {tuple(x.shape)}")
            h = self.encode(x)
            if state is None:
                logits = self.discrete(h)[0].masked_fill(~self.mask_known, float("-inf"))
                if not self.mask_known.any():
                    raise ValueError("no state masks known: pass state_masks or train with loss() first")
                states = torch.multinomial(F.softmax(logits, -1), n, replacement=True)
            else:
                if not bool(self.mask_known[state]):
                    raise ValueError(f"mask of state {state} is unknown: pass state_masks or train with loss() first")
                states = torch.full((n,), int(state), dtype=torch.long, device=dev)
            mask = self.state_masks[states]
            m = mask.to(h.dtype)
            hn = h.expand(n, -1)
            z = torch.randn(n, self.n_cont, dtype=h.dtype, device=dev) * m
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.full((n, 1), i * dt, dtype=h.dtype, device=dev)
                z = (z + dt * self._velocity(z, t, hn, states, mask)) * m
            return states.cpu(), z.cpu()
        finally:
            self.train(was_training)

    def save(self, path):
        torch.save({"cfg": self.cfg, "state_dict": self.state_dict()}, Path(path))

    @classmethod
    def load(cls, path):
        ck = torch.load(Path(path), map_location="cpu", weights_only=True)
        cfg = dict(ck["cfg"])
        fixed = cfg.pop("masks_fixed")
        sd = ck["state_dict"]
        model = cls(state_masks=sd["state_masks"] if fixed else None, **cfg)
        model.load_state_dict(sd)
        model.cfg["masks_fixed"] = fixed
        return model.eval()
