# notebooks/material_atlas/meaning/contrastive.py
"""Structure embedding: distance means "same ribbon" (paraphrase) or "similar clean physics" (physics)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from atlaslib.encoder import Conv1dAE


def contrastive_loss(z, y, tau=0.1, D=None, sigma=None):
    """Cross-entropy between target weights and softmax(z_i . z_j / tau) over j != i.

    D=None: targets are the other spectra of the same ribbon (supervised contrastive).
    D given: targets exp(-D[y_i, y_j] / sigma), so physically similar ribbons are repelled less.
    """
    n = len(z)
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    logits = (z @ z.T / tau).masked_fill(self_mask, -1e9)
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    if D is None:
        target = ((y[:, None] == y[None, :]) & ~self_mask).float()
        if bool((target.sum(1) == 0).any()):
            raise ValueError("every anchor needs at least one other spectrum of its ribbon in the batch")
    else:
        if sigma is None or sigma <= 0:
            raise ValueError(f"physics mode needs sigma > 0, got {sigma}")
        target = torch.exp(-D[y][:, y] / sigma).masked_fill(self_mask, 0.0)
    target = target / target.sum(1, keepdim=True)
    return -(target * logp).sum(1).mean()


class StructureEncoder(nn.Module):
    """Shazam's encoder layers (Conv1dAE) with an L2-normalised output, so distance is cosine-based."""

    def __init__(self, latent=32, seq_len=400):
        super().__init__()
        if seq_len % 8:
            raise ValueError(f"seq_len must be a multiple of 8, got {seq_len}")
        ae = Conv1dAE(latent, seq_len)
        self.encoder, self.to_latent = ae.encoder, ae.to_latent

    def forward(self, x):
        return F.normalize(self.to_latent(self.encoder(x.unsqueeze(1)).flatten(1)), dim=1)


def class_batch(rng, X_by_class, k):
    """k spectra from every class, without replacement."""
    xs, ys = [], []
    for c, X in enumerate(X_by_class):
        if len(X) < k:
            raise ValueError(f"class {c} has {len(X)} spectra, fewer than k={k}")
        pick = rng.choice(len(X), k, replace=False)
        xs.append(X[pick])
        ys.append(np.full(k, c))
    x = torch.from_numpy(np.ascontiguousarray(np.concatenate(xs), dtype=np.float32))
    return x, torch.from_numpy(np.concatenate(ys)).long()


def train_structure_encoder(X_train, X_val, mode, D=None, sigma=None, steps=2500, k=16, tau=0.1, lr=1e-3,
                            seed=0, threads=16, log_every=250, latent=32):
    if mode not in ("paraphrase", "physics"):
        raise ValueError(f"mode must be 'paraphrase' or 'physics', got {mode!r}")
    if mode == "physics" and (D is None or sigma is None):
        raise ValueError("physics mode needs the clean-distance matrix D and sigma")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Dt = torch.as_tensor(D, dtype=torch.float32) if mode == "physics" else None
    model = StructureEncoder(latent, X_train[0].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    xv, yv = class_batch(np.random.default_rng(seed + 1), X_val, k)      # one fixed validation batch
    history = []
    for step in range(1, steps + 1):
        model.train()
        xb, yb = class_batch(rng, X_train, k)
        loss = contrastive_loss(model(xb), yb, tau, Dt, sigma)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step == 1 or step % log_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val = float(contrastive_loss(model(xv), yv, tau, Dt, sigma))
            history.append({"step": step, "train_loss": loss.item(), "val_loss": val})
            print(f"[meaning] {mode} step {step:5d} train {loss.item():.4f} val {val:.4f}", flush=True)
    model.eval()
    return model, history


@torch.no_grad()
def embed_structure(model, X, batch=8192):
    X = np.ascontiguousarray(X, dtype=np.float32)
    if len(X) == 0:
        return np.empty((0, model.to_latent.out_features), dtype=np.float32)
    return np.concatenate([model(torch.from_numpy(X[i:i + batch])).numpy() for i in range(0, len(X), batch)])
