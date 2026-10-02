# notebooks/material_atlas/meaning/contrastive.py
"""Structure embedding: distance means "same ribbon" (paraphrase) or "similar clean physics" (physics)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
