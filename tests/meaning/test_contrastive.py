# tests/meaning/test_contrastive.py
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from meaning.contrastive import contrastive_loss


def _unit(n, d, seed=0):
    return F.normalize(torch.randn(n, d, generator=torch.Generator().manual_seed(seed)), dim=1)


def test_paraphrase_loss_prefers_class_clusters():
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    clustered = torch.eye(3)[y]
    assert float(contrastive_loss(clustered, y)) < float(contrastive_loss(_unit(6, 3), y))


def test_physics_targets_reduce_to_paraphrase_for_tiny_sigma():
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    z = _unit(6, 4)
    D = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    assert float(contrastive_loss(z, y, D=D, sigma=1e-3)) == pytest.approx(float(contrastive_loss(z, y)), rel=1e-4)


def test_physics_loss_needs_positive_sigma():
    y = torch.tensor([0, 0, 1, 1])
    D = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="sigma"):
        contrastive_loss(_unit(4, 3), y, D=D, sigma=0.0)


def test_anchor_without_positive_is_refused():
    with pytest.raises(ValueError, match="at least one other"):
        contrastive_loss(_unit(3, 4), torch.tensor([0, 1, 1]))


def test_loss_gradients_are_finite():
    z = _unit(6, 4).requires_grad_(True)
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    contrastive_loss(z, y).backward()
    assert torch.isfinite(z.grad).all()
