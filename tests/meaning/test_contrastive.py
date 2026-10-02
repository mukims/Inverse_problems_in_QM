# tests/meaning/test_contrastive.py
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from meaning.contrastive import (
    StructureEncoder,
    class_batch,
    contrastive_loss,
    embed_structure,
    train_structure_encoder,
)


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


def _toy_classes(seed, n_per=40, L=64):
    rng = np.random.default_rng(seed)
    out = []
    for pos in (10, 30, 50):                       # three "ribbons": a step at different energies
        base = np.zeros(L)
        base[pos:] = 1.0
        out.append((base + 0.05 * rng.standard_normal((n_per, L))).astype(np.float32))
    return out


def test_class_batch_shapes_and_labels(rng):
    x, y = class_batch(rng, [np.zeros((20, 8)), np.ones((20, 8))], k=4)
    assert x.shape == (8, 8) and x.dtype == torch.float32
    assert y.tolist() == [0] * 4 + [1] * 4


def test_class_batch_refuses_small_class(rng):
    with pytest.raises(ValueError, match="class 1"):
        class_batch(rng, [np.zeros((20, 8), np.float32), np.zeros((3, 8), np.float32)], k=4)


def test_encoder_outputs_unit_vectors_and_refuses_bad_length():
    enc = StructureEncoder(latent=8, seq_len=64)
    z = enc(torch.randn(5, 64))
    assert torch.allclose(z.norm(dim=1), torch.ones(5), atol=1e-5)
    with pytest.raises(ValueError, match="multiple of 8"):
        StructureEncoder(latent=8, seq_len=60)


def test_training_reduces_loss_and_is_reproducible():
    tr, va = _toy_classes(1), _toy_classes(2)
    kw = dict(steps=60, k=8, latent=8, threads=1, log_every=30, seed=3)
    m1, h1 = train_structure_encoder(tr, va, "paraphrase", **kw)
    m2, _ = train_structure_encoder(tr, va, "paraphrase", **kw)
    assert h1[-1]["val_loss"] < h1[0]["val_loss"]
    X = np.concatenate(va)
    assert np.allclose(embed_structure(m1, X), embed_structure(m2, X))


def test_physics_mode_requires_distance_matrix():
    tr, va = _toy_classes(1), _toy_classes(2)
    with pytest.raises(ValueError, match="physics"):
        train_structure_encoder(tr, va, "physics", steps=1, k=8, latent=8, threads=1)


def test_embed_structure_handles_empty_input():
    enc = StructureEncoder(latent=8, seq_len=64)
    assert embed_structure(enc, np.zeros((0, 64), np.float32)).shape == (0, 8)
