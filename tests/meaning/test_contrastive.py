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


from meaning.contrastive import stress_loss


def test_single_sigma_list_equals_float_sigma():
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    z = _unit(6, 4)
    D = torch.tensor([[0.0, 0.04, 0.4], [0.04, 0.0, 0.3], [0.4, 0.3, 0.0]])
    assert float(contrastive_loss(z, y, D=D, sigma=[0.038])) == pytest.approx(float(contrastive_loss(z, y, D=D, sigma=0.038)))


def test_multiscale_targets_still_rank_far_ribbons():
    # anchor ribbon 0; ribbon 1 at 0.04 (a width step), ribbon 2 at 0.4 (another material)
    D = torch.tensor([0.0, 0.04, 0.4])
    single = torch.exp(-D / 0.038)
    multi = torch.stack([torch.exp(-D / s) for s in (0.038, 0.152, 0.608)]).mean(0)
    assert float(single[2] / single[1]) < 1e-3          # BUILD-18: the far ribbon's target is ~0 (saturated)
    assert float(multi[2] / multi[1]) > 0.1             # multiscale keeps it ranked


def test_multiscale_sigmas_must_be_positive():
    y = torch.tensor([0, 0, 1, 1])
    D = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="sigma"):
        contrastive_loss(_unit(4, 3), y, D=D, sigma=[0.1, 0.0])


def test_stress_is_zero_when_distances_match_and_grows_otherwise():
    y = torch.tensor([0, 1, 2])
    D = torch.tensor([[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]])
    on_line = torch.tensor([[0.0], [0.5], [1.5]])        # alpha = 0.5 reproduces D exactly
    assert float(stress_loss(on_line, y, D, alpha=0.5)) == pytest.approx(0.0, abs=1e-6)
    assert float(stress_loss(on_line * 2, y, D, alpha=0.5)) > 0.1


def test_stress_gradient_is_finite_for_identical_embeddings():
    y = torch.tensor([0, 0, 1, 1])
    D = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    z = torch.zeros(4, 3, requires_grad=True)            # every pair at distance 0
    stress_loss(z, y, D, alpha=0.5).backward()
    assert torch.isfinite(z.grad).all()


def test_stress_needs_two_ribbons():
    with pytest.raises(ValueError, match="two ribbons"):
        stress_loss(_unit(3, 2), torch.tensor([0, 0, 0]), torch.zeros(1, 1), alpha=1.0)
