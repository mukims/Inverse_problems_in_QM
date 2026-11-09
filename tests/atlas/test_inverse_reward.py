import numpy as np
import pytest
import torch

from shazam.inverse import reward as rw
from shazam.inverse.encoding import STATES, decode, encode
from tbribbon.parent import HexParams
from tbribbon.parent_prior import Draw, sample_prior

# the same host written with lead C and with lead K: equal within-type hoppings are the only case where both leads lie
# inside the box (t_within <= 1)
LEAD_C = HexParams(True, False, True, "C", 2.0, eps_C=0.1, eps_K=-0.3, t_CC=1.0, t_KK=1.0, t_CK=0.5)
LEAD_K = HexParams(True, False, True, "K", 2.0, eps_C=0.1, eps_K=-0.3, t_CC=1.0, t_KK=1.0, t_CK=0.5)
SCALES = rw.PhysicsScales(on_magnitude_ev={"delta": 1.0, "t_AA": 0.3, "t_CH": 1.0, "t_CK": 1.0, "t_HK": 1.0,
                                           "eps": 1.5},
                          mean_weight={p: [1 / 3] * 3 for p in ("C", "H", "K", "CH", "CK", "HK", "CHK")})


def _state(params, edge="zigzag"):
    return encode(Draw(params, edge, 9, 0.01))[0]


def _hbn_like(delta_model, s=2.0):
    return HexParams(False, True, False, "H", s, eps_H_mean=0.2, delta=delta_model, t_AB=1.0)


def _flip(k, **change):
    return next(i for i, s in enumerate(STATES) if s == STATES[k]._replace(**change))


def test_family_key_ignores_the_lead_and_counts_74_families():
    assert rw.N_FAMILIES == 74
    ka, kb = _state(LEAD_C), _state(LEAD_K)
    assert ka != kb and rw.family_key(ka) == rw.family_key(kb)
    assert rw.FAMILY_OF[ka] == rw.FAMILY_OF[kb]


def test_physical_vector_is_lead_free():
    a, b = Draw(LEAD_C, "zigzag", 9, 0.01), Draw(LEAD_K, "zigzag", 9, 0.01)
    np.testing.assert_allclose(rw.physical(a), rw.physical(b), atol=1e-12)


def test_physical_vector_is_in_ev_with_switched_off_zero():
    v = rw.physical(Draw(_hbn_like(0.8), "armchair", 7, 0.02))
    assert v[rw.EV_NAMES.index("delta")] == pytest.approx(1.6)
    assert v[rw.EV_NAMES.index("t_AB")] == pytest.approx(2.0)
    assert v[rw.EV_NAMES.index("t_AA")] == 0 and v[rw.EV_NAMES.index("eps_K")] == 0
    assert v[-2] == pytest.approx(np.log(0.02)) and v[-1] == 7


def test_physical_torch_matches_decode():
    rng = np.random.default_rng(0)
    draws = [sample_prior(rng) for _ in range(500)]
    enc = [encode(d) for d in draws]
    states = torch.tensor([e[0] for e in enc])
    z = torch.tensor(np.stack([e[1] for e in enc]), dtype=torch.float64)
    z = z + torch.randn_like(z) * torch.tensor(np.stack([e[2] for e in enc]))          # off the training points too
    got = rw.physical_torch(states, z).numpy()
    for k, zz, g in zip(states.numpy(), z.numpy(), got):
        d = decode(int(k), zz)
        want = rw.physical(d)
        np.testing.assert_allclose(g[:-1], want[:-1], atol=1e-6)                     # width: unrounded in torch
        assert abs(g[-1] - want[-1]) <= 0.5 + 1e-9


def test_empty_delta_interval_gives_zero_delta():
    p = HexParams(False, True, False, "H", 2.0, eps_H_mean=1.97, delta=0.0, t_AB=1.0)
    k, z, _ = encode(Draw(p, "armchair", 7, 0.01))
    k_on = _flip(k, delta_zero=False)                         # a Delta-on state, but the interval above eps_H is empty
    z = torch.tensor(z[None], dtype=torch.float64)
    z[0, rw.SLOT["delta"]] = 3.0
    assert rw.physical_torch(torch.tensor([k_on]), z)[0, rw.EV_NAMES.index("delta")] == 0
    assert decode(k_on, z[0].numpy()).params.delta == 0


def test_physical_torch_has_gradients():
    rng = np.random.default_rng(1)
    k, z, _ = encode(sample_prior(rng))
    z = torch.tensor(z[None], dtype=torch.float64, requires_grad=True)
    rw.physical_torch(torch.tensor([k]), z).sum().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_width_ramp_and_surrogate():
    assert [rw.width_cost(d) for d in (0, 1, 2, 3, 4, 7)] == [0, 0, pytest.approx(1 / 3), pytest.approx(2 / 3), 1, 1]
    dw = torch.tensor([-9.0, -4.0, 0.5, 4.0, 9.0], requires_grad=True)
    rw.width_surrogate(dw).sum().backward()
    assert torch.all(torch.isfinite(dw.grad)) and dw.grad[0] < 0 and dw.grad[-1] > 0 and dw.grad[2] == 0
    assert dw.grad[-1] == dw.grad[-2]                         # linear beyond the Huber knee: never saturates


def test_width_surrogate_has_a_gradient_at_the_box_edges():
    # a sample decoded at w = 4 or 22 sits where the slot saturates; the surrogate on the unrounded width still points
    # back towards the truth
    lo, hi = rw.W_BOUNDS
    w = torch.tensor([lo + 1e-3, hi - 1e-3], requires_grad=True)
    rw.width_surrogate(w - torch.tensor([12.0, 12.0])).sum().backward()
    assert w.grad[0] < 0 and w.grad[1] > 0


def test_label_table_is_zero_only_on_the_true_family():
    k_true = _state(LEAD_C)
    t = rw.label_family_cost_table(np.array([k_true]))[0]
    assert t.dtype == np.float32 and t.shape == (156,)
    assert set(np.flatnonzero(t == 0)) == {k for k in range(156) if rw.FAMILY_OF[k] == rw.FAMILY_OF[k_true]}
    assert set(np.unique(t)) == {0.0, 1.0}


def _physics_row(draw, weights=None):
    w = weights if weights is not None else rw.band_weight_row(draw.params)
    return rw.physics_family_cost_table([draw], np.array([w]), SCALES)[0]


def test_physics_cost_is_continuous_as_delta_goes_to_zero():
    for delta_ev, want in ((0.01, 1 - np.exp(-0.02)), (0.5, 1 - np.exp(-1.0))):
        d = Draw(_hbn_like(delta_ev / 2.0), "zigzag", 9, 0.01)
        k = encode(d)[0]
        row = _physics_row(d, [0, 1, 0])
        assert row[k] == 0
        assert row[_flip(k, delta_zero=True)] == pytest.approx(want, rel=1e-5)


def test_physics_cost_of_a_dropped_type_scales_with_its_band_weight():
    chk = HexParams(True, True, True, "H", 2.0, eps_C=0.1, eps_H_mean=0.2, delta=0.5, eps_K=-0.4, t_CC=0.8, t_AB=1.0,
                    t_KK=0.7, t_CK=0.5, t_HK=0.4)
    d = Draw(chk, "zigzag", 9, 0.01)
    k = encode(d)[0]
    ck = next(i for i, s in enumerate(STATES)
              if s.edge == "zigzag" and s.pattern == (True, False, True) and s.lead == "C")
    costs = [_physics_row(d, [0.5, w, 0.5 - w])[ck] for w in (0.02, 0.10, 0.20)]
    assert costs[0] < costs[1] < costs[2]
    assert costs[1] == pytest.approx(1 - np.exp(-1.0), rel=1e-5)
    assert _physics_row(d, [0.5, 0.1, 0.4])[k] == 0


def test_physics_cost_ignores_the_lead():
    d = Draw(LEAD_C, "zigzag", 9, 0.01)
    row = _physics_row(d, [0.5, 0, 0.5])
    assert row[_state(LEAD_C)] == 0 and row[_state(LEAD_K)] == 0


def test_wrong_edge_costs_one_in_both_arms():
    d = Draw(LEAD_C, "zigzag", 9, 0.01)
    k = _state(LEAD_C)
    other = _flip(k, edge="armchair")
    assert rw.label_family_cost_table(np.array([k]))[0, other] == 1
    assert _physics_row(d, [0.5, 0, 0.5])[other] == 1


def test_physics_sample_cost_matches_the_table_on_the_true_switches():
    rng = np.random.default_rng(2)
    draws = [sample_prior(rng) for _ in range(50)]
    weights = np.array([rw.band_weight_row(d.params) for d in draws])
    enc = [encode(d) for d in draws]
    states = torch.tensor([e[0] for e in enc])
    z = torch.tensor(np.stack([e[1] for e in enc]))
    truth = rw.truth_arrays(draws, weights)
    sample = rw.physics_family_cost_sample(truth, states, z, SCALES)
    assert torch.allclose(sample, torch.zeros(50, dtype=sample.dtype))     # every sample is its own truth


def test_continuous_cost_is_zero_at_the_truth_and_lead_free():
    a, b = Draw(LEAD_C, "zigzag", 9, 0.01), Draw(LEAD_K, "zigzag", 9, 0.01)
    ta = torch.tensor(rw.physical(a)[None])
    tb = torch.tensor(rw.physical(b)[None])
    sa, sb = torch.tensor([_state(LEAD_C)]), torch.tensor([_state(LEAD_K)])
    for arm in ("label", "physics"):
        assert rw.continuous_cost(arm, ta, tb, sa, sb, 1.0, 1.0)[0] == pytest.approx(0, abs=1e-12)
    off = tb.clone()
    off[0, rw.EV_NAMES.index("t_CK")] += 0.1                                 # one parameter 0.1 eV off: d = 1
    n = int((ta[0, :len(rw.EV_NAMES)] != 0).sum()) + 2
    assert rw.continuous_cost("label", ta, off, sa, sb, 1.0, 1.0)[0] == pytest.approx((1 - np.exp(-0.5)) / n)


def test_label_continuous_cost_is_one_for_a_wrong_family():
    d = Draw(_hbn_like(0.5), "zigzag", 9, 0.01)
    k = encode(d)[0]
    t = torch.tensor(rw.physical(d)[None])
    assert rw.continuous_cost("label", t, t, torch.tensor([k]), torch.tensor([_flip(k, delta_zero=True)]), 1, 1)[0] == 1


def test_physics_continuous_cost_is_continuous_as_delta_goes_to_zero():
    small = Draw(_hbn_like(0.0005), "zigzag", 9, 0.01)
    k = encode(small)[0]
    t = torch.tensor(rw.physical(small)[None])
    p = t.clone()
    p[0, rw.EV_NAMES.index("delta")] = 0.0
    c = rw.continuous_cost("physics", t, p, torch.tensor([k]), torch.tensor([_flip(k, delta_zero=True)]), 1, 1)[0]
    assert c < 1e-3


def test_state_state_table_is_zero_on_the_diagonal_and_lead_free():
    table = rw.state_state_physics_table(SCALES)
    assert table.shape == (156, 156) and np.all(np.diag(table) == 0)
    assert table[_state(LEAD_C), _state(LEAD_K)] == 0


# ---------------------------------------------------------------- the batched sampler (Task 4): it feeds the flow loss

def _small_model():
    from shazam.inverse.model import InverseModel
    masks = torch.tensor([[1] * 4 + [0] * 10, [1] * 8 + [0] * 6, [1] * 14], dtype=torch.bool)
    torch.manual_seed(0)
    return InverseModel(3, channels=416, state_masks=masks).eval()


def test_sample_batch_matches_sample():
    model = _small_model()
    x = torch.randn(1, 416)
    torch.manual_seed(5)
    states, z_ref = model.sample(x, 8, state=2)
    torch.manual_seed(5)
    with torch.no_grad():
        h = model.encode(x)
    z0 = torch.randn(8, 14)                       # the same draw sample() makes after its encode
    z = model.sample_batch(h.expand(8, -1), states, z0)
    assert torch.allclose(z, z_ref, atol=1e-6)


def test_sample_batch_gradient_reaches_velocity_only_when_asked():
    model = _small_model()
    h = model.encode(torch.randn(4, 416))
    states = torch.tensor([0, 1, 2, 2])
    z0 = torch.randn(4, 14)
    assert not model.sample_batch(h.detach(), states, z0, grad_last=0).requires_grad
    out = model.sample_batch(h, states, z0, steps=8, grad_last=2)
    out.sum().backward()
    last = model.velocity[-1].weight.grad
    assert last is not None and last.abs().sum() > 0
    assert model.training is False                # the mode is left as it was
