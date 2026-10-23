# tests/tbribbon/test_parent_prior.py
"""Prior over the hexagonal parent lattice (spec section 4.2): sampler, box-edge warnings, hidden materials."""
from dataclasses import replace

import numpy as np
import pytest

from tbribbon.parent import A1, A2, EDGE_VECTORS, bulk_hamiltonian, named_material, parent_ribbon
from tbribbon.parent_prior import Draw, box_edge_warnings, excluded, in_prior, sample_prior

DRAWS = [sample_prior(np.random.default_rng(i)) for i in range(20000)]


def test_every_draw_is_valid_canonical_and_inside_the_prior():
    for d in DRAWS[:5000]:
        d.params.validate(); assert in_prior(d) and 4 <= d.width <= 22 and d.params.eps_A >= d.params.eps_B


def test_patterns_are_uniform_over_the_seven():
    pats = [d.params.pattern for d in DRAWS]
    for p in set(pats):
        assert abs(pats.count(p) / len(pats) - 1 / 7) < 0.02
    assert len(set(pats)) == 7


def test_point_masses_at_zero():
    h = [d.params for d in DRAWS if d.params.on_H]
    assert abs(np.mean([p.delta == 0 for p in h]) - 0.5) < 0.02
    single_c = [d.params for d in DRAWS if d.params.pattern == (True, False, False)]
    assert abs(np.mean([p.eps_C == 0 for p in single_c]) - 0.5) < 0.04


def test_optional_hoppings_are_zero_or_bounded_away_from_zero():
    for d in DRAWS[:5000]:
        p = d.params
        assert p.t_AA == 0 or 0.03 <= p.t_AA <= 0.3
        for t in (p.t_CH, p.t_CK, p.t_HK):
            assert t == 0 or 0.05 <= t <= 1.0


def test_density_is_log_uniform():
    assert np.median([d.density for d in DRAWS]) == pytest.approx(np.sqrt(0.0025 * 0.05), rel=0.05)


def test_box_edge_warning_flags_widths_and_bounds():
    g = named_material("graphene-ideal")
    assert box_edge_warnings(Draw(g, "armchair", 22, 0.01)) == ["w"]
    assert box_edge_warnings(Draw(replace(g, s=3.99), "armchair", 10, 0.01)) == ["s"]
    assert box_edge_warnings(Draw(g, "armchair", 10, 0.01)) == []


def test_hidden_neighbourhoods():
    assert excluded(Draw(named_material("hbn"), "armchair", 10, 0.01), "hbn")
    assert not excluded(Draw(named_material("graphene-ideal"), "armchair", 10, 0.01), "hbn")
    assert excluded(Draw(named_material("silicene"), "armchair", 10, 0.01), "silicene")
    assert not excluded(Draw(named_material("graphene-ideal"), "armchair", 10, 0.01), "silicene")   # gapless outside s range stays


def test_kagome_and_unknown_hide():
    k = Draw(named_material("kagome"), "zigzag", 10, 0.01)
    assert excluded(k, "kagome") and not excluded(Draw(named_material("triangular"), "zigzag", 10, 0.01), "kagome")
    assert not excluded(Draw(replace(named_material("kagome"), eps_K=0.3), "zigzag", 10, 0.01), "kagome")
    with pytest.raises(ValueError):
        excluded(k, "graphene")


def test_box_edge_ignores_fixed_zero_and_inactive_values():
    g = named_material("graphene-ideal")                     # lead hopping 1, delta 0, t_AA 0: none is a free draw
    assert box_edge_warnings(Draw(g, "zigzag", 4, 0.0025)) == ["w", "c"]
    h = replace(g, delta=0.05, eps_H_mean=1.0, t_AA=0.3)
    assert box_edge_warnings(Draw(h, "armchair", 10, 0.01)) == ["delta", "t_AA"]
    assert box_edge_warnings(Draw(replace(h, delta=0.96), "armchair", 10, 0.01)) == ["t_AA"]   # upper bound is 2 - 1.0 = 1.0
    assert box_edge_warnings(Draw(replace(h, delta=0.99), "armchair", 10, 0.01)) == ["delta", "t_AA"]


def test_in_prior_rejects_outside_draws():
    g = named_material("graphene-ideal")
    ok = lambda p, edge="armchair", w=10, c=0.01: in_prior(Draw(p, edge, w, c))
    assert ok(g) and ok(replace(g, on_K=True, t_KK=0.5, t_HK=0.5))
    assert not ok(replace(g, on_K=True, t_KK=0.5))                      # K not connected
    for bad in (replace(g, s=0.5), replace(g, t_AB=0.9), replace(g, t_AA=0.01), replace(g, delta=0.02),
                replace(g, eps_C=0.5), replace(g, eps_H_mean=2.5), replace(g, delta=1.5, eps_H_mean=1.0)):
        assert not ok(bad)
    assert not ok(g, w=3) and not ok(g, w=23) and not ok(g, c=0.1) and not ok(g, c=0.001) and not ok(g, edge="bearded")


def test_ribbon_bands_lie_in_the_bulk_projection_for_200_draws():
    out = tot = 0
    for i in range(200):
        d = sample_prior(np.random.default_rng(10_000 + i))
        h = parent_ribbon(d.params, d.edge, 8)
        T = EDGE_VECTORS[d.edge][0]
        period = T[0] * A1 + T[1] * A2
        perp = np.array([-period[1], period[0]]) / np.linalg.norm(period)
        kdir = period / np.linalg.norm(period) ** 2
        for th in np.linspace(-np.pi, np.pi, 11):
            er = np.linalg.eigvalsh(h.H0 + h.H1 * np.exp(1j * th) + h.H1.conj().T * np.exp(-1j * th))
            q = np.linspace(-8, 8, 101)
            eb = np.linalg.eigvalsh(bulk_hamiltonian(d.params, th * kdir + q[:, None] * perp))
            inside = np.zeros(len(er), bool)
            for b in range(eb.shape[1]):
                inside |= (er >= eb[:, b].min() - 1e-3) & (er <= eb[:, b].max() + 1e-3)
            out += (~inside).sum()
            tot += len(er)
    assert out / tot < 0.08
