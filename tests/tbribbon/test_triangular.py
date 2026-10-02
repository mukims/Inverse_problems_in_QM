import numpy as np
import pytest

from tbribbon.bands import band_edges, open_channels
from tbribbon.lattices import triangular_ribbon
from tbribbon.leads import LeadCache
from tbribbon.materials import hamiltonian_for, make_model
from tbribbon.transport import spectrum


@pytest.mark.parametrize("edge, spc_factor", [("zigzag", 1), ("armchair", 2), ("strip", 1)])
def test_make_model_triangular(edge, spc_factor):
    width = 7
    m = make_model("triangular", edge, width)
    assert m.material == "triangular" and m.edge == edge and m.width == width
    assert m.sites_per_cell == width * spc_factor
    assert 2.5 <= m.band_top_t <= 3.0
    h = hamiltonian_for(m)
    assert h.H0.shape == (m.sites_per_cell, m.sites_per_cell)
    assert np.allclose(h.H0, h.H0.conj().T)


@pytest.mark.parametrize("edge", ["zigzag", "armchair"])
def test_triangular_band_extrema_and_dispersion(edge):
    h = triangular_ribbon(10, edge)
    b_min, b_max = band_edges(h.H0, h.H1)
    # Bulk limits for onsite=0, hopping -t are [-6.0t, +3.0t]
    assert b_min >= -6.0001
    assert b_max <= 3.0001
    assert b_min < -5.5
    assert b_max > 2.8


@pytest.mark.parametrize("edge", ["zigzag", "armchair"])
def test_triangular_channel_invariant(edge):
    width = 6 if edge == "zigzag" else 4
    h = triangular_ribbon(width, edge)
    assert np.allclose(h.H0, h.H0.conj().T)

    # Test energies across the positive energy spectrum
    E = np.linspace(0.1, 2.7, 30)
    leads = LeadCache(h.H0, h.H1, E)
    T = spectrum(h.H0, h.H1, E, np.zeros((10, h.H0.shape[0])), leads, formula="caroli")
    ch = open_channels(h.H0, h.H1, E)

    stable = np.array([len(set(open_channels(h.H0, h.H1, [e - 0.02, e, e + 0.02]))) == 1 for e in E])
    assert stable.sum() >= 20, f"Too few stable points: {stable.sum()}"
    diff = np.max(np.abs(T[stable] - ch[stable]))
    assert diff < 1e-3, f"Triangular {edge} channel mismatch: {diff}"


@pytest.mark.parametrize("edge", ["zigzag", "armchair"])
def test_triangular_zero_above_band_top(edge):
    h = triangular_ribbon(6, edge)
    _, b_max = band_edges(h.H0, h.H1)
    E = np.linspace(b_max + 0.05, 3.8, 10)
    leads = LeadCache(h.H0, h.H1, E)
    T = spectrum(h.H0, h.H1, E, np.zeros((5, h.H0.shape[0])), leads, formula="caroli")
    assert np.all(T < 1e-6), f"Triangular {edge} transmission above band top not zero: {T.max()}"
