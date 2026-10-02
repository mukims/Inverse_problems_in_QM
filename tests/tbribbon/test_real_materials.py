import numpy as np
import pytest

from tbribbon.bands import band_edges, open_channels
from tbribbon.lattices import hbn_ribbon, mos2_ribbon, phosphorene_ribbon
from tbribbon.leads import LeadCache
from tbribbon.materials import hamiltonian_for, make_model
from tbribbon.transport import spectrum


@pytest.mark.parametrize("material", ["hbn", "phosphorene", "mos2"])
@pytest.mark.parametrize("edge", ["armchair", "zigzag"])
def test_make_model_real_materials(material, edge):
    m = make_model(material, edge, 7)
    assert m.material == material and m.edge == edge and m.width == 7
    assert m.t_ev > 0.0
    assert m.sites_per_cell > 0
    assert m.band_top_t > 0.0
    h = hamiltonian_for(m)
    assert h.H0.shape == (m.sites_per_cell, m.sites_per_cell)
    assert np.allclose(h.H0, h.H0.conj().T)


@pytest.mark.parametrize("edge", ["armchair", "zigzag"])
def test_hbn_physics_and_channel_invariant(edge):
    t_ev = 2.30
    delta_ev = 3.625
    delta_t = delta_ev / t_ev  # ~1.576 t

    h = hbn_ribbon(7, edge)
    assert np.allclose(h.H0, h.H0.conj().T)

    # Test energies across gap and conduction band
    E = np.linspace(0.1, 3.2, 32)
    leads = LeadCache(h.H0, h.H1, E)
    T = spectrum(h.H0, h.H1, E, np.zeros((10, h.H0.shape[0])), leads, formula="caroli")
    ch = open_channels(h.H0, h.H1, E)

    # Zero transmission strictly inside bulk gap
    in_gap = E < delta_t - 0.05
    assert np.all(T[in_gap] < 1e-6)

    # Clean T equals open channels away from subband steps
    stable = np.array([len(set(open_channels(h.H0, h.H1, [e - 0.02, e, e + 0.02]))) == 1 for e in E])
    assert np.allclose(T[stable], ch[stable], atol=1e-3)


@pytest.mark.parametrize("edge", ["armchair", "zigzag"])
def test_phosphorene_physics_and_channel_invariant(edge):
    h = phosphorene_ribbon(6, edge)
    assert np.allclose(h.H0, h.H0.conj().T)

    # Test energies in the conduction band (midgap is centered at 0.0)
    E = np.linspace(0.25, 1.8, 30)
    leads = LeadCache(h.H0, h.H1, E)
    T = spectrum(h.H0, h.H1, E, np.zeros((10, h.H0.shape[0])), leads, formula="caroli")
    ch = open_channels(h.H0, h.H1, E)

    stable = np.array([len(set(open_channels(h.H0, h.H1, [e - 0.02, e, e + 0.02]))) == 1 for e in E])
    diff = np.max(np.abs(T[stable] - ch[stable]))
    assert diff < 1e-3, f"Phosphorene {edge} channel mismatch: {diff}"


def _find_band_extrema(H0, H1, nk=800):
    k_vals = np.linspace(0, np.pi, nk)
    evals = np.array([np.linalg.eigvalsh(H0 + H1 * np.exp(1j * k) + H1.conj().T * np.exp(-1j * k)) for k in k_vals])
    extrema = []
    for b in range(evals.shape[1]):
        band = evals[:, b]
        extrema.extend([band[0], band[-1]])
        diffs = np.diff(band)
        sc = np.where(diffs[:-1] * diffs[1:] <= 0)[0]
        extrema.extend(band[sc + 1])
    return np.sort(np.unique(extrema))


@pytest.mark.parametrize("edge", ["armchair", "zigzag"])
def test_mos2_physics_and_channel_invariant(edge):
    h = mos2_ribbon(4 if edge == "armchair" else 6, edge)
    assert np.allclose(h.H0, h.H0.conj().T)

    # Test energies in conduction band (above midgap 0.0)
    E = np.linspace(0.85, 2.7, 40)
    leads = LeadCache(h.H0, h.H1, E)
    T = spectrum(h.H0, h.H1, E, np.zeros((10, h.H0.shape[0])), leads, formula="caroli")
    ch = open_channels(h.H0, h.H1, E)

    extrema = _find_band_extrema(h.H0, h.H1)
    dists = np.min(np.abs(E[:, None] - extrema[None, :]), axis=1)
    stable = dists > 0.025
    assert stable.sum() >= 10, f"Too few stable points: {stable.sum()}"
    diff = np.max(np.abs(T[stable] - ch[stable]))
    assert diff < 1e-3, f"MoS2 {edge} channel mismatch: {diff}"
