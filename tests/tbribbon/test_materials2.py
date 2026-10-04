# tests/tbribbon/test_materials2.py
"""MATERIALS-2: generic lattice builder (kagome, Lieb, checkerboard), new TMDs, silicene and germanene."""
import numpy as np
import pytest

from tbribbon.bands import band_edges, open_channels
from tbribbon.lattice2d import (CHECKERBOARD, HONEYCOMB, KAGOME, LIEB, TRIANGULAR, Edge, Lattice, bulk_hamiltonian,
                                lattice_ribbon)
from tbribbon.lattices import honeycomb_ribbon, triangular_ribbon
from tbribbon.leads import LeadCache
from tbribbon.transport import spectrum
S3 = np.sqrt(3.0)
FLAT = {"kagome": 2.0, "lieb": 0.0, "checkerboard": 2.0}
BULK = {"kagome": [(-4.0, -1.0), (-1.0, 2.0), (2.0, 2.0)],
        "lieb": [(-2 * np.sqrt(2), 0.0), (0.0, 0.0), (0.0, 2 * np.sqrt(2))],
        "checkerboard": [(-6.0, 2.0), (2.0, 2.0)]}
LATS = {"kagome": KAGOME, "lieb": LIEB, "checkerboard": CHECKERBOARD}
LATTICE_EDGES = [("kagome", "zigzag"), ("kagome", "armchair"), ("lieb", "strip"), ("checkerboard", "strip")]


def _bloch(h, th):
    return h.H0 + h.H1 * np.exp(1j * th) + h.H1.conj().T * np.exp(-1j * th)


def _same_spectrum(h1, h2):
    return all(np.allclose(np.linalg.eigvalsh(_bloch(h1, th)), np.linalg.eigvalsh(_bloch(h2, th)), atol=1e-9)
               for th in np.linspace(-np.pi, np.pi, 13))


@pytest.mark.parametrize("N", [4, 7, 9])
@pytest.mark.parametrize("edge", ["armchair", "zigzag"])
def test_generic_builder_reproduces_honeycomb_and_triangular(N, edge):
    assert _same_spectrum(lattice_ribbon(HONEYCOMB, N, edge), honeycomb_ribbon(N, edge))
    assert _same_spectrum(lattice_ribbon(TRIANGULAR, N, edge), triangular_ribbon(N, edge))


@pytest.mark.parametrize("name", ["kagome", "lieb", "checkerboard"])
def test_bulk_bands_match_the_exact_ranges(name):
    lat = LATS[name]
    B = 2 * np.pi * np.linalg.inv(np.array([lat.a1, lat.a2])).T
    E = np.array([np.linalg.eigvalsh(bulk_hamiltonian(lat, u * B[0] + v * B[1]))
                  for u in np.linspace(0, 1, 61) for v in np.linspace(0, 1, 61)])   # 61 points include K = (1/3, 2/3)
    got = [(E[:, b].min(), E[:, b].max()) for b in range(E.shape[1])]
    assert np.allclose(got, BULK[name], atol=1e-6)


@pytest.mark.parametrize("name,edge", LATTICE_EDGES)
def test_lattice_ribbons_are_hermitian_with_the_expected_size(name, edge):
    lat = LATS[name]
    h = lattice_ribbon(lat, 7, edge)
    e = lat.edge(edge)
    cells = abs(e.T[0] * e.W[1] - e.T[1] * e.W[0])
    assert h.H0.shape == (7 * cells * len(lat.basis),) * 2
    assert np.allclose(h.H0, h.H0.conj().T)


@pytest.mark.parametrize("name,edge", LATTICE_EDGES)
def test_lattice_ribbons_lie_in_the_bulk_projection(name, edge):
    lat = LATS[name]
    h = lattice_ribbon(lat, 12, edge)
    e = lat.edge(edge)
    period = e.T[0] * np.asarray(lat.a1) + e.T[1] * np.asarray(lat.a2)
    perp = np.array([-period[1], period[0]]) / np.linalg.norm(period)
    kdir = period / np.linalg.norm(period) ** 2
    out = tot = 0
    for th in np.linspace(-np.pi, np.pi, 21):
        er = np.linalg.eigvalsh(_bloch(h, th))
        eb = np.array([np.linalg.eigvalsh(bulk_hamiltonian(lat, th * kdir + q * perp)) for q in np.linspace(-8, 8, 301)])
        inside = np.zeros(len(er), bool)
        for b in range(eb.shape[1]):
            inside |= (er >= eb[:, b].min() - 1e-3) & (er <= eb[:, b].max() + 1e-3)
        out += (~inside).sum()
        tot += len(er)
    assert out / tot < 0.08


@pytest.mark.parametrize("name,edge", LATTICE_EDGES)
def test_flat_band_survives_in_the_ribbon(name, edge):
    N = 12
    h = lattice_ribbon(LATS[name], N, edge)
    for th in np.linspace(-np.pi, np.pi, 9):
        assert np.sum(np.abs(np.linalg.eigvalsh(_bloch(h, th)) - FLAT[name]) < 1e-6) >= N - 1


@pytest.mark.parametrize("name,edge", LATTICE_EDGES)
def test_clean_transmission_equals_open_channels_and_vanishes_above_the_band(name, edge):
    h = lattice_ribbon(LATS[name], 7, edge)
    top = band_edges(h.H0, h.H1)[1]
    E = np.linspace(0.01, top + 0.4, 40)
    T = spectrum(h.H0, h.H1, E, np.zeros((10, h.H0.shape[0])), LeadCache(h.H0, h.H1, E), formula="caroli")
    ch = open_channels(h.H0, h.H1, E)
    stable = np.array([len(set(open_channels(h.H0, h.H1, [x - 0.02, x, x + 0.02]))) == 1 for x in E])
    assert stable.sum() >= 20
    assert np.allclose(T[stable], ch[stable], atol=1e-3)
    assert np.all(np.abs(T[E > top + 1e-3]) < 1e-6)


def test_lattice_errors_are_clear():
    with pytest.raises(ValueError, match="no edge"):
        lattice_ribbon(LIEB, 5, "zigzag")
    with pytest.raises(ValueError, match="width"):
        lattice_ribbon(KAGOME, 0, "zigzag")
    bad = Lattice("bad", (1.0, 0.0), (0.0, 1.0), ((0.0, 0.0),), (Edge("strip", (1, 0), (2, 0)),))
    with pytest.raises(ValueError, match="parallel"):
        lattice_ribbon(bad, 3, "strip")
