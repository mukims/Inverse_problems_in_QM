# tests/tbribbon/test_materials2.py
"""MATERIALS-2: generic lattice builder (kagome, Lieb, checkerboard), new TMDs, silicene and germanene."""
import numpy as np
import pytest

from tbribbon.bands import band_edges, open_channels
from tbribbon.lattice2d import (CHECKERBOARD, HONEYCOMB, KAGOME, LIEB, TRIANGULAR, Edge, Lattice, bulk_hamiltonian,
                                lattice_ribbon)
from tbribbon.lattices import honeycomb_ribbon, mos2_ribbon, tmd_midgap, triangular_ribbon
from tbribbon.leads import LeadCache
from tbribbon.materials import MATERIALS, hamiltonian_for, make_model
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


# ---------- TMDs ----------

TMD = {"mos2": dict(eps1=1.046, eps2=2.104, t0=-0.184, t1=0.401, t2=0.507, t11=0.218, t12=0.338, t22=0.057)}
TMD.update({m: MATERIALS[m]["params"] for m in ("ws2", "mose2", "wse2")})
MIDGAP = {"mos2": 0.766600, "ws2": 0.845089, "mose2": 0.764808, "wse2": 0.793983}     # Liu 2013 Table I at K
GAP_K = {"mos2": 1.662799, "ws2": 1.805823, "mose2": 1.436384, "wse2": 1.540034}
RV = {1: np.array([1.0, 0.0]), 2: np.array([0.5, S3 / 2]), 3: np.array([-0.5, S3 / 2])}


def _tmd_bulk(p):
    em = round(tmd_midgap(p["eps1"], p["eps2"], p["t0"], p["t11"], p["t12"], p["t22"]), 4)
    t0, t1, t2, t11, t12, t22 = (p[k] for k in ("t0", "t1", "t2", "t11", "t12", "t22"))
    onsite = np.diag([p["eps1"] - em, p["eps2"] - em, p["eps2"] - em]).astype(complex)
    h = {1: np.array([[t0, t1, t2], [-t1, t11, t12], [t2, -t12, t22]]),
         2: np.array([[t0, t1/2 + S3*t2/2, S3*t1/2 - t2/2],
                      [-t1/2 + S3*t2/2, t11/4 + 3*t22/4, S3*t11/4 - t12 - S3*t22/4],
                      [-S3*t1/2 - t2/2, S3*t11/4 + t12 - S3*t22/4, 3*t11/4 + t22/4]]),
         3: np.array([[t0, -t1/2 - S3*t2/2, S3*t1/2 - t2/2],
                      [t1/2 - S3*t2/2, t11/4 + 3*t22/4, -S3*t11/4 + t12 + S3*t22/4],
                      [-S3*t1/2 - t2/2, -S3*t11/4 - t12 + S3*t22/4, 3*t11/4 + t22/4]])}

    def ev(k):
        H = onsite.copy()
        for j in (1, 2, 3):
            ph = np.exp(1j * k @ RV[j])
            H = H + h[j] * ph + h[j].T / ph
        return np.linalg.eigvalsh(H)
    return ev


@pytest.mark.parametrize("m", ["mos2", "ws2", "mose2", "wse2"])
def test_tmd_midgap_and_gap_match_liu_table_i(m):
    p = TMD[m]
    assert tmd_midgap(p["eps1"], p["eps2"], p["t0"], p["t11"], p["t12"], p["t22"]) == pytest.approx(MIDGAP[m], abs=1e-6)
    eK = _tmd_bulk(p)(np.array([4 * np.pi / 3, 0.0]))
    assert eK[1] - eK[0] == pytest.approx(GAP_K[m], abs=1e-6)
    assert abs((eK[0] + eK[1]) / 2) < 1e-4                  # mid-gap at E = 0 (D4)


def test_mos2_default_is_unchanged():
    h = mos2_ribbon(7, "armchair")
    assert np.allclose(np.diag(h.H0)[:3].real, [1.046 - 0.7666, 2.104 - 0.7666, 2.104 - 0.7666])


@pytest.mark.parametrize("m", ["ws2", "mose2", "wse2"])
@pytest.mark.parametrize("edge,N,period", [("zigzag", 30, (1.0, 0.0)), ("armchair", 15, (0.0, S3))])
def test_tmd_ribbons_lie_in_the_bulk_projection(m, edge, N, period):
    h = mos2_ribbon(N, edge, **TMD[m])
    ev = _tmd_bulk(TMD[m])
    period = np.array(period)
    perp = np.array([-period[1], period[0]]) / np.linalg.norm(period)
    kdir = period / np.linalg.norm(period) ** 2
    out = tot = 0
    for th in np.linspace(-np.pi, np.pi, 31):
        er = np.linalg.eigvalsh(_bloch(h, th))
        eb = np.array([ev(th * kdir + q * perp) for q in np.linspace(-8, 8, 601)])
        inside = np.zeros(len(er), bool)
        for b in range(3):
            inside |= (er >= eb[:, b].min() - 1e-3) & (er <= eb[:, b].max() + 1e-3)
        out += (~inside).sum()
        tot += len(er)
    assert out / tot < 0.08


# ---------- registry ----------

NEW = [("ws2", "armchair"), ("ws2", "zigzag"), ("mose2", "armchair"), ("mose2", "zigzag"), ("wse2", "armchair"),
       ("wse2", "zigzag"), ("silicene", "armchair"), ("silicene", "zigzag"), ("germanene", "armchair"),
       ("germanene", "zigzag"), ("kagome", "armchair"), ("kagome", "zigzag"), ("lieb", "strip"), ("checkerboard", "strip")]


@pytest.mark.parametrize("material,edge", NEW)
def test_make_model_new_materials(material, edge):
    m = make_model(material, edge, 9)
    h = hamiltonian_for(m)
    assert h.H0.shape == (m.sites_per_cell, m.sites_per_cell) and np.allclose(h.H0, h.H0.conj().T)
    assert m.band_top_t > 0
    if material in ("ws2", "mose2", "wse2"):
        assert m.orbitals_per_site == 3 and m.impurity_v_t == pytest.approx(0.5 * MATERIALS[material]["params"]["t2"], abs=5e-4)
    else:
        assert m.orbitals_per_site == 1 and m.impurity_v_t == 0.5


def test_silicene_and_germanene_are_graphene_on_their_own_energy_scale():
    for mat, t_ev in (("silicene", 1.067), ("germanene", 0.991)):
        m = make_model(mat, "zigzag", 9)
        assert m.t_ev == t_ev
        assert np.allclose(hamiltonian_for(m).H0, honeycomb_ribbon(9, "zigzag").H0)

