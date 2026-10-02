import numpy as np
import pytest

from tbribbon.bands import band_edges, open_channels
from tbribbon.lattices import hbn_ribbon, mos2_ribbon, phosphorene_ribbon, triangular_ribbon
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


S3 = np.sqrt(3)
RV = {1: np.array([1.0, 0.0]), 2: np.array([0.5, S3 / 2]), 3: np.array([-0.5, S3 / 2])}


def _mos2_bulk():
    z = mos2_ribbon(2, "zigzag")                       # rows 0 and 1 carry every bulk block
    onsite = z.H0[0:3, 0:3]
    hops = {1: z.H1[0:3, 0:3]}
    # take R2/R3 from the reference values, not from the ribbon (the ribbon is what is under test)
    t0, t1, t2, t11, t12, t22 = -0.184, 0.401, 0.507, 0.218, 0.338, 0.057
    hops[2] = np.array([[t0, t1/2 + S3*t2/2, S3*t1/2 - t2/2],
                        [-t1/2 + S3*t2/2, t11/4 + 3*t22/4, S3*t11/4 - t12 - S3*t22/4],
                        [-S3*t1/2 - t2/2, S3*t11/4 + t12 - S3*t22/4, 3*t11/4 + t22/4]])
    hops[3] = np.array([[t0, -t1/2 - S3*t2/2, S3*t1/2 - t2/2],
                        [t1/2 - S3*t2/2, t11/4 + 3*t22/4, -S3*t11/4 + t12 + S3*t22/4],
                        [-S3*t1/2 - t2/2, -S3*t11/4 - t12 + S3*t22/4, 3*t11/4 + t22/4]])

    def ev(k):
        H = onsite.astype(complex).copy()
        for j in (1, 2, 3):
            ph = np.exp(1j * k @ RV[j])
            H = H + hops[j] * ph + hops[j].T / ph
        return np.linalg.eigvalsh(H)
    return ev


def _share_outside(h, ev, period, nb):
    perp = np.array([-period[1], period[0]]) / np.linalg.norm(period)
    kdir = period / np.linalg.norm(period) ** 2
    out = tot = 0
    for th in np.linspace(-np.pi, np.pi, 31):
        er = np.linalg.eigvalsh(h.H0 + h.H1 * np.exp(1j * th) + h.H1.conj().T * np.exp(-1j * th))
        eb = np.array([ev(th * kdir + q * perp) for q in np.linspace(-8, 8, 601)]).reshape(601, nb)
        lo, hi = eb.min(0), eb.max(0)
        inside = np.zeros(len(er), bool)
        for b in range(nb):
            inside |= (er >= lo[b] - 1e-3) & (er <= hi[b] + 1e-3)
        out += (~inside).sum()
        tot += len(er)
    return out / tot


def test_mos2_bulk_matches_liu_nn_model():
    ev = _mos2_bulk()
    e_mid = 0.7666
    assert np.allclose(ev(np.zeros(2)) + e_mid, [-0.058, 2.929, 2.929], atol=1e-3)
    eK = ev(np.array([4 * np.pi / 3, 0.0])) + e_mid
    assert eK[1] - eK[0] == pytest.approx(1.663, abs=2e-3)


@pytest.mark.parametrize("edge,N,period", [("zigzag", 30, (1.0, 0.0)), ("armchair", 15, (0.0, S3))])
def test_mos2_ribbon_bands_lie_in_bulk_projection(edge, N, period):
    # only edge states may fall in bulk gaps; a misassigned bond puts more than half the states outside
    assert _share_outside(mos2_ribbon(N, edge), _mos2_bulk(), np.array(period), 3) < 0.08


@pytest.mark.parametrize("edge,N,period", [("zigzag", 30, (1.0, 0.0)), ("armchair", 15, (0.0, S3))])
def test_triangular_ribbon_bands_lie_in_bulk_projection(edge, N, period):
    def ev(k):
        return np.array([-2 * sum(np.cos(k @ RV[j]) for j in (1, 2, 3))])
    assert _share_outside(triangular_ribbon(N, edge), ev, np.array(period), 1) == 0.0


def test_phosphorene_wide_armchair_gap_approaches_bulk():
    from tbribbon.lattices import phosphorene_ribbon
    h = phosphorene_ribbon(30, "armchair")
    e = np.concatenate([np.linalg.eigvalsh(h.H0 + h.H1 * np.exp(1j * th) + h.H1.conj().T * np.exp(-1j * th))
                        for th in np.linspace(-np.pi, np.pi, 201)])
    gap_ev = (e[e > 0].min() - e[e < 0].max()) * 3.665
    assert 1.52 <= gap_ev < 1.60

