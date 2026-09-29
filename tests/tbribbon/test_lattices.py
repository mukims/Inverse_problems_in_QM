import numpy as np
import pytest
from tbribbon.bands import band_edges, open_channels
from tbribbon.lattices import honeycomb_ribbon, square_strip


@pytest.mark.parametrize("N,edge", [(7, "armchair"), (9, "armchair"), (6, "zigzag"), (14, "zigzag")])
def test_honeycomb_has_2N_sites_and_is_hermitian(N, edge):
    h = honeycomb_ribbon(N, edge)
    assert h.H0.shape == (2 * N, 2 * N)
    assert np.allclose(h.H0, h.H0.conj().T)
    coord = (np.abs(h.H0) > 0).sum(1) + (np.abs(h.H1) > 0).sum(1) + (np.abs(h.H1) > 0).sum(0)
    assert coord.max() == 3 and coord.min() == 2           # bulk 3 neighbours, edges 2


@pytest.mark.parametrize("N", [7, 9, 14])
def test_armchair_band_top_matches_analytic(N):
    # N-AGNR bands: E = +-t|1 + 2cos(p pi/(N+1)) e^{..}|, so the top is t(1 + 2cos(pi/(N+1))) <= 3t
    top = band_edges(*honeycomb_ribbon(N, "armchair")[:2])[1]
    assert top == pytest.approx(1 + 2 * np.cos(np.pi / (N + 1)), abs=1e-3)


def test_hbn_onsite_opens_gap():
    h = honeycomb_ribbon(9, "armchair", onsite_a=0.8, onsite_b=-0.8)
    assert open_channels(h.H0, h.H1, [0.0])[0] == 0


def test_square_strip_channels_match_analytic():
    h = square_strip(10)
    eps = -2 * np.cos(np.arange(1, 11) * np.pi / 11)          # transverse modes (hopping -t)
    for E in (0.3, 1.1, 2.5, 3.5):
        assert open_channels(h.H0, h.H1, [E])[0] == np.sum(np.abs(E - eps) < 2)
