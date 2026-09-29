import os
import sys

import numpy as np
import pytest
from tbribbon.bands import band_edges, open_channels
from tbribbon.disorder import impurity_shifts
from tbribbon.lattices import honeycomb_ribbon, square_strip
from tbribbon.leads import LeadCache
from tbribbon.transport import spectrum

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.parametrize("h", [square_strip(10), honeycomb_ribbon(7, "armchair"), honeycomb_ribbon(6, "zigzag")])
def test_pristine_equals_open_channels_away_from_edges(h):
    E = np.linspace(0.05, 3.95, 40)
    leads = LeadCache(h.H0, h.H1, E)
    T = spectrum(h.H0, h.H1, E, np.zeros((20, h.H0.shape[0])), leads)
    ch = open_channels(h.H0, h.H1, E)
    stable = np.array([len(set(open_channels(h.H0, h.H1, [e - 0.02, e, e + 0.02]))) == 1 for e in E])
    assert np.allclose(T[stable], ch[stable], atol=1e-3)
    assert np.all(T[E > band_edges(h.H0, h.H1)[1] + 0.02] < 1e-6)


def test_impurity_sites_are_nested_prefixes_of_one_seed():
    a = impurity_shifts(100, 10, 20, seed=3, v=0.5)
    b = impurity_shifts(100, 10, 40, seed=3, v=0.5)
    assert np.all(b[a > 0] == 0.5) and (a > 0).sum() == 20 and (b > 0).sum() == 40


def test_legacy_trace_reproduces_ca_sq_exactly():
    sys.path.insert(0, os.path.join(REPO, "notebooks", "square_lattice"))
    import ca_sq
    legacy = np.load(os.path.expanduser("~/transmissions_sq/leads/leads_10.npy"))
    h = square_strip(10)
    E = np.array([0.37, 1.21, 2.84])
    shifts = impurity_shifts(100, 10, 30, seed=7, v=0.5)

    class Legacy:                                   # the legacy code uses the same surface GF on both sides
        gL = gR = [legacy[int(round(e * 100))] for e in E]

    ours = spectrum(h.H0, h.H1, E, shifts, Legacy, formula="legacy_trace")
    ref = [ca_sq.device(e, 1e-3, 1.0, 0.0, 10, 7, 30, leads=legacy) for e in E]
    assert np.allclose(ours, ref, rtol=1e-8)


def test_idealised_7agnr_pristine_matches_legacy_file():
    h = honeycomb_ribbon(7, "armchair")
    E = np.arange(300) * 0.01
    leg = np.load(os.path.join(REPO, "7_agnr_pris.npy"))[:300]
    T = spectrum(h.H0, h.H1, E, np.zeros((20, 14)), LeadCache(h.H0, h.H1, E))
    stable = np.array([len(set(open_channels(h.H0, h.H1, [e - 0.02, e, e + 0.02]))) == 1 for e in E])
    assert np.allclose(T[stable], leg[stable], atol=1e-2)


def test_legacy_trace_rejects_non_symmetric_honeycomb_h1():
    h = honeycomb_ribbon(7, "armchair")
    E = np.array([0.5])
    leads = LeadCache(h.H0, h.H1, E)
    shifts = np.zeros((1, 14))
    with pytest.raises(ValueError, match="symmetric H1"):
        spectrum(h.H0, h.H1, E, shifts, leads, formula="legacy_trace")

