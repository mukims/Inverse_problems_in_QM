import numpy as np, pytest
from tbribbon.parent import COUNTERPARTS, named_material, parent_ribbon, reference_ribbon, match_sites, device_cells_for
from tbribbon.parent import A1, A2, EDGE_VECTORS, match_fit, reference_period
from tbribbon.leads import LeadCache
from tbribbon.transport import spectrum
from tbribbon.disorder import impurity_shifts
from tbribbon.materials import device_cells

E = np.linspace(0.05, 2.95, 30)          # model units
CASES = sorted(COUNTERPARTS.items())

def _clean(h, n_cells=6):
    return spectrum(h.H0, h.H1, E, np.zeros((n_cells, h.H0.shape[0])), LeadCache(h.H0, h.H1, E, eta=1.2e-5))

def _period(edge):
    T, _ = EDGE_VECTORS[edge]
    return T[0] * A1 + T[1] * A2

def test_the_required_counterparts_are_listed():
    keys = set(COUNTERPARTS)
    assert {("graphene-ideal", "armchair"), ("graphene-ideal", "zigzag"), ("hbn", "armchair"), ("hbn", "zigzag")} <= keys
    assert ("kagome", "zigzag") in COUNTERPARTS.values() and any(v[0] == "triangular" for v in COUNTERPARTS.values())

@pytest.mark.parametrize("key,ref", CASES)
@pytest.mark.parametrize("w", [5, 10])
def test_sites_match_the_existing_builder_with_no_residual(key, ref, w):
    name, edge = key
    hp, hr = parent_ribbon(named_material(name), edge, w), reference_ribbon(*ref, w)
    perm, shift = match_sites(hp.positions, hr.positions, period=_period(edge), ref_period=reference_period(*ref))
    assert match_fit(hp.positions, hr.positions, perm, shift, _period(edge))[2] < 1e-6

@pytest.mark.parametrize("key,ref", CASES)
@pytest.mark.parametrize("w", [5, 10])
def test_clean_spectra_equal_the_existing_builder(key, ref, w):
    name, edge = key
    assert np.allclose(_clean(parent_ribbon(named_material(name), edge, w)), _clean(reference_ribbon(*ref, w)), atol=1e-10)

@pytest.mark.parametrize("key,ref", CASES)
def test_disordered_spectra_equal_when_the_same_atoms_are_disordered(key, ref):
    name, edge = key
    hp, hr = parent_ribbon(named_material(name), edge, 10), reference_ribbon(*ref, 10)
    # Ruling R4: reference site j of cell c is parent site perm[j] of cell c + shift[j]
    perm, shift = match_sites(hp.positions, hr.positions, period=_period(edge), ref_period=reference_period(*ref))
    n_cells, n = 6, hr.H0.shape[0]
    sr = np.zeros((n_cells, n))
    sr[1:-1] = impurity_shifts(n_cells - 2, n, 8, seed=3, v=0.5)              # interior device cells only
    sp = np.zeros_like(sr)
    for c in range(1, n_cells - 1):
        sp[c + shift, perm] = sr[c]
    assert np.allclose(spectrum(hp.H0, hp.H1, E, sp, LeadCache(hp.H0, hp.H1, E, eta=1.2e-5)),
                       spectrum(hr.H0, hr.H1, E, sr, LeadCache(hr.H0, hr.H1, E, eta=1.2e-5)), atol=1e-10)

@pytest.mark.parametrize("key,ref,cells", [(("graphene-ideal", "armchair"), "graphene-ideal", 125), (("hbn", "zigzag"), "hbn", 125),
                                           (("kagome", "zigzag"), "kagome", 83), (("triangular", "zigzag"), "triangular", 250)])
def test_device_length_matches_the_catalogue(key, ref, cells):
    assert key in COUNTERPARTS and COUNTERPARTS[key][1] == key[1]
    assert device_cells_for(named_material(key[0]), key[1]) == device_cells(ref, key[1], 2000) == cells
