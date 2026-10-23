import numpy as np, pytest
from tbribbon.parent import BASIS, BASIS_TYPE, parent_ribbon, bulk_hamiltonian, match_sites, named_material, HexParams

S3 = np.sqrt(3)
@pytest.mark.parametrize("i,j,d", [(1, 2, 1.0), (0, 1, 1.0), (0, 4, S3 / 2), (3, 4, S3 / 2), (1, 3, 0.5)])
def test_basis_distances_of_section_4_1(i, j, d):
    assert np.linalg.norm(np.subtract(BASIS[i], BASIS[j])) == pytest.approx(d, abs=1e-12)

def test_each_kagome_site_has_four_kagome_neighbours_and_each_centre_six():
    p = HexParams(on_C=True, on_H=False, on_K=True, lead="K", s=1.0, t_KK=1, t_CC=0.5, t_CK=0.3)
    h = parent_ribbon(p, "zigzag", 12)
    # bonds inside the cell and to both neighbours, counted per block: the zigzag period sqrt3 is twice the K-K spacing,
    # so K2 bonds to K3 of its own cell in H0 and again to K3 of the next cell in H1
    links = sum((np.abs(B) > 1e-12).astype(int) for B in (h.H0, h.H1, h.H1.conj().T))
    np.fill_diagonal(links, 0)
    y = h.positions[:, 0]                                                    # zigzag: x runs across the ribbon
    mid = (y > np.percentile(y, 30)) & (y < np.percentile(y, 70))           # interior sites only
    is_k, is_c = h.sublattice >= 3, h.sublattice == 0
    assert np.all(links[np.ix_(mid & is_k, is_k)].sum(1) == 4)              # t_KK bonds
    assert np.all(links[np.ix_(mid & is_c, is_k)].sum(1) == 6)              # t_CK bonds

def test_ribbon_is_hermitian_and_sized_by_active_sites():
    p = HexParams(on_C=True, on_H=True, on_K=False, lead="H", s=1.0, t_AB=1, t_CC=0.4, t_CH=0.2)
    h = parent_ribbon(p, "armchair", 6)
    assert np.allclose(h.H0, h.H0.conj().T) and h.H0.shape[0] == 6 * 3      # 1 C + 2 H per cell across
    assert set(np.unique(h.sublattice)) == {0, 1, 2}

def test_inactive_types_create_no_sites():
    h = parent_ribbon(named_material("graphene-ideal"), "zigzag", 5)
    assert set(np.unique(h.sublattice)) == {1, 2} and h.H0.shape[0] == 10

def test_match_sites_recovers_a_permutation_under_rotation_and_scale():
    rng = np.random.default_rng(0); ref = rng.normal(size=(9, 2))
    c, s_ = np.cos(0.5), np.sin(0.5); R = np.array([[c, -s_], [s_, c]])
    perm = rng.permutation(9); pos = (ref @ R.T * 1.7 + 3.0)[np.argsort(perm)]
    assert np.allclose(pos[match_sites(pos, ref)], ref @ R.T * 1.7 + 3.0)

def test_match_sites_modulo_a_period_recovers_the_period_shifts():
    rng = np.random.default_rng(1)
    ref = np.column_stack([rng.uniform(0, 3, 12), rng.uniform(-5, 5, 12)])     # one cell of a strip, period (3, 0)
    c, s_ = np.cos(1.2), np.sin(1.2); R = np.array([[c, -s_], [s_, c]])
    period = R @ np.array([3.0, 0.0]) * 0.8                                   # the period in pos's frame
    true = ref @ R.T * 0.8 + 2.0                                              # ref seen in pos's frame
    moved = np.zeros(12, int); moved[[1, 4, 7]] = [1, -1, 2]                  # these sites sit in other period images
    order = rng.permutation(12)
    pos = (true + moved[:, None] * period)[order]
    perm, shift = match_sites(pos, ref, period=period)
    assert np.array_equal(order[perm], np.arange(12)) and np.array_equal(shift, -moved)
    assert np.allclose(pos[perm] + shift[:, None] * period, true)

def test_bulk_graphene_bands_equal_the_honeycomb_lattice_data():
    from tbribbon.lattice2d import HONEYCOMB, bulk_hamiltonian as lattice_bulk
    for k in np.random.default_rng(2).normal(size=(5, 2)):
        assert np.allclose(np.linalg.eigvalsh(bulk_hamiltonian(named_material("graphene-ideal"), k)),
                           np.linalg.eigvalsh(lattice_bulk(HONEYCOMB, k)))

def test_atoms_per_cell_and_device_length():
    from tbribbon.parent import active_atoms_per_cell, device_cells_for
    g = named_material("graphene-ideal")
    assert active_atoms_per_cell(g, "armchair", 10) == 20 and device_cells_for(g, "zigzag") == 125
