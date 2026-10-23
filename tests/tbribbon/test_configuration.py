import numpy as np
import pytest

from tbribbon.configuration import (
    device_atom_coordinates,
    device_host_onsite,
    device_onsite_profile,
    period_vector,
    reconstruct_configuration,
)
from tbribbon.disorder import impurity_shifts
from tbribbon.materials import MATERIALS, make_model, hamiltonian_for
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "notebooks" / "agnr" / "physics"))
import agnr_lib


@pytest.mark.parametrize("material", ["graphene-ideal", "hbn", "phosphorene", "mos2", "silicene", "germanene", "triangular", "kagome"])
@pytest.mark.parametrize("edge", ["armchair", "zigzag"])
def test_period_vector_and_coordinates(material, edge):
    m = make_model(material, edge, 7)
    T = period_vector(material, edge)
    assert T.shape == (2,) and np.linalg.norm(T) > 0.5
    coords = device_atom_coordinates(m, n_cells=10)
    assert coords.shape == (10 * (m.sites_per_cell // m.orbitals_per_site), 2)
    assert np.all(np.isfinite(coords))


def test_square_and_strip_lattices():
    m_sq = make_model("square", "strip", 10)
    T_sq = period_vector("square", "strip")
    assert np.allclose(T_sq, [1.0, 0.0])
    coords_sq = device_atom_coordinates(m_sq, n_cells=5)
    assert coords_sq.shape == (50, 2)

    for mat in ["checkerboard", "lieb"]:
        m = make_model(mat, "strip", 6)
        T = period_vector(mat, "strip")
        assert T.shape == (2,) and np.linalg.norm(T) > 0.5
        coords = device_atom_coordinates(m, n_cells=4)
        assert coords.shape[0] == 4 * m.sites_per_cell


def test_impurity_shifts_exact_match():
    m = make_model("hbn", "armchair", 7)
    n_cells = 50
    n_imp = 20
    for seed in [0, 42, 999]:
        cfg = reconstruct_configuration(m, seed=seed, n_impurities=n_imp, n_cells=n_cells)
        assert len(cfg.atom_indices) == n_imp
        assert np.array_equal(cfg.atom_indices, np.random.RandomState(seed).choice(cfg.n_atoms, n_imp, replace=False))

        # Check against tbribbon disorder.impurity_shifts
        s = impurity_shifts(n_cells, m.sites_per_cell, n_imp, seed, m.impurity_v_t, m.orbitals_per_site)
        shifted_cells, shifted_sites = np.where(s != 0)
        assert np.array_equal(np.sort(cfg.cell_indices * m.sites_per_cell + cfg.site_indices),
                             np.sort(shifted_cells * m.sites_per_cell + shifted_sites))


def test_agnr_lib_exact_match():
    for width in [7, 9]:
        m = make_model("graphene-ideal", "armchair", width)
        for seed in [1, 7, 100]:
            n_imp = 14
            cfg = reconstruct_configuration(m, seed=seed, n_impurities=n_imp, n_cells=100)
            legacy_combs = agnr_lib.chosen_for_config(n_imp, width, seed)
            cfg_combs = np.column_stack([cfg.cell_indices, cfg.site_indices])
            assert np.array_equal(cfg_combs, legacy_combs)


def test_host_onsite_and_impurity_onsite_relations():
    # hBN armchair: Boron (+3.625 eV) and Nitrogen (-3.625 eV)
    m = make_model("hbn", "armchair", 7)
    cfg = reconstruct_configuration(m, seed=42, n_impurities=10, n_cells=50)
    assert np.all(np.isclose(np.abs(cfg.host_onsite_ev), 3.625))
    assert np.allclose(cfg.host_onsite_ev, cfg.host_onsite_t * m.t_ev)
    assert np.allclose(cfg.impurity_onsite_t, cfg.host_onsite_t + cfg.defect_v_t)
    assert np.allclose(cfg.impurity_onsite_ev, cfg.host_onsite_ev + cfg.defect_v_ev)

    # Multi-orbital MoS2: 3 orbitals per site
    m_mos2 = make_model("mos2", "zigzag", 7)
    cfg_mos2 = reconstruct_configuration(m_mos2, seed=123, n_impurities=8, n_cells=30)
    assert cfg_mos2.host_onsite_t.shape == (8, 3)
    assert cfg_mos2.impurity_onsite_t.shape == (8, 3)
    assert np.allclose(cfg_mos2.impurity_onsite_t, cfg_mos2.host_onsite_t + cfg_mos2.defect_v_t)


def test_nesting_property():
    m = make_model("triangular", "zigzag", 8)
    lo = reconstruct_configuration(m, seed=77, density=0.01, n_cells=100)
    hi = reconstruct_configuration(m, seed=77, density=0.04, n_cells=100)
    assert set(lo.atom_indices).issubset(set(hi.atom_indices))


def test_device_onsite_profile():
    m = make_model("phosphorene", "armchair", 7)
    prof = device_onsite_profile(m, seed=15, n_impurities=12, n_cells=20, unit="eV")
    assert prof.shape == (20, m.sites_per_cell)
    # Check that pristine sites match host onsite in eV
    h = hamiltonian_for(m)
    h0_ev = np.diag(h.H0.real) * m.t_ev
    v_ev = m.impurity_v_t * m.t_ev
    cfg = reconstruct_configuration(m, seed=15, n_impurities=12, n_cells=20)
    for c, s in zip(cfg.cell_indices, cfg.site_indices):
        assert pytest.approx(prof[c, s]) == h0_ev[s] + v_ev
