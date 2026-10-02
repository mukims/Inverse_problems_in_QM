# tests/tbribbon/test_disorder_atoms.py
import numpy as np
import pytest

from tbribbon.disorder import impurity_shifts
from tbribbon.materials import make_model


def _legacy(n_cells, spc, n_imp, seed, v):
    idx = np.random.RandomState(seed).choice(n_cells * spc, n_imp, replace=False)
    s = np.zeros(n_cells * spc)
    s[idx] = v
    return s.reshape(n_cells, spc)


def test_single_orbital_draw_is_unchanged():
    for seed in range(5):
        assert np.array_equal(impurity_shifts(100, 14, 28, seed, 0.5), _legacy(100, 14, 28, seed, 0.5))


def test_multi_orbital_impurity_shifts_whole_atoms():
    s = impurity_shifts(100, 3 * 9, 20, seed=3, v=0.25, orbitals_per_site=3).reshape(-1, 3)
    per_atom = (s != 0).sum(1)
    assert set(per_atom.tolist()) <= {0, 3} and (per_atom == 3).sum() == 20


def test_multi_orbital_draw_is_nested_across_densities():
    lo = impurity_shifts(100, 27, 10, seed=7, v=1.0, orbitals_per_site=3) != 0
    hi = impurity_shifts(100, 27, 40, seed=7, v=1.0, orbitals_per_site=3) != 0
    assert np.all(hi[lo])


def test_orbital_count_must_divide_sites():
    with pytest.raises(ValueError):
        impurity_shifts(100, 10, 5, seed=0, v=1.0, orbitals_per_site=3)


def test_mos2_counts_atoms_and_uses_spec_impurity_strength():
    m = make_model("mos2", "zigzag", 9)
    assert m.orbitals_per_site == 3 and m.n_sites == 100 * 9
    assert m.impurity_v_t == pytest.approx(0.2535)
    assert make_model("hbn", "zigzag", 9).orbitals_per_site == 1
