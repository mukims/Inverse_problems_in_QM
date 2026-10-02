import numpy as np


def impurity_shifts(n_cells, sites_per_cell, n_impurities, seed, v, orbitals_per_site=1):
    """Project convention: RandomState(seed).choice(..., replace=False), so larger counts extend the same
    seed's set (nested). Impurities act on atoms: each shifts all orbitals_per_site orbitals of one atom
    (orbital index = atom * orbitals_per_site + o). For orbitals_per_site = 1 the draw is unchanged."""
    if sites_per_cell % orbitals_per_site:
        raise ValueError(f"sites_per_cell ({sites_per_cell}) is not a multiple of orbitals_per_site ({orbitals_per_site})")
    n_atoms = n_cells * sites_per_cell // orbitals_per_site
    atoms = np.random.RandomState(seed).choice(n_atoms, n_impurities, replace=False)
    shifts = np.zeros(n_cells * sites_per_cell)
    for o in range(orbitals_per_site):
        shifts[atoms * orbitals_per_site + o] = v
    return shifts.reshape(n_cells, sites_per_cell)
