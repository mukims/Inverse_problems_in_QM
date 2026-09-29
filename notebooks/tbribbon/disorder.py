import numpy as np


def impurity_shifts(n_cells, sites_per_cell, n_impurities, seed, v):
    """Project convention: RandomState(seed).choice(..., replace=False), so larger counts extend
    the same seed's set (nested); site index = cell * sites_per_cell + orbital."""
    idx = np.random.RandomState(seed).choice(n_cells * sites_per_cell, n_impurities, replace=False)
    shifts = np.zeros(n_cells * sites_per_cell)
    shifts[idx] = v
    return shifts.reshape(n_cells, sites_per_cell)
