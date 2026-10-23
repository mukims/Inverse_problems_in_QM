"""Labels that describe how impurities are arranged at a fixed count, computed from the configuration files
(cloud_<density>_config.npz), and the misfit used to score predictions of them."""
import numpy as np

TOL = 1e-3          # coordinates are stored as float32; device geometry is float64


def across_axis(model_id):
    """Column of the coordinates that runs across the ribbon (0 = x, 1 = y), from the ribbon's period vector."""
    from tbribbon.configuration import period_vector
    material, edge, _ = model_id.split("/")
    R = period_vector(material, edge)
    return int(abs(R[0]) >= abs(R[1]))          # the period points along the ribbon; the other column is across


def edge_fraction(coords, device_xy, across):
    """Fraction of each seed's impurities on the two outermost lines of atoms across the ribbon.

    coords: (n_seeds, n_impurities, 2) impurity positions; device_xy: (n_atoms, 2) every atom of the device;
    across: the column that runs across the ribbon (across_axis). Returns (n_seeds,). Both edges count alike, so the
    label is the same for a configuration and its mirror image.
    """
    lines = np.asarray(device_xy)[:, across]
    lo, hi = lines.min(), lines.max()
    c = np.asarray(coords, dtype=np.float64)[..., across]
    return np.mean(np.isclose(c, lo, atol=TOL) | np.isclose(c, hi, atol=TOL), axis=-1)


def misfit(pred, true):
    """Mean squared difference, the same form as the misfit of the reference-library baseline (mw_misfit.py)."""
    return float(np.mean((np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)) ** 2))


def _along(across):
    return 1 - across


def centre_distance(coords, device_xy, across):
    """Mean distance of each seed's impurities from the ribbon's centre line, as a share of the width: 0 on the
    centre line, 0.5 on an edge."""
    lines = np.asarray(device_xy)[:, across]
    u = (np.asarray(coords, dtype=np.float64)[..., across] - lines.min()) / (lines.max() - lines.min())
    return np.mean(np.abs(u - 0.5), axis=-1)


def sublattice_fraction(sublattice, value=0):
    """Share of each seed's impurities on one sublattice."""
    return np.mean(np.asarray(sublattice) == value, axis=-1)


def _distances(c):
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return d


def nearest_neighbour_distance(coords):
    """Mean, over each seed's impurities, of the distance to the closest other impurity (bond lengths)."""
    return np.array([_distances(np.asarray(c, dtype=np.float64)).min(1).mean() for c in coords])


def close_pairs(coords, cutoff=1.01):
    """Number of impurity pairs within `cutoff` bond lengths of each other (nearest neighbours by default)."""
    return np.array([int((_distances(np.asarray(c, dtype=np.float64)) < cutoff).sum() // 2) for c in coords])


def near_contact(coords, device_xy, across, share=0.1):
    """Share of each seed's impurities within `share` of the device length from either lead."""
    along = _along(across)
    pos = np.asarray(device_xy)[:, along]
    v = (np.asarray(coords, dtype=np.float64)[..., along] - pos.min()) / (pos.max() - pos.min())
    return np.mean(np.minimum(v, 1 - v) < share, axis=-1)


def laplacian_entropy(coords, length):
    """Von Neumann entropy of each seed's impurity graph.

    Impurities are nodes joined with weight exp(-d^2 / 2 length^2). With L = D - W the graph Laplacian and
    rho = L / tr L (unit trace, eigenvalues >= 0), S = -sum(lambda ln lambda). One linked pair gives 0; n equally linked
    impurities give ln(n - 1); graphs with no links at all give 0. Depends only on distances, so moving or mirroring
    the arrangement leaves it unchanged."""
    out = np.zeros(len(coords))
    for i, c in enumerate(coords):
        d = _distances(np.asarray(c, dtype=np.float64))
        W = np.exp(-(d ** 2) / (2.0 * length ** 2))                 # exp(-inf) = 0 on the diagonal
        L = np.diag(W.sum(1)) - W
        tr = np.trace(L)
        if tr <= 0:
            continue
        lam = np.linalg.eigvalsh(L / tr)
        lam = lam[lam > 1e-12]
        out[i] = float(-(lam * np.log(lam)).sum())
    return out


def descriptors(coords, sublattice, device_xy, across, lengths=(2.0, 5.0, 10.0)):
    """Every arrangement label for every seed, as {name: (n_seeds,) array}."""
    d = {"edge_frac": edge_fraction(coords, device_xy, across),
         "centre_dist": centre_distance(coords, device_xy, across),
         "sub0_frac": sublattice_fraction(sublattice),
         "nn_dist": nearest_neighbour_distance(coords),
         "close_pairs": close_pairs(coords),
         "near_contact": near_contact(coords, device_xy, across)}
    for length in lengths:
        d[f"graph_entropy_l{length:g}"] = laplacian_entropy(coords, length)
    return d
