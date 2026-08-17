#!/usr/bin/env python
"""
agnr_lib.py -- clean, self-contained AGNR physics core.

Consolidates the transmission machinery that is currently duplicated (and
subtly divergent) across ``agnr.py``, ``ca_agnr.py`` and ``generate_test_data.py``
into one module with **no global lead state** -- leads are always passed in.

Convention note (important)
---------------------------
The three existing copies disagree on the non-local cross Green's function:

    agnr.py / generate_test_data.py :  Gnonlocal = left @ rho @ IR   <- test data
    ca_agnr.py                      :  Gnonlocal = left @ rho @ IL   <- training data

and on the broadening used to generate data (test: d=1e-4, train: d=1e-5).
Rather than silently picking one, ``device_transmission`` takes a ``nonlocal_mode``
argument ("IR" or "IL") so either convention can be reproduced exactly and the
two can be compared. Default is "IR" (the convention used for the held-out test
set in ``data/test/transmission_results``).

Energy grid: w in [0, 3) step 0.01 -> 300 points (AGNR convention).
Device: 100 unit cells; impurity = on-site shift to (w + i d - 0.5).
"""

import os
from functools import lru_cache

import numpy as np

W_MIN, W_MAX, W_STEP = 0.0, 3.0, 0.01
NCELLS = 100

# --- physical defaults for this study (set by project decision) ---
D_DEFAULT = 1e-5          # imaginary broadening eta
T_DEFAULT = 1.0           # hopping amplitude
IMPURITY_POTENTIAL = 0.5  # impurity on-site potential; enters as (w + i d - V)


def energy_grid() -> np.ndarray:
    return np.arange(W_MIN, W_MAX, W_STEP)


# ----------------------------------------------------------------------
# Hamiltonian building blocks (m = AGNR width; unit cell has 2m sites)
# ----------------------------------------------------------------------
def unitcell(w, d, t, e, m):
    """Pristine AGNR unit cell (2m x 2m) with anti-diagonal couplings."""
    m = int(m)
    base = (w + 1j * d) * np.eye(2 * m, dtype=complex)
    idy = np.arange(0, 2 * m - 1)
    base[idy, idy + 1] = t
    base[idy + 1, idy] = t
    idx = np.arange(0, m, 2)
    base[idx, 2 * m - 1 - idx] = t
    base[2 * m - 1 - idx, idx] = t
    return base


def beta_matrix(w_vals, d, t, e, m):
    """Batched (over energy) pristine unit cell -> (B, 2m, 2m)."""
    m = int(m)
    w = np.asarray(w_vals)[:, None, None]
    base = (w + 1j * d) * np.eye(2 * m)[None, :, :]
    base = base.astype(complex)
    idy = np.arange(0, 2 * m - 1)
    base[:, idy, idy + 1] = t
    base[:, idy + 1, idy] = t
    idx = np.arange(0, m, 2)
    base[:, idx, 2 * m - 1 - idx] = t
    base[:, 2 * m - 1 - idx, idx] = t
    return base


def T1_matrix(t, m):
    """Inter-cell hopping T (cell i -> i+1)."""
    m = int(m)
    dim = 2 * m
    T = np.zeros((dim, dim), dtype=np.complex128)
    n = np.arange(1, (m - 1) // 2 + 1)
    T[2 * n - 1, 2 * m - 2 * n] = t
    return T


def rho_matrix(t, m):
    """Contact operator rho used in the Landauer trace."""
    m = int(m)
    dim = 2 * m
    rho = np.zeros((dim, dim), dtype=complex)
    for n in range(1, (m - 1) // 2 + 1):
        rho[2 * n - 1, 2 * n - 1] = t
    return rho


# ----------------------------------------------------------------------
# Lead surface Green's function (Sancho-Rubio; see compute_leads_sq notes)
# ----------------------------------------------------------------------
def leads_sancho_rubio(w_vals, d, t, e, m, tol=1e-10, max_iter=200):
    """AGNR lead surface GF for all energies at once, via decimation.

    Same scheme as notebooks/square_lattice/compute_leads_sq.py, but with the
    AGNR unit cell and the AGNR inter-cell hopping T1 (which is *not* the
    identity, so the left/right renormalised hoppings genuinely differ).
    """
    m = int(m)
    B = len(w_vals)
    unit = beta_matrix(w_vals, d, t, e, m)          # E - h
    T = T1_matrix(t, m)

    W = unit.copy()
    Ws = unit.copy()
    a = np.broadcast_to(T.conj().T, (B, 2 * m, 2 * m)).copy()
    b = np.broadcast_to(T, (B, 2 * m, 2 * m)).copy()

    conv = np.inf
    count = 0
    while conv > tol and count < max_iter:
        g = np.linalg.inv(W)
        a_gb = a @ g @ b
        b_ga = b @ g @ a
        Ws = Ws - a_gb
        W = W - a_gb - b_ga
        a = a @ g @ a
        b = b @ g @ b
        conv = max(np.max(np.abs(a)), np.max(np.abs(b)))
        count += 1
    return np.linalg.inv(Ws), count


def load_leads(m, leads_dir="~/Desktop/backup/agnr"):
    """Load precomputed AGNR leads for width m -> (300, 2m, 2m)."""
    p = os.path.expanduser(os.path.join(leads_dir, f"size_{m}", f"leads_{m}.npy"))
    arr = np.load(p)
    # some saved files are (G, iters) tuples flattened to object/extra dim
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


# ----------------------------------------------------------------------
# Impurity configuration (seeded, cached)
# ----------------------------------------------------------------------
DEVICE_COMBS = {}


def _get_device_combs(width: int) -> np.ndarray:
    """(cell, site) grid: 100 cells x 2m sites."""
    width = int(2 * width)
    if width not in DEVICE_COMBS:
        DEVICE_COMBS[width] = np.stack(
            np.meshgrid(np.arange(NCELLS), np.arange(width), indexing="ij"),
            axis=-1).reshape(-1, 2)
    return DEVICE_COMBS[width]


@lru_cache(maxsize=4096)
def chosen_for_config(n: int, width: int, config: int) -> np.ndarray:
    combs = _get_device_combs(int(width))
    rng = np.random.RandomState(int(config))
    return combs[rng.choice(len(combs), size=int(n), replace=False)]


def possible_combs(n: int, width: int):
    def combs_for_seed(seed: int):
        return chosen_for_config(int(n), int(width), int(seed))
    return combs_for_seed


def unidevice(w, d, t, e, m, config, n, cell, combs_fn=None,
              impurity_potential=IMPURITY_POTENTIAL):
    """Unit cell `cell` with its impurities inserted.

    An impurity site's diagonal becomes (w + i d - V) with V the impurity
    on-site potential (default 0.5).
    """
    m = int(m)
    if combs_fn is None:
        combs_fn = possible_combs(int(n), m)
    mat = unitcell(w, d, t, e, m)
    if int(n) == 0:
        return mat
    imps = combs_fn(int(config))
    mask = imps[:, 0] == int(cell)
    if not np.any(mask):
        return mat
    idx = imps[mask, 1]
    mat[idx, idx] = (w + 1j * d - impurity_potential)
    return mat


# ----------------------------------------------------------------------
# Transmission
# ----------------------------------------------------------------------
def device_transmission(w, d, t, e, m, config, concentration, leads,
                        nonlocal_mode="IR", impurity_potential=IMPURITY_POTENTIAL):
    """Landauer transmission at one energy. `leads` is the (300, 2m, 2m) array.

    nonlocal_mode: "IR" (agnr.py / test data) or "IL" (ca_agnr.py / train data).
    """
    m = int(m)
    dim = 2 * m
    I = np.eye(dim, dtype=complex)
    ene = int(round(w * 100))

    left = leads[ene]
    tin = T1_matrix(t, m)
    tin_d = tin.T
    rho = rho_matrix(t, m)

    combs_fn = possible_combs(int(concentration), m)
    g_new = left
    for i in range(NCELLS):
        unit_i = unidevice(w, d, t, e, m, config, concentration, i, combs_fn=combs_fn,
                           impurity_potential=impurity_potential)
        gd = np.linalg.inv(unit_i)
        g_new = np.linalg.solve(I - gd @ tin_d @ g_new @ tin, gd)

    left_device = g_new
    IL = np.linalg.solve(I - left_device @ rho @ left @ rho, left_device)
    IR = np.linalg.solve(I - left @ rho @ left_device @ rho, left)

    gdd = IL - IL.conj().T
    grr = IR - IR.conj().T
    Gnonlocal = left @ rho @ (IR if nonlocal_mode == "IR" else IL)
    GNON = Gnonlocal - Gnonlocal.conj().T

    term1 = gdd @ rho @ grr @ rho
    term2 = rho @ GNON @ rho @ GNON
    return float(np.abs(np.trace(term1 - term2)))


def spectrum(m, leads, config=0, concentration=0, d=D_DEFAULT, t=T_DEFAULT, e=0.0,
             nonlocal_mode="IR", impurity_potential=IMPURITY_POTENTIAL):
    """Full 300-point transmission spectrum."""
    return np.array([device_transmission(w, d, t, e, m, config, concentration,
                                         leads, nonlocal_mode, impurity_potential)
                     for w in energy_grid()])


# ----------------------------------------------------------------------
# Band-gap estimation from a spectrum
# ----------------------------------------------------------------------
def band_gap(spec, thresh=0.05, w=None, persist=3, skip_first=1, rel_to="median"):
    """Estimate the transport gap: the low-transmission region starting at w=0.

    The threshold is taken relative to a ROBUST level (the median of the
    spectrum) rather than max(spec): sharp impurity resonances push the max to
    ~1e2, which would make a max-relative threshold wildly overestimate the gap.
    A purely absolute threshold fails too, because impurities leak a small but
    non-zero sub-gap transmission (~1e-3).

    `skip_first` drops leading points (default: index 0). w=0 sits exactly on a
    band edge, where the lead Green's function is numerically ill-conditioned and
    the saved data carries a spurious spike (T ~ 0.4 against ~1e-5 in the true
    gap). Including it collapses every gap estimate to zero.

    `persist` requires the spectrum to stay above threshold for that many
    consecutive points, so an isolated spike does not terminate the gap.

    Returns (gap_edge_energy, first_index_above_threshold).
    """
    if w is None:
        w = energy_grid()
    spec = np.asarray(spec, dtype=float)

    body = spec[skip_first:]
    if body.size == 0:
        return float(w[-1]), len(spec)

    if rel_to == "median":
        level = float(np.median(body))
    elif rel_to == "max":
        level = float(np.max(body))
    else:                      # absolute
        level = 1.0
    cut = thresh * level if level > 0 else thresh

    above = body > cut
    if not above.any():
        return float(w[-1]), len(spec)

    if persist > 1:
        k = np.convolve(above.astype(int), np.ones(persist, dtype=int), mode="valid")
        hits = np.where(k == persist)[0]
        i0 = int(hits[0]) if len(hits) else int(np.where(above)[0][0])
    else:
        i0 = int(np.where(above)[0][0])
    i0 += skip_first
    return float(w[i0]), i0
