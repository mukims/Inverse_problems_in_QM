"""Simulate one prior draw (spec section 5) and decide whether to reject it."""
import numpy as np

from atlaslib.spec import InputSpec
from tbribbon.bands import open_channels
from tbribbon.disorder import impurity_shifts
from tbribbon.leads import LeadCache
from tbribbon.parent import (A1, A2, BASIS_TYPE, active_atoms_per_cell, bulk_hamiltonian, device_cells_for,
                             parent_ribbon)
from tbribbon.transport import spectrum

V4 = InputSpec(version="v4")
E_WINDOW = 8.3            # eV: the part of the v4 window that band weights and visibility look at
IMPURITY_V = 0.5          # model units (scales with s), the spec's canonical form
LEAD_ETA = 1.2e-5
MIN_WEIGHT = 0.01         # a type with less band weight than this is invisible in the spectrum
MIN_CHANNELS = 10         # fewer non-zero channels than this is not a usable spectrum


def energies_model(s):
    """The v4 grid (eV) in model units of the host: E_eV / s. 416 values."""
    return V4.energies_t() / s


def band_weights(params, n_k=60):
    """Share of the bulk states in the window 0 <= E*s <= 8.3 eV that sits on each active site type.

    Diagonalises the bulk Hamiltonian on an n_k x n_k grid of the Brillouin zone, keeps the window's eigenstates,
    sums |psi|^2 over each type's sites and averages over the kept states; the values sum to 1. If no state lies
    in the window (rare edge case) the average runs over all states of the full band instead.
    """
    a = np.array([A1, A2], float)
    b = 2 * np.pi * np.linalg.inv(a).T                                   # rows b_i with b_i . a_j = 2 pi delta_ij
    f = np.arange(n_k) / n_k
    k = f[:, None, None, None] * b[0] + f[None, :, None, None] * b[1]    # (n_k, n_k, 1, 2)
    k = k.reshape(-1, 2)
    E, psi = np.linalg.eigh(bulk_hamiltonian(params, k))                 # (N, m), (N, m, m): columns are states
    types = np.array([t for t in BASIS_TYPE if t in params.active_types])
    keep = (E * params.s >= 0) & (E * params.s <= E_WINDOW)
    if not keep.any():
        keep = np.ones_like(keep)
    prob = np.abs(psi) ** 2                                              # (N, site, state)
    out = {}
    for t in params.active_types:
        on_t = prob[:, types == t, :].sum(axis=1)                        # (N, state)
        out[t] = float(on_t[keep].mean())
    total = sum(out.values())
    return {t: w / total for t, w in out.items()}


def clean_channels(draw):
    """Open channels of the clean ribbon at each of the 416 v4 energies (integers)."""
    rb = parent_ribbon(draw.params, draw.edge, draw.width)
    return np.rint(open_channels(rb.H0, rb.H1, energies_model(draw.params.s))).astype(int)


def n_impurities(draw, n_cells):
    """max(1, round(c * active atoms in the device))."""
    return max(1, round(draw.density * n_cells * active_atoms_per_cell(draw.params, draw.edge, draw.width)))


def simulate_raw(draw, n_imp, seed, n_cells=None):
    """Raw Caroli transmission (416 values) of the disordered device on the v4 grid; n_cells None = device_cells_for."""
    if n_cells is None:
        n_cells = device_cells_for(draw.params, draw.edge)
    rb = parent_ribbon(draw.params, draw.edge, draw.width)
    E = energies_model(draw.params.s)
    shifts = impurity_shifts(n_cells, rb.H0.shape[0], n_imp, seed, IMPURITY_V)
    return spectrum(rb.H0, rb.H1, E, shifts, LeadCache(rb.H0, rb.H1, E, eta=LEAD_ETA))


def to_input(raw):
    """The v4 network input (float32, 416 values in [0, 1])."""
    return V4.to_input(raw, V4.energies_t())[0]


def reject_reason(draw, clean, raw, weights):
    """Why this draw should not enter the data set, or None."""
    if np.sum(clean) == 0:
        return "no_channel"
    if min(weights.values()) < MIN_WEIGHT:
        return "invisible_type"
    if np.count_nonzero(np.round(raw, 3)) < MIN_CHANNELS:
        return "few_channels"
    return None
