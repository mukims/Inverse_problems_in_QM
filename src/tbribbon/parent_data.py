"""Simulate one prior draw (spec section 5) and decide whether to reject it."""
import numpy as np

from atlaslib.spec import InputSpec, despike
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


def host_leads(draw):
    """(ribbon, LeadCache) of the host of `draw`: the clean ribbon (parent_ribbon) and its lead self-energies on the v4
    grid. Both depend on the host only (parameters, edge, width), not on the disorder seed or the device length."""
    rb = parent_ribbon(draw.params, draw.edge, draw.width)
    return rb, LeadCache(rb.H0, rb.H1, energies_model(draw.params.s), eta=LEAD_ETA)


def simulate_raw(draw, n_imp, seed, n_cells=None, rb=None, lc=None):
    """Raw Caroli transmission (416 values) of the disordered device on the v4 grid; n_cells None = device_cells_for.

    rb, lc: the host's ribbon and LeadCache (host_leads(draw)); None builds them here. A caller simulating many seeds of
    one host passes them in to build them once; the result is bit-identical either way."""
    if n_cells is None:
        n_cells = device_cells_for(draw.params, draw.edge)
    if rb is None:
        rb = parent_ribbon(draw.params, draw.edge, draw.width)
    E = energies_model(draw.params.s)
    shifts = impurity_shifts(n_cells, rb.H0.shape[0], n_imp, seed, IMPURITY_V)
    if lc is None:
        lc = LeadCache(rb.H0, rb.H1, E, eta=LEAD_ETA)
    return spectrum(rb.H0, rb.H1, E, shifts, lc)


def to_input(raw):
    """The v4 network input (float32, 416 values in [0, 1])."""
    return V4.to_input(raw, V4.energies_t())[0]


def to_input_linear(raw, chunk=10_000):
    """The v4 input without the log scale: T itself in channels (float32, rows of 416), after the same label-free
    cleaning as to_input (round to 3 decimals, despike, clip to [0, V4.cap]). `raw` is one spectrum or a 2-D array of
    them on the v4 grid; rows are processed in chunks to keep the despike median small in memory."""
    raw = np.atleast_2d(np.asarray(raw, dtype=np.float64))
    if raw.shape[1] != V4.energies_t().size:
        raise ValueError(f"raw has {raw.shape[1]} channels, the v4 grid {V4.energies_t().size}")
    out = np.empty(raw.shape, dtype=np.float32)
    for i in range(0, len(raw), chunk):
        T = despike(np.round(raw[i:i + chunk], 3))
        out[i:i + chunk] = np.clip(np.round(T, 3), 0.0, V4.cap)
    return out


def bandtop_k(lin):
    """Per row of the linear input: the band top as a channel count (last non-zero channel + 1; the full window if
    none). Label-free: computed from T."""
    lin = np.atleast_2d(lin)
    nz = lin > 0
    last = lin.shape[1] - 1 - np.argmax(nz[:, ::-1], axis=1)
    return np.where(nz.any(axis=1), last + 1, lin.shape[1])


def resample_to_top(rows, k):
    """Each row's channels 0..k-1 linearly resampled onto as many points as the row has (float32)."""
    rows = np.atleast_2d(rows)
    n = rows.shape[1]
    out = np.empty(rows.shape, dtype=np.float32)
    pos = np.arange(n)
    for i, (row, ki) in enumerate(zip(rows, k)):
        out[i] = np.interp(np.linspace(0, ki - 1, n), pos[:ki], row[:ki])
    return out


def to_input_bandtop(raw):
    """Label-free scale normalisation for encoders without weight sharing (INV-MLP): the linear input from 0 up to the
    spectrum's own band top (its last non-zero channel after cleaning) resampled onto 416 points, plus the band top as a
    fraction of the window (k / 416) as value 417, so the energy scale is kept. Every host then fills the same length
    whatever its energy scale s. float32 rows of 417."""
    lin = to_input_linear(raw)
    k = bandtop_k(lin)
    return np.concatenate([resample_to_top(lin, k), (k / lin.shape[1]).astype(np.float32)[:, None]], axis=1)


def pristine_target_bandtop(clean, raw):
    """The denoising autoencoder's target (INV-AE): the pristine open-channel count `clean` resampled with the band top
    of the DISORDERED input (bandtop_k of to_input_linear(raw)), so it lines up with to_input_bandtop(raw)[:, :416].
    Used only as a training target; the input stays label-free."""
    return resample_to_top(np.asarray(clean, dtype=np.float64), bandtop_k(to_input_linear(raw)))


def reject_reason(draw, clean, raw, weights):
    """Why this draw should not enter the data set, or None."""
    if np.sum(clean) == 0:
        return "no_channel"
    if min(weights.values()) < MIN_WEIGHT:
        return "invisible_type"
    if np.count_nonzero(np.round(raw, 3)) < MIN_CHANNELS:
        return "few_channels"
    return None
