"""Shazam's ensemble lookup (LOOKUP-1): a signature in, the closest catalogued device out.

Each catalogued device is described by its disordered ensemble: the median and spread of the label-free input at
every energy, interpolated in impurity concentration from the clean spectrum (0%) and the stored densities.
A signature is scored by how probable it is under each device's ensemble, over the energies it covers.
"""
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from .energy import on_axis
from .registry import RibbonModel
from .spec import InputSpec, despike

S0 = 0.01              # spread floor in input units (about 0.04 in T): absorbs interpolation error, keeps the 0% anchor finite
MIN_CHANNELS = 10      # a window must cover at least this many channels
P_NO_MATCH = 0.01      # below this p-value no catalogued device matches
P_VALUE_SPECTRA = 200  # validation spectra per device used for one p-value


def window_inputs(spec, energies, T):
    """Label-free input over the energies supplied: (X, mask), one row of X per spectrum.

    Same steps as InputSpec.to_input (round to 3 decimals, despike, resample, round, clip, log), but channels outside
    [energies[0], energies[-1]] are absent: mask False, value 0. Nothing is zero-filled from a band top (label-free).
    """
    if spec.unit != "eV":
        raise ValueError("the lookup needs an eV-axis InputSpec (v3 or v4)")
    e = np.asarray(energies, dtype=np.float64)
    T = np.atleast_2d(np.asarray(T, dtype=np.float64))
    if e.ndim != 1 or T.shape[1] != e.size:
        raise ValueError(f"T has {T.shape[1]} values per spectrum but there are {e.size} energies")
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(T))):
        raise ValueError("energies and T must be finite")
    if np.any(e < 0):
        raise ValueError("energies must be >= 0 eV, measured from charge neutrality")
    if np.any(np.diff(e) <= 0):
        raise ValueError("energies must increase")
    grid = spec.energies_t()
    mask = (grid >= e[0] - 1e-9) & (grid <= e[-1] + 1e-9)
    if mask.sum() < MIN_CHANNELS:
        raise ValueError(f"the window {e[0]:.3f}-{e[-1]:.3f} eV covers {int(mask.sum())} channels of {spec.step_t} eV; "
                         f"at least {MIN_CHANNELS} are needed")
    T = np.round(T, 3)
    if spec.despike:
        T = despike(T)
    out = np.zeros((T.shape[0], grid.size))
    for i in range(T.shape[0]):
        out[i, mask] = np.interp(grid[mask], e, T[i])
    out = np.clip(np.round(out, 3), 0.0, spec.cap)
    return (np.log1p(out) / np.log1p(spec.cap)).astype(np.float32), mask


def window_input(spec, energies, T):
    """One signature: (x, mask)."""
    T = np.asarray(T, dtype=np.float64)
    if T.ndim != 1:
        raise ValueError("a signature is one spectrum: T must be 1D")
    X, mask = window_inputs(spec, energies, T)
    return X[0], mask
