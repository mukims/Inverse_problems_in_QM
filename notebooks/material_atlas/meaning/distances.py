# notebooks/material_atlas/meaning/distances.py
"""Physical distance between ribbons, from their clean spectra under the label-free input transform."""
import numpy as np


def rms_distance(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"clean spectra differ in shape: {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def clean_distance_matrix(clean):
    clean = np.asarray(clean, float)
    if clean.ndim != 2:
        raise ValueError(f"clean must be (n_ribbons, n_channels), got shape {clean.shape}")
    diff = clean[:, None, :] - clean[None, :, :]
    return np.sqrt(np.mean(diff ** 2, axis=2))


def distances_to(query, clean):
    query, clean = np.asarray(query, float), np.asarray(clean, float)
    if query.ndim != 1 or clean.ndim != 2 or query.shape[0] != clean.shape[1]:
        raise ValueError(f"query {query.shape} does not match clean spectra {clean.shape}")
    return np.sqrt(np.mean((clean - query[None, :]) ** 2, axis=1))


def kernel_sigma(D):
    """Median over ribbons of the distance to the nearest other ribbon: that neighbour gets weight e^-1."""
    D = np.asarray(D, float)
    n = len(D)
    if n < 2:
        raise ValueError("need at least two ribbons")
    off = D[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    return float(np.median(off.min(axis=1)))


def shape_profile(T, e, n=200, cap=64.0):
    """Clean spectrum on a band-top-normalised axis u = E / E_top (label-free: E_top = last energy with T > 0.5).

    Removes the energy scale (hopping), so the distance between profiles compares band shape only.
    """
    T, e = np.asarray(T, float), np.asarray(e, float)
    open_ = np.flatnonzero(T > 0.5)
    if open_.size == 0:
        raise ValueError("spectrum has no open channel (T <= 0.5 everywhere)")
    top = e[open_[-1]]
    u = np.arange(n) / n
    prof = np.interp(u * top, e, T)
    return np.log1p(np.clip(prof, 0.0, cap)) / np.log1p(cap)
