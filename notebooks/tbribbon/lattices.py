"""Ribbon unit cells from geometry: sites of one period, bonds by distance (bond length 1)."""
from typing import NamedTuple

import numpy as np

S = np.sqrt(3) / 2


class RibbonHamiltonian(NamedTuple):
    H0: np.ndarray            # intra-cell
    H1: np.ndarray            # H_{n, n+1}
    positions: np.ndarray     # (n, 2)
    sublattice: np.ndarray    # 0 = A, 1 = B


def _graphene_sites(extent=60):
    a1, a2 = np.array([1.5, S]), np.array([1.5, -S])
    n1, n2 = np.meshgrid(np.arange(-extent, extent), np.arange(-extent, extent))
    base = n1.ravel()[:, None] * a1 + n2.ravel()[:, None] * a2
    return np.vstack([base, base + [1.0, 0.0]]), np.repeat([0, 1], len(base))


def honeycomb_ribbon(N, edge, t=1.0, onsite_a=0.0, onsite_b=0.0, edge_bond_factor=1.0):
    pos, sub = _graphene_sites()
    eps = 1e-6
    if edge == "armchair":                          # periodic along x (period 3), N rows y = k*S
        axis, L = np.array([1.0, 0.0]), 3.0
        keep = (pos[:, 0] > -eps) & (pos[:, 0] < L - eps) & (pos[:, 1] > -eps) & (pos[:, 1] < (N - 1) * S + eps)
    elif edge == "zigzag":                          # periodic along y (period sqrt3), N zigzag chains
        axis, L = np.array([0.0, 1.0]), 2 * S
        xs = np.concatenate([[1 + 1.5 * j, 1.5 + 1.5 * j] for j in range(N)])
        keep = (pos[:, 1] > -eps) & (pos[:, 1] < L - eps) & np.any(np.abs(pos[:, :1] - xs[None]) < eps, axis=1)
    else:
        raise ValueError(f"edge must be armchair or zigzag, got {edge!r}")
    p, s = pos[keep], sub[keep]
    order = np.lexsort((p[:, 0], p[:, 1]))
    p, s = p[order], s[order]
    n = len(p)
    H0 = np.diag(np.where(s == 0, onsite_a, onsite_b)).astype(complex)
    H1 = np.zeros((n, n), complex)
    rows = p[:, 1] if edge == "armchair" else p[:, 0]
    outer = (np.abs(rows - rows.min()) < eps) | (np.abs(rows - rows.max()) < eps)
    for i in range(n):
        for j in range(n):
            d0 = np.linalg.norm(p[i] - p[j])
            if abs(d0 - 1) < eps:
                along = abs(np.dot(p[j] - p[i], axis)) > 1 - eps
                f = edge_bond_factor if (edge == "armchair" and along and outer[i] and outer[j]) else 1.0
                H0[i, j] = -t * f
            if abs(np.linalg.norm(p[j] + L * axis - p[i]) - 1) < eps:
                H1[i, j] = -t
    return RibbonHamiltonian(H0, H1, p, s)


def square_strip(width, t=1.0):
    H0 = np.zeros((width, width), complex)
    i = np.arange(width - 1)
    H0[i, i + 1] = H0[i + 1, i] = -t
    pos = np.column_stack([np.zeros(width), np.arange(width)])
    return RibbonHamiltonian(H0, -t * np.eye(width, dtype=complex), pos, np.zeros(width, int))
