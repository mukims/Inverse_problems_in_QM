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


def hbn_ribbon(N, edge, t=2.30, delta=3.625):
    """Hexagonal boron nitride ribbon with B on A (+delta) and N on B (-delta).
    Energies scaled by t, so t_dim = 1.0, delta_dim = delta / t."""
    t_dim = 1.0
    delta_dim = delta / t
    return honeycomb_ribbon(N, edge, t=t_dim, onsite_a=delta_dim, onsite_b=-delta_dim)


def phosphorene_ribbon(N, edge, t1=-1.220, t2=3.665, t3=-0.205, t4=-0.105, t5=-0.055):
    """Monolayer black phosphorus (phosphorene) ribbon (Rudenko & Katsnelson 2014, Ezawa 2014).
    Energies scaled by t_ref = |t2| = 3.665 eV. On-site shifted by -4*t4 so midgap is at E = 0."""
    t_ref = abs(t2)
    t1_d, t2_d, t3_d, t4_d, t5_d = t1 / t_ref, t2 / t_ref, t3 / t_ref, t4 / t_ref, t5 / t_ref
    on_site_shift = -4 * t4_d

    pos, sub = _graphene_sites()
    eps = 1e-5
    if edge == "armchair":
        axis, L = np.array([1.0, 0.0]), 3.0
        keep = (pos[:, 0] > -eps) & (pos[:, 0] < L - eps) & (pos[:, 1] > -eps) & (pos[:, 1] < (N - 1) * S + eps)
    elif edge == "zigzag":
        axis, L = np.array([0.0, 1.0]), 2 * S
        xs = np.concatenate([[1 + 1.5 * j, 1.5 + 1.5 * j] for j in range(N)])
        keep = (pos[:, 1] > -eps) & (pos[:, 1] < L - eps) & np.any(np.abs(pos[:, :1] - xs[None]) < eps, axis=1)
    else:
        raise ValueError(f"edge must be armchair or zigzag, got {edge!r}")

    p, s = pos[keep], sub[keep]
    order = np.lexsort((p[:, 0], p[:, 1]))
    p, s = p[order], s[order]
    n = len(p)

    H0 = np.eye(n, dtype=complex) * on_site_shift
    H1 = np.zeros((n, n), dtype=complex)

    def _get_hop(dx, dy):
        adx, ady = abs(dx), abs(dy)
        if abs(adx - 0.5) < eps and abs(ady - S) < eps:
            return t1_d
        if abs(adx - 1.0) < eps and ady < eps:
            return t2_d
        if abs(adx - 1.5) < eps and abs(ady - S) < eps:
            return t4_d
        if abs(adx - 2.0) < eps and ady < eps:
            return t5_d
        if abs(adx - 2.5) < eps and abs(ady - S) < eps:
            return t3_d
        return 0.0

    for i in range(n):
        for j in range(n):
            d0 = p[j] - p[i]
            v0 = _get_hop(d0[0], d0[1])
            if v0 != 0.0:
                H0[i, j] += v0

            d1 = p[j] + L * axis - p[i]
            v1 = _get_hop(d1[0], d1[1])
            if v1 != 0.0:
                H1[i, j] += v1

    return RibbonHamiltonian(H0, H1, p, s)


def mos2_ribbon(N, edge, eps1=1.046, eps2=2.104, t0=-0.184, t1=0.401, t2=0.507, t11=0.218, t12=0.338, t22=0.057):
    """Monolayer MoS2 3-band tight-binding ribbon on triangular Mo lattice (Liu et al. PRB 2013).
    Basis: {dz2, dxy, dx2-y2} per Mo site. On-site shifted by midgap (0.7666 eV) so midgap is at E = 0.
    Energy unit t_ref = 1.0 eV."""
    e_mid = 0.7666
    H0_site = np.diag([eps1 - e_mid, eps2 - e_mid, eps2 - e_mid]).astype(complex)

    HR1 = np.array([
        [t0, t1, t2],
        [-t1, t11, t12],
        [t2, -t12, t22]
    ], dtype=complex)

    HR2 = np.array([
        [t0, t1/2 + np.sqrt(3)*t2/2, np.sqrt(3)*t1/2 - t2/2],
        [-t1/2 + np.sqrt(3)*t2/2, t11/4 + 3*t22/4, np.sqrt(3)*t11/4 - t12 - np.sqrt(3)*t22/4],
        [-np.sqrt(3)*t1/2 - t2/2, np.sqrt(3)*t11/4 + t12 - np.sqrt(3)*t22/4, 3*t11/4 + t22/4]
    ], dtype=complex)

    HR3 = np.array([
        [t0, -t1/2 - np.sqrt(3)*t2/2, np.sqrt(3)*t1/2 - t2/2],
        [t1/2 - np.sqrt(3)*t2/2, t11/4 + 3*t22/4, -np.sqrt(3)*t11/4 + t12 + np.sqrt(3)*t22/4],
        [-np.sqrt(3)*t1/2 - t2/2, -np.sqrt(3)*t11/4 - t12 + np.sqrt(3)*t22/4, 3*t11/4 + t22/4]
    ], dtype=complex)

    HR6 = np.array([
        [t0, t1/2 - np.sqrt(3)*t2/2, -np.sqrt(3)*t1/2 - t2/2],
        [-t1/2 - np.sqrt(3)*t2/2, t11/4 + 3*t22/4, -np.sqrt(3)*t11/4 - t12 + np.sqrt(3)*t22/4],
        [np.sqrt(3)*t1/2 - t2/2, -np.sqrt(3)*t11/4 + t12 + np.sqrt(3)*t22/4, 3*t11/4 + t22/4]
    ], dtype=complex)

    if edge == "zigzag":
        # Periodic along x (period a = 1.0), N rows across y (spacing sqrt(3)/2)
        n = 3 * N
        H0 = np.zeros((n, n), dtype=complex)
        H1 = np.zeros((n, n), dtype=complex)
        pos = np.column_stack([np.zeros(N), np.arange(N) * S])
        sub = np.zeros(N, dtype=int)

        for i in range(N):
            H0[3*i:3*i+3, 3*i:3*i+3] = H0_site
            H1[3*i:3*i+3, 3*i:3*i+3] = HR1
            if i + 1 < N:
                H0[3*i:3*i+3, 3*(i+1):3*(i+1)+3] = HR2
                H0[3*(i+1):3*(i+1)+3, 3*i:3*i+3] = HR2.conj().T
                H1[3*i:3*i+3, 3*(i+1):3*(i+1)+3] = HR3
        return RibbonHamiltonian(H0, H1, pos, sub)

    elif edge == "armchair":
        # Periodic along y (period sqrt(3)), N columns across x. Each column has 2 Mo atoms (A at y=0, B at y=sqrt(3)/2).
        n = 6 * N
        H0 = np.zeros((n, n), dtype=complex)
        H1 = np.zeros((n, n), dtype=complex)
        pos_list = []
        for j in range(N):
            pos_list.append([j, 0.0])
            pos_list.append([j + 0.5, S])
        pos = np.array(pos_list)
        sub = np.tile([0, 1], N)

        for j in range(N):
            H0[6*j:6*j+3, 6*j:6*j+3] = H0_site
            H0[6*j+3:6*j+6, 6*j+3:6*j+6] = H0_site
            H0[6*j:6*j+3, 6*j+3:6*j+6] += HR2
            H0[6*j+3:6*j+6, 6*j:6*j+3] += HR2.conj().T
            H1[6*j+3:6*j+6, 6*j:6*j+3] += HR3.conj().T
            if j + 1 < N:
                H0[6*j:6*j+3, 6*(j+1):6*(j+1)+3] += HR1
                H0[6*(j+1):6*(j+1)+3, 6*j:6*j+3] += HR1.conj().T
                H0[6*j+3:6*j+6, 6*(j+1)+3:6*(j+1)+6] += HR1
                H0[6*(j+1)+3:6*(j+1)+6, 6*j+3:6*j+6] += HR1.conj().T
                H0[6*j+3:6*j+6, 6*(j+1):6*(j+1)+3] += HR6
                H0[6*(j+1):6*(j+1)+3, 6*j+3:6*j+6] += HR6.conj().T
                H0[6*(j+1):6*(j+1)+3, 6*j+3:6*j+6] += HR3
                H0[6*j+3:6*j+6, 6*(j+1):6*(j+1)+3] += HR3.conj().T
        return RibbonHamiltonian(H0, H1, pos, sub)

    else:
        raise ValueError(f"edge must be armchair or zigzag, got {edge!r}")
