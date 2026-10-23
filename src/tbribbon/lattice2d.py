"""Ribbons of any 2D lattice from data (MATERIALS-2): Bravais vectors, basis, bonds, and an edge cut.

A lattice is described by data, not code: two Bravais vectors, the basis sites, the nearest-neighbour hopping (every
pair of sites at distance `nn`), and any extra bonds. An edge is a pair of lattice vectors, T along the ribbon and W
across it, plus optional per-sublattice lattice offsets that choose how the edge is terminated. Bonds are found by
matching displacements, as in the geometric MoS2 builder (Bug #10).
"""
from dataclasses import dataclass

import numpy as np

from .lattices import RibbonHamiltonian

TOL = 1e-6


@dataclass(frozen=True)
class Edge:
    name: str
    T: tuple                 # periodic direction along the ribbon, in (a1, a2) coordinates
    W: tuple                 # one step across the ribbon, in (a1, a2) coordinates
    offsets: tuple = ()      # per basis site, an integer (m, n) shift choosing the termination; default none


@dataclass(frozen=True)
class Lattice:
    name: str
    a1: tuple
    a2: tuple
    basis: tuple             # ((x, y), ...): one position per sublattice
    edges: tuple             # (Edge, ...)
    nn: float = 1.0          # every pair of sites at this distance is bonded with hopping -t
    extra: tuple = ()        # ((i, j, (dx, dy)), ...): extra bonds, sublattice i to j at displacement d, listed once

    def edge(self, name):
        for e in self.edges:
            if e.name == name:
                return e
        raise ValueError(f"{self.name} has no edge {name!r}; edges are {[e.name for e in self.edges]}")


def _vec(lat, mn):
    return mn[0] * np.asarray(lat.a1, float) + mn[1] * np.asarray(lat.a2, float)


def _cell_points(T, W):
    """Lattice points (m, n) inside the parallelogram spanned by T and W; there are |det(T, W)| of them."""
    det = T[0] * W[1] - T[1] * W[0]
    if det == 0:
        raise ValueError(f"T={T} and W={W} are parallel")
    inv = np.linalg.inv(np.array([T, W], float).T)
    r = abs(T[0]) + abs(T[1]) + abs(W[0]) + abs(W[1])
    pts = [(m, n) for m in range(-r, r + 1) for n in range(-r, r + 1)
           if np.all((f := inv @ np.array([m, n], float)) > -TOL) and np.all(f < 1 - TOL)]
    assert len(pts) == abs(det), (pts, det)
    return sorted(pts)


def _bonds(lat):
    """Extra bonds in both directions: (i, j, d)."""
    out = []
    for i, j, d in lat.extra:
        out.append((i, j, np.asarray(d, float)))
        out.append((j, i, -np.asarray(d, float)))
    return out


def _hop(lat, pos, sub, shift, t):
    """Hopping block between sites at `pos` and the same sites displaced by `shift`: H[i, j] couples i and j + shift."""
    D = pos[None, :, :] + shift - pos[:, None, :]
    H = np.where(np.abs(np.linalg.norm(D, axis=-1) - lat.nn) < TOL, -t, 0.0).astype(complex)
    for i_s, j_s, d in _bonds(lat):
        hit = (np.abs(D - d).max(axis=-1) < TOL) & (sub[:, None] == i_s) & (sub[None, :] == j_s)
        H[hit] += -t
    return H


def lattice_ribbon(lat, width, edge, t=1.0):
    """Ribbon of `width` cells across, periodic along the edge's T. Returns RibbonHamiltonian(H0, H1, positions, sublattice)."""
    e = lat.edge(edge)
    if int(width) < 1:
        raise ValueError("width must be >= 1")
    offsets = e.offsets or tuple((0, 0) for _ in lat.basis)
    if len(offsets) != len(lat.basis):
        raise ValueError(f"{lat.name}/{edge}: {len(offsets)} offsets for {len(lat.basis)} basis sites")
    pos, sub = [], []
    for k in range(int(width)):
        for p in _cell_points(e.T, e.W):
            for s, (b, o) in enumerate(zip(lat.basis, offsets)):
                pos.append(_vec(lat, p) + _vec(lat, o) + k * _vec(lat, e.W) + np.asarray(b, float))
                sub.append(s)
    pos, sub = np.array(pos), np.array(sub)
    return RibbonHamiltonian(_hop(lat, pos, sub, np.zeros(2), t), _hop(lat, pos, sub, _vec(lat, e.T), t), pos, sub)


def bulk_hamiltonian(lat, k, t=1.0, reach=2):
    """Bloch Hamiltonian of the infinite lattice at wavevector k, from the same data."""
    b = np.asarray(lat.basis, float)
    n = len(b)
    H = np.zeros((n, n), complex)
    for m in range(-reach, reach + 1):
        for q in range(-reach, reach + 1):
            R = _vec(lat, (m, q))
            ph = np.exp(1j * np.dot(k, R))
            for i in range(n):
                for j in range(n):
                    d = b[j] + R - b[i]
                    if abs(np.linalg.norm(d) - lat.nn) < TOL:
                        H[i, j] += -t * ph
                    for i_s, j_s, dd in _bonds(lat):
                        if i_s == i and j_s == j and np.abs(d - dd).max() < TOL:
                            H[i, j] += -t * ph
    return H


S3 = np.sqrt(3.0)
HONEYCOMB = Lattice("honeycomb", (1.5, S3 / 2), (1.5, -S3 / 2), ((0.0, 0.0), (1.0, 0.0)),
                    (Edge("armchair", (1, 1), (1, 0)), Edge("zigzag", (1, -1), (1, 0), ((0, 0), (0, -1)))))
TRIANGULAR = Lattice("triangular", (1.0, 0.0), (0.5, S3 / 2), ((0.0, 0.0),),
                     (Edge("zigzag", (1, 0), (0, 1)), Edge("armchair", (-1, 2), (1, 0))))
KAGOME = Lattice("kagome", (2.0, 0.0), (1.0, S3), ((0.0, 0.0), (1.0, 0.0), (0.5, S3 / 2)),
                 (Edge("zigzag", (1, 0), (0, 1)), Edge("armchair", (-1, 2), (1, 0))))
LIEB = Lattice("lieb", (2.0, 0.0), (0.0, 2.0), ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
               (Edge("strip", (1, 0), (0, 1)),))
CHECKERBOARD = Lattice("checkerboard", (1.0, 1.0), (1.0, -1.0), ((0.0, 0.0), (1.0, 0.0)),
                       (Edge("strip", (1, 1), (1, -1)),), extra=((0, 0, (1.0, 1.0)), (1, 1, (1.0, -1.0))))
LATTICES = {lat.name: lat for lat in (KAGOME, LIEB, CHECKERBOARD)}
