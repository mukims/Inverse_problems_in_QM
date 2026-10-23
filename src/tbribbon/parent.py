"""Hexagonal parent lattice: one tight-binding host family that contains many 2D materials.

Three site types share the honeycomb cell: C (hexagon centre), H (the A/B honeycomb sites) and K
(the kagome sites on the bond midpoints). A host switches types on or off and sets on-site energies
and hoppings (units of the lead hopping, which is 1); `s` is the energy scale in eV.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from .lattice2d import HONEYCOMB, TOL, _cell_points
from .lattices import RibbonHamiltonian

TYPES = ("C", "H", "K")
NAMED = ("graphene-ideal", "silicene", "germanene", "hbn", "triangular", "kagome")


@dataclass(frozen=True)
class HexParams:
    on_C: bool
    on_H: bool
    on_K: bool
    lead: str                  # "C", "H" or "K": the type whose within-type hopping is the unit
    s: float                   # energy scale (eV)
    eps_C: float = 0.0
    eps_H_mean: float = 0.0
    delta: float = 0.0         # eps_A = eps_H_mean + delta, eps_B = eps_H_mean - delta
    eps_K: float = 0.0
    t_CC: float = 0.0
    t_AB: float = 0.0
    t_KK: float = 0.0
    t_AA: float = 0.0          # A-A and B-B (H only)
    t_CH: float = 0.0
    t_CK: float = 0.0
    t_HK: float = 0.0

    @property
    def eps_A(self):
        return self.eps_H_mean + self.delta

    @property
    def eps_B(self):
        return self.eps_H_mean - self.delta

    @property
    def pattern(self):
        return (self.on_C, self.on_H, self.on_K)

    @property
    def active_types(self):
        return tuple(t for t, on in zip(TYPES, self.pattern) if on)

    def validate(self):
        """Raise ValueError unless this is a canonical host (spec section 4.2)."""
        active = self.active_types
        if not active:
            raise ValueError("no site type is active")
        if self.lead not in active:
            raise ValueError(f"lead {self.lead!r} is not an active type")
        within = {"C": self.t_CC, "H": self.t_AB, "K": self.t_KK}
        if within[self.lead] != 1:
            raise ValueError(f"lead hopping must be exactly 1, got {within[self.lead]}")
        for t in active:
            if not 0 < within[t] <= 1:
                raise ValueError(f"within-type hopping of {t} must be in (0, 1], got {within[t]}")
        carried = {"C": (self.eps_C, self.t_CC), "H": (self.eps_H_mean, self.delta, self.t_AB, self.t_AA),
                   "K": (self.eps_K, self.t_KK)}
        for t in TYPES:
            if t not in active and any(v != 0 for v in carried[t]):
                raise ValueError(f"inactive type {t} carries non-zero parameters")
        if self.t_AA != 0 and not 0 < self.t_AA <= 1:
            raise ValueError(f"t_AA must be in (0, 1], got {self.t_AA}")
        cross = {("C", "H"): self.t_CH, ("C", "K"): self.t_CK, ("H", "K"): self.t_HK}
        for (a, b), v in cross.items():
            if v != 0 and not (a in active and b in active):
                raise ValueError(f"t_{a}{b} needs both {a} and {b} active")
        if self.delta < 0:
            raise ValueError(f"delta must be >= 0, got {self.delta}")
        if not self.s > 0:
            raise ValueError(f"s must be > 0, got {self.s}")
        bonds = sum(1 for (a, b), v in cross.items() if a in active and b in active and v > 0)
        if len(active) == 2 and bonds < 1:
            raise ValueError("the two active types are not connected")
        if len(active) == 3 and bonds < 2:
            raise ValueError("three active types need at least two cross hoppings")


def named_material(name):
    """The known materials of spec section 4.3 as canonical hosts."""
    if name == "graphene-ideal":
        return HexParams(False, True, False, "H", 2.7, t_AB=1.0)
    if name == "silicene":
        return HexParams(False, True, False, "H", 1.067, t_AB=1.0)
    if name == "germanene":
        return HexParams(False, True, False, "H", 0.991, t_AB=1.0)
    if name == "hbn":
        return HexParams(False, True, False, "H", 2.30, delta=3.625 / 2.30, t_AB=1.0)
    if name == "triangular":
        return HexParams(True, False, False, "C", 1.0, t_CC=1.0)
    if name == "kagome":
        return HexParams(False, False, True, "K", 1.0, t_KK=1.0)
    raise ValueError(f"unknown material {name!r}; choose from {NAMED}")


# ---------------------------------------------------------------------------------------------------------------------
# Geometry (spec section 4.1): the honeycomb Bravais lattice of lattice2d.HONEYCOMB with six basis sites.

S3 = np.sqrt(3.0)
A1, A2 = np.asarray(HONEYCOMB.a1, float), np.asarray(HONEYCOMB.a2, float)
BASIS = ((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.5, 0.0), (-0.25, S3 / 4), (-0.25, -S3 / 4))   # C, A, B, K1, K2, K3
BASIS_TYPE = ("C", "H", "H", "K", "K", "K")
SITE_KIND = ("C", "A", "B", "K", "K", "K")          # H split into its A and B sublattices

# A hopping is identified by (unordered site-kind pair, distance), never by distance alone: C-C and A-A share sqrt3,
# C-K and K-K share sqrt3/2. Each entry: (kind pair, distance, HexParams field holding t); the matrix element is -t.
HOPPINGS = ((("C", "C"), S3, "t_CC"),
            (("A", "B"), 1.0, "t_AB"),
            (("A", "A"), S3, "t_AA"), (("B", "B"), S3, "t_AA"),
            (("C", "A"), 1.0, "t_CH"), (("C", "B"), 1.0, "t_CH"),
            (("C", "K"), S3 / 2, "t_CK"),
            (("K", "K"), S3 / 2, "t_KK"),
            (("A", "K"), 0.5, "t_HK"), (("B", "K"), 0.5, "t_HK"))

# Edges: T (along the ribbon) and W (one step across), in (a1, a2) coordinates, taken from HONEYCOMB.
EDGE_VECTORS = {e.name: (e.T, e.W) for e in HONEYCOMB.edges}
# Per-basis termination offsets: an integer (m, n) lattice shift for each basis site in BASIS order (C, A, B, K1, K2,
# K3). They choose which period image of each site belongs to the cell, and so how each edge is terminated. A and B
# follow HONEYCOMB; C and K are tuned against the existing builders (Task 3).
OFFSETS = {
    "armchair": ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)),
    "zigzag": ((0, 0), (0, 0), (0, -1), (0, 0), (0, 0), (0, 0)),
}


def _vec(mn):
    return mn[0] * A1 + mn[1] * A2


def _check_shells(reach=2):
    """Assert every distance of the section 4.1 table and the kagome / centre coordination on the infinite lattice."""
    b = np.asarray(BASIS, float)
    imgs = [(i, b[i] + _vec((m, n))) for m in range(-reach, reach + 1) for n in range(-reach, reach + 1)
            for i in range(len(b))]
    for (a, c), d, name in HOPPINGS:
        found = any(abs(np.linalg.norm(r - b[i]) - d) < TOL for i in range(len(b)) if SITE_KIND[i] == a
                    for j, r in imgs if SITE_KIND[j] == c)
        if not found:
            raise RuntimeError(f"no {a}-{c} pair at distance {d} for {name}")
    for i in range(len(b)):
        near_k = sum(1 for j, r in imgs if SITE_KIND[j] == "K" and abs(np.linalg.norm(r - b[i]) - S3 / 2) < TOL)
        want = {"K": 4, "C": 6}.get(SITE_KIND[i])
        if want is not None and near_k != want:
            raise RuntimeError(f"basis site {i} ({SITE_KIND[i]}) has {near_k} K sites at sqrt3/2, expected {want}")


_check_shells()


def _onsite(params):
    return {"C": params.eps_C, "A": params.eps_A, "B": params.eps_B, "K": params.eps_K}


def _active_basis(params):
    return [i for i, t in enumerate(BASIS_TYPE) if t in params.active_types]


def _hop(params, pos, kind, shift):
    """Hopping block between sites at `pos` and the same sites displaced by `shift`: H[i, j] couples i and j + shift."""
    D = pos[None, :, :] + shift - pos[:, None, :]
    dist = np.linalg.norm(D, axis=-1)
    H = np.zeros(dist.shape, complex)
    ki, kj = kind[:, None], kind[None, :]
    for (a, c), d, name in HOPPINGS:
        t = getattr(params, name)
        if t == 0:
            continue
        pair = ((ki == a) & (kj == c)) | ((ki == c) & (kj == a))
        H[pair & (np.abs(dist - d) < TOL)] += -t
    return H


def _edge(edge):
    if edge not in EDGE_VECTORS:
        raise ValueError(f"edge must be one of {sorted(EDGE_VECTORS)}, got {edge!r}")
    return EDGE_VECTORS[edge]


def parent_ribbon(params, edge, width):
    """Ribbon of `width` steps of the edge's W, periodic along its T; only active types get sites.

    Returns RibbonHamiltonian(H0, H1, positions, sublattice) with `sublattice` the basis index 0-5 of each site.
    """
    T, W = _edge(edge)
    if int(width) < 1:
        raise ValueError("width must be >= 1")
    offsets, active = OFFSETS[edge], _active_basis(params)
    pos, sub = [], []
    for k in range(int(width)):
        for p in _cell_points(T, W):
            for i in active:
                pos.append(_vec(p) + _vec(offsets[i]) + k * _vec(W) + np.asarray(BASIS[i], float))
                sub.append(i)
    pos, sub = np.array(pos), np.array(sub)
    kind = np.array(SITE_KIND)[sub]
    eps = _onsite(params)
    H0 = _hop(params, pos, kind, np.zeros(2)) + np.diag([eps[k] for k in kind]).astype(complex)
    return RibbonHamiltonian(H0, _hop(params, pos, kind, _vec(T)), pos, sub)


def bulk_hamiltonian(params, k, reach=2):
    """Bloch Hamiltonian at wavevector(s) k (shape (2,) or (..., 2)) over the active basis sites, in BASIS order."""
    active = _active_basis(params)
    b = np.asarray(BASIS, float)[active]
    kind = np.array(SITE_KIND)[active]
    eps = _onsite(params)
    Rs, blocks = [], []
    for m in range(-reach, reach + 1):
        for n in range(-reach, reach + 1):
            R = _vec((m, n))
            T = _hop(params, b, kind, R)
            if m == 0 and n == 0:
                T = T + np.diag([eps[x] for x in kind])
            if np.any(T != 0):
                Rs.append(R)
                blocks.append(T)
    k = np.asarray(k, float)
    if not Rs:
        return np.zeros(k.shape[:-1] + (len(b), len(b)), complex)
    phase = np.exp(1j * (k @ np.array(Rs).T))                    # (..., nR)
    return np.tensordot(phase, np.array(blocks), axes=(-1, 0))


def active_atoms_per_cell(params, edge, width):
    """Sites in one unit cell of the parent ribbon."""
    T, W = _edge(edge)
    return int(width) * len(_cell_points(T, W)) * len(_active_basis(params))


def device_cells_for(params, edge, target_atoms=2000, ref_widths=(7, 9)):
    """Device length: the cells that put the w = 7 and w = 9 devices near target_atoms (formula of materials.device_cells)."""
    per_cell = sum(active_atoms_per_cell(params, edge, w) for w in ref_widths) / len(ref_widths)
    return max(1, round(target_atoms / per_cell))


# ---------------------------------------------------------------------------------------------------------------------
# Site matching between two builders of the same ribbon.

def _rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


_FLIP = (np.eye(2), np.diag([1.0, -1.0]))


def _similarity(src, dst, reflect):
    """Least-squares dst ~ s * R @ src + t with det R = -1 if reflect else +1 (Umeyama). Returns (s, R, t)."""
    ms, md = src.mean(0), dst.mean(0)
    a, c = src - ms, dst - md
    U, S, Vt = np.linalg.svd(c.T @ a / len(src))
    D = np.diag([1.0, (-1.0 if reflect else 1.0) * np.sign(np.linalg.det(U @ Vt)) or 1.0])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean()
    return s, R, md - s * R @ ms


def _assign(cost):
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(cost.shape[0], int)
    perm[rows] = cols
    return perm


def _normalise(x):
    c = x - x.mean(0)
    r = np.sqrt((c ** 2).sum(1).mean())
    return c / (r if r > 0 else 1.0)


def _match_free(pos, ref):
    P, Q = _normalise(pos), _normalise(ref)
    tree = cKDTree(Q)
    best = (np.inf, None, None)
    for refl, F in enumerate(_FLIP):
        for deg in range(360):
            M = _rot(deg) @ F
            r = tree.query(P @ M.T)[0].mean()
            if r < best[0] - 1e-12:
                best = (r, M, bool(refl))
    _, M, refl = best
    perm = _assign(cdist(Q, P @ M.T))
    s, R, t = _similarity(pos[perm], ref, refl)                  # refine the 1-degree grid to the exact transform
    return _assign(cdist(ref, pos @ (s * R).T + t))


def _frame(q):
    L = np.linalg.norm(q)
    u = q / L
    return u, np.array([-u[1], u[0]]), L


def _transverse_score(pos, ref, M, phat):
    """Distance between the standardised distributions across the period direction (independent of period images)."""
    _, n, _ = _frame(M @ phat)
    vp, vr = pos @ (M.T @ n), ref @ n
    sp, sr = vp.std(), vr.std()
    if sp == 0 or sr == 0:
        return 0.0 if sp == sr else np.inf
    return np.abs(np.sort((vr - vr.mean()) / sr) - np.sort((vp - vp.mean()) / sp)).mean()


def _periodic_cost(X, ref, q):
    """Distances between X and ref modulo the period q (both in ref's frame) and the period offsets ref - X."""
    u, n, L = _frame(q)
    du = (ref @ u)[:, None] - (X @ u)[None, :]
    k = np.round(du / L)
    return np.hypot(du - k * L, (ref @ n)[:, None] - (X @ n)[None, :]), k


def _wrapped(x, q):
    u, n, L = _frame(q)
    return np.column_stack([np.mod(x @ u, L), x @ n]), L


def _periodic_residual(X, ref, q):
    Xw, L = _wrapped(X, q)
    Rw, _ = _wrapped(ref, q)
    tree = cKDTree(np.vstack([Xw - [L, 0], Xw, Xw + [L, 0]]))
    return tree.query(Rw)[0].mean()


def _match_periodic(pos, ref, period, n_candidates=6):
    p = np.asarray(period, float)
    phat = p / np.linalg.norm(p)
    cands = []
    for refl, F in enumerate(_FLIP):
        sc = np.array([_transverse_score(pos, ref, _rot(d) @ F, phat) for d in range(360)])
        for d in range(360):                                     # local minima on the circle
            if sc[d] <= sc[d - 1] and sc[d] <= sc[(d + 1) % 360]:
                cands.append((sc[d], refl, d))
    scale = max(np.sqrt(((ref - ref.mean(0)) ** 2).sum(1).mean()), 1e-300)
    best = None                     # ties (symmetric ribbons): prefer no reflection, then the smallest rotation
    for _, refl, d in sorted(cands)[:n_candidates]:
        F = _FLIP[refl]
        fine = np.arange(d - 1, d + 1.0001, 0.01)
        deg = fine[np.argmin([_transverse_score(pos, ref, _rot(x) @ F, phat) for x in fine])]
        M = _rot(deg) @ F
        q_hat, n, _ = _frame(M @ phat)
        vp, vr = pos @ (M.T @ n), ref @ n
        s = vr.std() / vp.std() if vp.std() > 0 else 1.0
        X0 = pos @ (s * M).T
        tv = vr.mean() - (X0 @ n).mean()
        j0 = int(np.argmin(vr))
        vx = X0 @ n + tv
        near = np.flatnonzero(np.abs(vx - vr[j0]) < 0.05 * max(vr.std(), 1e-12))
        for i in (near if len(near) else range(len(pos))):
            t = (ref[j0] @ q_hat - X0[i] @ q_hat) * q_hat + tv * n
            r = _periodic_residual(X0 + t, ref, s * M @ p)
            key = (round(r / scale, 9), refl, round(min(deg % 360, 360 - deg % 360), 6))
            if best is None or key < best[0]:
                best = (key, refl, s * M, t)
    _, refl, A, t = best
    for _ in range(2):                                           # assign, then refine the transform exactly
        X, q = pos @ A.T + t, A @ p
        cost, k = _periodic_cost(X, ref, q)
        perm = _assign(cost)
        shift = k[np.arange(len(ref)), perm].astype(int)
        s, R, t = _similarity(pos[perm] + shift[:, None] * p, ref, bool(refl))
        A = s * R
    X, q = pos @ A.T + t, A @ p
    cost, k = _periodic_cost(X, ref, q)
    perm = _assign(cost)
    shift = k[np.arange(len(ref)), perm].astype(int)
    # A whole-period translation is undetermined; fix it so that the most common shift is 0 (smallest on a tie).
    return perm, shift - (np.argmax(np.bincount(shift - shift.min())) + shift.min())


def match_sites(pos, ref, period=None):
    """Match two site sets that are the same up to translation, rotation, reflection and scale.

    Without `period`, returns `perm` with pos[perm] matching ref. With `period` (the ribbon period in pos's frame),
    sites are compared modulo the period and the result is (perm, shift): pos[perm[j]] + shift[j] * period lands on
    ref[j] (after the transform); the common whole-period offset is fixed so that the most frequent shift is 0.
    When the sets are symmetric, transforms that fit equally well are tied; the one without reflection and with
    the smallest rotation wins.
    """
    pos, ref = np.asarray(pos, float), np.asarray(ref, float)
    if pos.shape != ref.shape or pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"pos and ref must both be (n, 2), got {pos.shape} and {ref.shape}")
    if len(pos) < 2:
        perm = np.arange(len(pos))
        return perm if period is None else (perm, np.zeros(len(pos), int))
    return _match_free(pos, ref) if period is None else _match_periodic(pos, ref, period)
