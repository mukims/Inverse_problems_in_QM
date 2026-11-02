"""Training prior over the hexagonal parent lattice (spec section 4.2), box-edge warnings and hidden-material tests."""
from itertools import combinations
from typing import NamedTuple

import numpy as np

from .parent import EDGE_VECTORS, TYPES, HexParams

RANGES = {"s": (0.7, 4.0), "eps": (-2.0, 2.0), "t_within": (0.2, 1.0), "t_AA": (0.03, 0.3), "t_cross": (0.05, 1.0),
          "delta_min": (0.05, 2.0), "w": (4, 22), "c": (0.0025, 0.05)}     # delta_min: smallest nonzero Delta, and its cap
PATTERNS = [tuple(bool(x) for x in p) for p in np.ndindex(2, 2, 2) if any(p)]
EDGES = ("armchair", "zigzag")
# The one definition of these names (shazam.inverse.encoding and .data import them): the cross-type bond of each type
# pair, the within-type hopping and the on-site energy of each type.
CROSS = {("C", "H"): "t_CH", ("C", "K"): "t_CK", ("H", "K"): "t_HK"}
WITHIN = {"C": "t_CC", "H": "t_AB", "K": "t_KK"}
EPS = {"C": "eps_C", "H": "eps_H_mean", "K": "eps_K"}
# Free parameters in HexParams field order (box_edge_warnings reports them in this order, then "w", then "c"); also the
# columns of the stored `params` array of the generated data (shazam.inverse.data.PARAM_COLUMNS).
FIELDS = ("s", "eps_C", "eps_H_mean", "delta", "eps_K", "t_CC", "t_AB", "t_KK", "t_AA", "t_CH", "t_CK", "t_HK")


class Draw(NamedTuple):
    params: HexParams
    edge: str
    width: int
    density: float


def _log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def delta_upper(eps_H_mean):
    """Upper end of the nonzero Delta interval [0.05, 2 - |eps_H_mean|] (empty when |eps_H_mean| > 1.95, ruling R11)."""
    return RANGES["delta_min"][1] - abs(eps_H_mean)


def sample_prior(rng):
    """One training draw, in the order of spec section 4.2."""
    pattern = PATTERNS[rng.integers(len(PATTERNS))]
    active = [t for t, on in zip(TYPES, pattern) if on]
    lead = active[rng.integers(len(active))]
    v = {"s": _log_uniform(rng, *RANGES["s"])}
    lo, hi = RANGES["eps"]
    for t in TYPES:                                   # on-site energies; a lone type has a point mass at 0
        if t in active:
            alone = len(active) == 1
            v[EPS[t]] = 0.0 if alone and rng.random() < 0.5 else float(rng.uniform(lo, hi))
        if t == "H" and "H" in active:
            top = delta_upper(v["eps_H_mean"])
            nonzero = rng.random() >= 0.5
            v["delta"] = float(rng.uniform(RANGES["delta_min"][0], top)) if nonzero and top > RANGES["delta_min"][0] else 0.0
    for t in active:
        v[WITHIN[t]] = 1.0 if t == lead else float(rng.uniform(*RANGES["t_within"]))
    if "H" in active:
        v["t_AA"] = float(rng.uniform(*RANGES["t_AA"])) if rng.random() >= 0.5 else 0.0
    bonds = [pair for pair in combinations(active, 2)]
    if len(active) == 3:                              # the 4 connected on/off combinations: 3 trees and the triangle
        on = [bonds[:2], bonds[1:], [bonds[0], bonds[2]], bonds][rng.integers(4)]
    else:
        on = bonds
    for pair in on:
        v[CROSS[pair]] = float(rng.uniform(*RANGES["t_cross"]))
    params = HexParams(*pattern, lead, **v)
    edge = EDGES[rng.integers(2)]
    width = int(rng.integers(RANGES["w"][0], RANGES["w"][1] + 1))
    return Draw(params, edge, width, _log_uniform(rng, *RANGES["c"]))


def _free_intervals(p):
    """(field, lo, hi, log scale) for every active, freely drawn continuous parameter whose value is not a point mass."""
    out = [("s", *RANGES["s"], True)]
    for t in p.active_types:
        out.append((EPS[t], *RANGES["eps"], False))
        if t != p.lead:
            out.append((WITHIN[t], *RANGES["t_within"], False))
    if p.on_H:
        out.append(("delta", RANGES["delta_min"][0], delta_upper(p.eps_H_mean), False))
        out.append(("t_AA", *RANGES["t_AA"], False))
    for (a, b), name in CROSS.items():
        if a in p.active_types and b in p.active_types:
            out.append((name, *RANGES["t_cross"], False))
    order = {f: i for i, f in enumerate(FIELDS)}
    return sorted(out, key=lambda x: order[x[0]])


def _near_bound(x, lo, hi, log):
    if log:
        x, lo, hi = np.log(x), np.log(lo), np.log(hi)
    margin = 0.02 * (hi - lo)
    return hi > lo and (x - lo <= margin or hi - x <= margin)


def box_edge_warnings(draw):
    """Names of the free continuous parameters within 2% of their interval's width from a bound (s and c judged in log
    space), then "w" when the width is 4 or 22 and "c"; in HexParams field order, then w, then c."""
    p = draw.params
    names = [f for f, lo, hi, log in _free_intervals(p)
             if getattr(p, f) != 0 and _near_bound(getattr(p, f), lo, hi, log)]
    if draw.width in RANGES["w"]:
        names.append("w")
    if _near_bound(draw.density, *RANGES["c"], True):
        names.append("c")
    return names


def excluded(draw, hide):
    """True when the draw lies in the neighbourhood of the hidden material `hide` (spec section 9)."""
    p = draw.params
    if hide == "hbn":
        return p.pattern == (False, True, False) and abs(p.delta - 1.576) < 0.2
    if hide == "kagome":
        return p.pattern == (False, False, True) and abs(p.eps_K) < 0.2
    if hide == "silicene":
        return p.pattern == (False, True, False) and p.delta < 0.1 and 0.85 <= p.s <= 1.3
    raise ValueError(f"hide must be one of hbn, kagome, silicene, got {hide!r}")


def _in(x, lo, hi):
    return lo <= x <= hi


def in_prior(draw):
    """True iff `draw` could have come from sample_prior."""
    p = draw.params
    try:
        p.validate()                                  # a pattern, an active lead of hopping 1, no inactive values, connectivity
    except ValueError:
        return False
    if not (_in(p.s, *RANGES["s"]) and _in(p.eps_C, *RANGES["eps"]) and _in(p.eps_H_mean, *RANGES["eps"])
            and _in(p.eps_K, *RANGES["eps"])):
        return False
    if p.delta != 0 and not _in(p.delta, RANGES["delta_min"][0], delta_upper(p.eps_H_mean)):
        return False
    for t in p.active_types:
        if t != p.lead and not _in(getattr(p, WITHIN[t]), *RANGES["t_within"]):
            return False
    if p.t_AA != 0 and not _in(p.t_AA, *RANGES["t_AA"]):
        return False
    if any(v != 0 and not _in(v, *RANGES["t_cross"]) for v in (p.t_CH, p.t_CK, p.t_HK)):
        return False
    return (draw.edge in EDGE_VECTORS and draw.width == int(draw.width) and _in(draw.width, *RANGES["w"])
            and _in(draw.density, *RANGES["c"]))
