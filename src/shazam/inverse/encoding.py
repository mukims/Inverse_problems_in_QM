"""Discrete states and the unbounded continuous encoding of a parent-lattice draw (spec sections 4.2 and 6).

A draw is (discrete state, 14 continuous values). The state holds the edge, the site-type pattern, the lead and the
on/off switches (point masses and which bonds exist); the continuous values are a logit of each active parameter's
position inside its prior interval, so a generative model can emit any real vector. Slots a state does not use are 0.
"""
from itertools import product
from typing import NamedTuple

import numpy as np
from scipy.special import expit, logit

from tbribbon.parent import TYPES, HexParams
from tbribbon.parent_prior import CROSS, EDGES, EPS, FIELDS, PATTERNS, RANGES, WITHIN, Draw, delta_upper

CONT = ("log_s", "eps_C", "eps_H_mean", "delta", "eps_K", "t_CC", "t_AB", "t_KK", "t_AA", "t_CH", "t_CK", "t_HK",
        "log_c", "w")
SLOT = {name: i for i, name in enumerate(CONT)}
_PAIRS = tuple(CROSS)                                             # (C-H, C-K, H-K): the order of DiscreteState.cross
_CROSS_NAMES = tuple(CROSS.values())                              # (t_CH, t_CK, t_HK)
W_BOUNDS = (RANGES["w"][0] - 0.5, RANGES["w"][1] + 0.5)           # continuous box for the integer width
_EPS_LOGIT = 1e-9                                                 # logit arguments are clipped to [1e-9, 1 - 1e-9]


class DiscreteState(NamedTuple):
    edge: str
    pattern: tuple[bool, bool, bool]
    lead: str
    delta_zero: bool                       # Delta = 0 (H active only)
    eps_zero: bool                         # the lone active type's on-site energy is 0 (single-type patterns only)
    t_AA_on: bool                          # t_AA nonzero (H active only)
    cross: tuple[bool, bool, bool]         # (t_CH, t_CK, t_HK) nonzero
    # Switches that do not apply to the pattern are False.


def _cross_options(pattern):
    active = [t for t, on in zip(TYPES, pattern) if on]
    if len(active) == 1:
        return [(False, False, False)]
    if len(active) == 2:                                          # the one connecting bond
        return [tuple(a in active and b in active for a, b in _PAIRS)]
    return [(True, True, False), (False, True, True), (True, False, True), (True, True, True)]   # 3 trees + triangle


def _build_states():
    out = []
    for edge, pattern in product(EDGES, PATTERNS):
        active = [t for t, on in zip(TYPES, pattern) if on]
        for lead in active:
            for eps_zero, delta_zero, t_AA_on, cross in product(
                    (False, True) if len(active) == 1 else (False,),
                    (False, True) if pattern[1] else (False,),
                    (False, True) if pattern[1] else (False,),
                    _cross_options(pattern)):
                out.append(DiscreteState(edge, pattern, lead, delta_zero, eps_zero, t_AA_on, cross))
    return tuple(out)


STATES = _build_states()
_INDEX = {s: i for i, s in enumerate(STATES)}


def state_index(s):
    return _INDEX[s]


def active_slots(state):
    """Boolean mask over CONT of the continuous values the state uses."""
    s = STATES[state] if isinstance(state, (int, np.integer)) else state
    m = np.zeros(len(CONT), bool)
    m[[SLOT["log_s"], SLOT["log_c"], SLOT["w"]]] = True
    alone = sum(s.pattern) == 1
    for t, on in zip(TYPES, s.pattern):
        if on:
            m[SLOT[EPS[t]]] = not (alone and s.eps_zero)
            m[SLOT[WITHIN[t]]] = t != s.lead
    m[SLOT["delta"]] = s.pattern[1] and not s.delta_zero
    m[SLOT["t_AA"]] = s.t_AA_on
    for name, on in zip(_CROSS_NAMES, s.cross):
        m[SLOT[name]] = on
    return m


def _bounds(name, eps_H_mean=0.0):
    """(lo, hi) of a slot's box in the space the logit acts on (log for log_s / log_c)."""
    if name == "log_s":
        return tuple(np.log(RANGES["s"]))
    if name == "log_c":
        return tuple(np.log(RANGES["c"]))
    if name == "w":
        return W_BOUNDS
    if name in EPS.values():
        return RANGES["eps"]
    if name == "delta":
        return RANGES["delta_min"][0], delta_upper(eps_H_mean)
    if name in WITHIN.values():
        return RANGES["t_within"]
    if name == "t_AA":
        return RANGES["t_AA"]
    return RANGES["t_cross"]


def _to_z(v, lo, hi):
    if hi <= lo:
        return 0.0
    u = np.clip((v - lo) / (hi - lo), _EPS_LOGIT, 1 - _EPS_LOGIT)
    return float(logit(u))


def _from_z(z, lo, hi):
    return float(np.clip(lo + (hi - lo) * expit(z), lo, hi))


def _state_of(draw):
    p = draw.params
    alone = sum(p.pattern) == 1
    eps = {"C": p.eps_C, "H": p.eps_H_mean, "K": p.eps_K}
    eps_zero = alone and eps[p.active_types[0]] == 0
    return DiscreteState(draw.edge, p.pattern, p.lead, p.on_H and p.delta == 0, eps_zero, p.on_H and p.t_AA != 0,
                         (p.t_CH != 0, p.t_CK != 0, p.t_HK != 0))


def encode(draw):
    """(state index, z of shape (14,), mask of shape (14,)): z is unbounded, 0 on inactive slots."""
    k = state_index(_state_of(draw))
    mask = active_slots(k)
    p = draw.params
    value = {"log_s": np.log(p.s), "log_c": np.log(draw.density), "w": float(draw.width)}
    value.update({n: getattr(p, n) for n in CONT if hasattr(p, n)})
    z = np.zeros(len(CONT))
    for i, name in enumerate(CONT):
        if mask[i]:
            z[i] = _to_z(value[name], *_bounds(name, p.eps_H_mean))
    return k, z, mask


def decode(state, z):
    """The draw a state and an unbounded vector stand for; always a canonical host inside the prior."""
    s = STATES[state]
    z = np.asarray(z, float)
    if z.shape != (len(CONT),):
        raise ValueError(f"z must have shape ({len(CONT)},), got {z.shape}")
    mask = active_slots(s)
    v = {}
    eps_H_mean = 0.0
    for i, name in enumerate(CONT):                       # eps_H_mean comes before delta in CONT
        if not mask[i]:
            v[name] = 0.0
            continue
        if name == "delta":
            lo, hi = _bounds(name, eps_H_mean)
            v[name] = _from_z(z[i], lo, hi) if hi >= lo else 0.0       # empty interval: Delta = 0 (ruling R11)
        else:
            v[name] = _from_z(z[i], *_bounds(name, eps_H_mean))
        if name == "eps_H_mean":
            eps_H_mean = v[name]
    v[WITHIN[s.lead]] = 1.0                               # the lead's within-type hopping is fixed at 1
    fields = {n: v[n] for n in FIELDS[1:]}                # every HexParams value but s
    params = HexParams(*s.pattern, s.lead, s=float(np.clip(np.exp(v["log_s"]), *RANGES["s"])), **fields)
    width = int(np.clip(round(v["w"]), *RANGES["w"]))
    return Draw(params, s.edge, width, float(np.clip(np.exp(v["log_c"]), *RANGES["c"])))
