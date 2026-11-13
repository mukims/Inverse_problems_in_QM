"""Realistic hosts: is a tight-binding host near a known material? (spec section 7, step 6: the physical criterion.)

Criterion (the user's choice, 2026-11-10): a host is realistic when it lies near one of the known materials of spec 4.3
(tbribbon.parent.NAMED). "Near" compares the host's physical quantities in eV (reward.EV_NAMES: on-site energies and
hoppings times the energy scale s, switched-off ones 0), so the unit type and the scale s do not enter separately:

    distance(host, material) = max over the quantities q of |q_host - q_material| / tol_q
    tol_q = max(abs_tol, rel_tol * |q_material|)

A host of another site-type pattern is infinitely far. `distance <= 1` is "near": every quantity within rel_tol of the
material's value, and every quantity the material does not have (0 eV) below abs_tol. The edge, the width and the
impurity density are properties of the ribbon, not of the material, and take no part.

REL_TOL, ABS_TOL and the catalogue of six (the toy triangular and kagome lattices at 1 eV included) were confirmed by
the user on 2026-11-10 and 11; they are a working definition, not measured tolerances of these materials.
candidates.is_realistic applies this criterion. Pure NumPy.

sample_near draws from the training prior restricted to within `radius` tolerances of one material (the data of
generate_parent.py --near).
"""
import numpy as np

from shazam.inverse.reward import EV_NAMES
from tbribbon.parent import NAMED, named_material
from tbribbon.parent_prior import FIELDS, sample_prior

REL_TOL = 0.10            # share of a material's own value
ABS_TOL = 0.10            # eV, for quantities the material does not have
_COLUMNS = [FIELDS.index(n) for n in EV_NAMES]              # EV_NAMES in the stored `params` columns
_S = FIELDS.index("s")
MAX_TRIES = 1_000_000     # prior draws per accepted draw of sample_near; the rarest region (hBN, 2 tolerances) needs ~600


def _ev(params):
    return np.array([getattr(params, n) * params.s for n in EV_NAMES])


def physical_ev(draw):
    """The host's EV_NAMES in eV (model value times s; switched off = 0)."""
    return _ev(draw.params)


def _catalogue(materials):
    """(patterns [m, 3] bool, values [m, q] eV) of the named materials."""
    unknown = [n for n in materials if n not in NAMED]
    if unknown:
        raise ValueError(f"unknown material {unknown[0]!r}; choose from {NAMED}")
    hosts = [named_material(n) for n in materials]
    return np.array([h.pattern for h in hosts], bool), np.array([_ev(h) for h in hosts])


def _distances(ev, pattern, materials, rel_tol, abs_tol):
    """[n, m] distance of each host (ev [n, q], pattern [n, 3]) to each material; inf across patterns."""
    if not (rel_tol >= 0 and abs_tol > 0):
        raise ValueError(f"tolerances need rel_tol >= 0 and abs_tol > 0, got {rel_tol} and {abs_tol}")
    m_pattern, m_ev = _catalogue(materials)
    tol = np.maximum(abs_tol, rel_tol * np.abs(m_ev))                                    # [m, q]
    d = (np.abs(ev[:, None, :] - m_ev[None, :, :]) / tol[None, :, :]).max(axis=2)
    same = (np.asarray(pattern, bool)[:, None, :] == m_pattern[None, :, :]).all(axis=2)
    return np.where(same, d, np.inf)


def _nearest(dist, materials):
    best = dist.argmin(axis=1)
    d = dist[np.arange(len(dist)), best]
    names = np.array([materials[j] if np.isfinite(v) else None for j, v in zip(best, d)], dtype=object)
    return names, d


def material_distance(draw, name, rel_tol=REL_TOL, abs_tol=ABS_TOL):
    """Distance of the host of `draw` to the named material (see the module docstring); inf for another pattern."""
    return float(_distances(physical_ev(draw)[None], [draw.params.pattern], (name,), rel_tol, abs_tol)[0, 0])


def nearest_material(draw, materials=NAMED, rel_tol=REL_TOL, abs_tol=ABS_TOL):
    """(name, distance) of the nearest known material; (None, inf) when none has the host's site-type pattern. Ties go
    to the first in `materials`."""
    names, d = _nearest(_distances(physical_ev(draw)[None], [draw.params.pattern], tuple(materials), rel_tol, abs_tol),
                        tuple(materials))
    return names[0], float(d[0])


def is_near_known(draw, materials=NAMED, rel_tol=REL_TOL, abs_tol=ABS_TOL):
    """True when the host is within one tolerance of a known material."""
    return nearest_material(draw, materials, rel_tol, abs_tol)[1] <= 1.0


def nearest_material_ev(ev, pattern, materials=NAMED, rel_tol=REL_TOL, abs_tol=ABS_TOL):
    """(names [n] (None where no material shares the pattern), distances [n]) for hosts given as physical vectors
    `ev` [n, len(EV_NAMES)] in eV (physical_ev) and site-type patterns `pattern` [n, 3] (C, H, K on)."""
    ev = np.asarray(ev, float).reshape(-1, len(EV_NAMES))
    pattern = np.asarray(pattern, bool).reshape(-1, 3)
    return _nearest(_distances(ev, pattern, tuple(materials), rel_tol, abs_tol), tuple(materials))


def nearest_material_rows(arrays, materials=NAMED, rel_tol=REL_TOL, abs_tol=ABS_TOL):
    """nearest_material_ev for the rows of a loaded split (its `on` and `params` arrays): the same values as
    nearest_material of each row's draw."""
    params = np.asarray(arrays["params"], float)
    return nearest_material_ev(params[:, _COLUMNS] * params[:, [_S]], arrays["on"], materials, rel_tol, abs_tol)


def near_known_rows(arrays, materials=NAMED, rel_tol=REL_TOL, abs_tol=ABS_TOL):
    """Boolean [n]: is_near_known of each row of a loaded split."""
    return nearest_material_rows(arrays, materials, rel_tol, abs_tol)[1] <= 1.0


def sample_near(rng, material, radius=2.0, rel_tol=REL_TOL, abs_tol=ABS_TOL, sampler=sample_prior, max_tries=MAX_TRIES):
    """(draw, tries): one draw of the prior restricted to distance <= radius of the named material. Prior draws are
    taken from `rng` until one is inside; `tries` counts them (the accepted one included), so over many draws
    1 / mean(tries) estimates the prior mass of the region."""
    if material not in NAMED:
        raise ValueError(f"unknown material {material!r}; choose from {NAMED}")
    if not radius > 0:
        raise ValueError(f"radius must be positive, got {radius}")
    pattern = named_material(material).pattern
    for tries in range(1, max_tries + 1):
        draw = sampler(rng)
        if draw.params.pattern == pattern and material_distance(draw, material, rel_tol, abs_tol) <= radius:
            return draw, tries
    raise RuntimeError(f"no draw within {radius} tolerances of {material} in {max_tries} prior draws")
