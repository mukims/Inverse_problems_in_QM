"""Updated probabilities (spec section 8, step 4).

Each realistic candidate i has a model probability p_i and the log-likelihood log_L_i of the target under its 50-seed
check ensemble. The mass p0 = 1 - sum(p_i) of everything else is paired with a background log-likelihood log_L0:

    p_i_updated = p_i L_i^(1/kappa) / (sum_j p_j L_j^(1/kappa) + p0 L0^(1/kappa))

It is computed in log space (logsumexp over the candidates and the background term), so no division by zero can happen:
the spec's 1e-12 guard is unnecessary and is left out. When every candidate fits poorly (log_L_i << log_L0) the
background term dominates and all updated probabilities fall towards 0 instead of being renormalised to 1.

`fit_update` chooses kappa and log_L0 once, on validation draws, to minimise the expected calibration error (ECE) of the
updated probabilities against the capture hits.
"""
import numpy as np
from scipy.special import logsumexp

from shazam.inverse.calibration import expected_calibration_error

__all__ = ["updated_probabilities", "fit_update", "N_LOG_L0", "TIE_TOL", "UPDATE_KAPPAS", "P0_TOL"]

TIE_TOL = 1e-12   # ECE values closer than this are ties (floating-point round-off)
# Real log_L are sums over 416 channels (thousands to tens of thousands), so the grid of atlaslib.lookup.KAPPAS (1 to 1000)
# is far too narrow here and kappa would pin at 1000 (ruling R23); lookup.KAPPAS itself stays untouched.
UPDATE_KAPPAS = np.geomspace(1.0, 1e6, 121)
P0_TOL = 1e-9     # p0 = 1 - sum(p) may be this negative from round-off (clamped to 0); more negative is an upstream bug
N_LOG_L0 = 41     # default log_L0 grid: this many quantiles of the observed log_L


def _log(x):
    with np.errstate(divide="ignore"):
        return np.log(x)


def _clean_p0(p0):
    """p0 clamped to >= 0 (1 - sum(p) can be -1e-16 from round-off); ValueError when it is really negative or not finite."""
    p0 = float(p0)
    if not np.isfinite(p0) or p0 < -P0_TOL:
        raise ValueError(f"p0 must be a finite mass >= 0 (up to round-off), got {p0}")
    return max(p0, 0.0)


def _check(p, log_L):
    """ValueError on NaN or negative p, NaN or +inf log_L (-inf, a zero likelihood, is allowed)."""
    if np.isnan(p).any() or (p < 0).any():
        raise ValueError("p must not contain NaN or negative values")
    if np.isnan(log_L).any() or (log_L == np.inf).any():
        raise ValueError("log_L must not contain NaN or +inf")


def _exp_diff(a, b):
    """exp(a - b), 0 where a is -inf or b is -inf (no mass at all), without warnings."""
    with np.errstate(invalid="ignore", over="ignore"):
        out = np.exp(a - b)
    return np.where(np.isfinite(a) & np.isfinite(b), out, 0.0)


def updated_probabilities(p, log_L, p0, kappa, log_L0):
    """p_i L_i^(1/kappa) / (sum_j p_j L_j^(1/kappa) + p0 L0^(1/kappa)), as an array like `p` (empty in, empty out).

    Computed with logsumexp over the stacked terms log p_i + log_L_i / kappa and log p0 + log_L0 / kappa; p_i = 0 and
    p0 = 0 (log 0 = -inf) are fine, a zero total mass gives all zeros, and a round-off p0 slightly below 0 is clamped to 0 (a clearly negative p0, NaN or
    +inf log_L raise ValueError). No 1e-12 guard is needed in log space."""
    p, log_L = np.asarray(p, float), np.asarray(log_L, float)
    if p.shape != log_L.shape or p.ndim != 1:
        raise ValueError("p and log_L must be 1-D arrays of equal length")
    _check(p, log_L)
    p0 = _clean_p0(p0)
    if not np.isfinite(log_L0):
        raise ValueError("log_L0 must be finite")
    if len(p) == 0:
        return np.zeros(0)
    a = np.append(_log(p) + log_L / kappa, _log(p0) + log_L0 / kappa)
    with np.errstate(invalid="ignore", divide="ignore"):
        total = logsumexp(a) if np.isfinite(a).any() else -np.inf
    return _exp_diff(a[:-1], total)


def _pooled(records):
    """Concatenate the records once: (log p, log_L, hits, group index of each candidate, log p0 per non-empty group)."""
    recs = [r for r in records if len(r["p"])]
    if not recs:
        raise ValueError("fit_update needs at least one candidate")
    for r in recs:
        if not (len(r["p"]) == len(r["log_L"]) == len(r["hits"])):
            raise ValueError("p, log_L and hits must have the same length in every record")
        _check(np.asarray(r["p"], float), np.asarray(r["log_L"], float))
    sizes = np.array([len(r["p"]) for r in recs])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return (_log(np.concatenate([np.asarray(r["p"], float) for r in recs])),
            np.concatenate([np.asarray(r["log_L"], float) for r in recs]),
            np.concatenate([np.asarray(r["hits"], bool) for r in recs]),
            np.repeat(np.arange(len(recs)), sizes), starts,
            _log(np.array([_clean_p0(r["p0"]) for r in recs])))


def _group_logsumexp(a, starts, group):
    """logsumexp of `a` within each contiguous group (-inf for a group with no mass)."""
    m = np.maximum.reduceat(a, starts)
    ms = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        return np.where(np.isfinite(m), ms + np.log(np.add.reduceat(np.exp(a - ms[group]), starts)), -np.inf)


def fit_update(records, kappas=UPDATE_KAPPAS, log_L0_grid=None):
    """Grid search of (kappa, log_L0) minimising the ECE of the updated probabilities against the hits.

    Each record is a dict with `p` (model probabilities of the returned candidates), `log_L`, `p0` (the mass of
    everything else) and `hits` (bool, did the candidate capture the true host), all arrays of the candidates' length.
    The probabilities and hits of all records are pooled. `kappas` defaults to UPDATE_KAPPAS (1 to 1e6, 121 values; see its
    comment). `log_L0_grid` defaults to 41 quantiles of all observed log_L.
    Ties (ECE within TIE_TOL = 1e-12) go to the smallest kappa, then the smallest log_L0.

    Returns {"kappa", "log_L0", "ece"} (plain floats), "ece_model" (the ECE of the raw model probabilities) and
    "kappa_at_edge" (bool: the chosen kappa is the first or last grid value, so the grid may be too narrow)."""
    log_p, log_L, hits, group, starts, log_p0 = _pooled(records)
    kappas = np.unique(np.asarray(kappas, float))
    if log_L0_grid is None:
        finite = log_L[np.isfinite(log_L)]
        if len(finite) == 0:
            raise ValueError("fit_update needs at least one finite log_L")
        log_L0_grid = np.quantile(finite, np.linspace(0, 1, N_LOG_L0))
    grid = np.unique(np.asarray(log_L0_grid, float))
    ece = np.empty((len(kappas), len(grid)))
    for i, kappa in enumerate(kappas):
        a = log_p + log_L / kappa                                            # (n,)
        lse = _group_logsumexp(a, starts, group)                              # (G,)
        bg = log_p0[:, None] + grid[None, :] / kappa                          # (G, M)
        denom = np.logaddexp(lse[:, None], bg)                                # (G, M)
        u = _exp_diff(a[:, None], denom[group])                               # (n, M)
        for j in range(len(grid)):
            ece[i, j] = expected_calibration_error(u[:, j], hits)
    # round-off in the probabilities moves the ECE by ~1e-16: differences below TIE_TOL are ties
    i, j = np.unravel_index(np.argmax(ece <= ece.min() + TIE_TOL), ece.shape)   # first tie: smallest kappa, log_L0
    model_p = np.exp(log_p)
    return dict(kappa=float(kappas[i]), log_L0=float(grid[j]), ece=float(ece[i, j]),
                ece_model=float(expected_calibration_error(model_p, hits)), kappa_at_edge=bool(i in (0, len(kappas) - 1)))
