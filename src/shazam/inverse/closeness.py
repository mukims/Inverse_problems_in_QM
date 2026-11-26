"""Spectrum closeness: how near a proposed host's transmission is to the target's, in units of seed-to-seed noise.

The inverse model is asked for hosts whose spectrum is close to a target. This module holds the pieces that score that
directly (scripts/evaluate/eval_closeness.py is the scoreboard built on them):

    ratio = misfit(smooth(candidate x), smooth(target x)) / m0(target width, target density)

misfit is the v4-space mean squared difference of the recreation check (shazam.arrangement.misfit), both spectra
averaged over SMOOTH_CHANNELS channels as in sim_reward (the ergodic averaging of ERGO-1), and m0 is the typical misfit
of one impurity arrangement of a host to the median of the others (sim_reward.M0Model, scripts/build/estimate_m0.py).
Two single arrangements of the same ribbon therefore sit somewhat above 1; the scoreboard measures that floor with its
`truth` row. The target's width and density set m0, so every candidate of one target shares one denominator, whatever
width or density the candidate itself proposes.

`nearest_spectra` is the lookup baseline: the stored spectra nearest a target. Pure NumPy.
"""
import numpy as np

from shazam.inverse.sim_reward import SMOOTH_CHANNELS, smooth

PERCENTILES = (5, 25, 50, 75, 95)


def noise_scale(m0, width, density):
    """m0(width, density) of an M0Model for arrays of widths and densities (the model itself takes one host)."""
    a, b, c = m0.coef
    return np.exp(a + b * np.log(np.asarray(width, float)) + c * np.log(np.asarray(density, float)))


def nearest_spectra(library_x, target_x, k=5, smooth_k=SMOOTH_CHANNELS, exclude=None, chunk=256):
    """(rows [n, k], misfit [n, k]): the k library spectra nearest each target, nearest first.

    Distance is the misfit of the smoothed spectra. `exclude` (optional, one library row per target) is never returned:
    the target's own row when the targets are taken from the library. The k rows are chosen with a float32 matrix
    product (|a|^2 + |b|^2 - 2ab) over `chunk` targets at a time; their misfits are then recomputed exactly."""
    library_x = np.asarray(library_x)
    lib = smooth(library_x, smooth_k).astype(np.float32)
    tgt = smooth(np.atleast_2d(target_x), smooth_k)
    n_free = len(lib) - (exclude is not None)
    if not 1 <= k <= n_free:
        raise ValueError(f"k = {k} neighbours asked of {n_free} usable library rows")
    lib2 = np.einsum("ij,ij->i", lib, lib)
    rows = np.empty((len(tgt), k), int)
    dist = np.empty((len(tgt), k))
    for i in range(0, len(tgt), chunk):
        q = tgt[i:i + chunk].astype(np.float32)
        d = lib2[None, :] - 2.0 * (q @ lib.T)                      # + |q|^2, the same for every library row
        if exclude is not None:
            d[np.arange(len(q)), np.asarray(exclude)[i:i + chunk]] = np.inf
        near = np.argpartition(d, k - 1, axis=1)[:, :k]
        for j, cand in enumerate(near):
            exact = np.mean((smooth(library_x[cand], smooth_k) - tgt[i + j]) ** 2, axis=1)
            order = np.lexsort((cand, exact))                       # by misfit, ties by library row
            rows[i + j], dist[i + j] = cand[order], exact[order]
    return rows, dist


def _spread(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(n=0, median=None, percentiles=None)
    return dict(n=int(len(v)), median=float(np.median(v)), percentiles=[float(q) for q in np.percentile(v, PERCENTILES)])


def _best(ratio):
    """Per target: the smallest finite ratio (NaN when the target has no candidate) and its column."""
    filled = np.where(np.isfinite(ratio), ratio, np.inf)
    col = filled.argmin(axis=1)
    best = filled[np.arange(len(ratio)), col]
    return np.where(np.isfinite(best), best, np.nan), col


def summarise(ratio, right=None, groups=None):
    """The scoreboard numbers of one method from its ratios [targets, candidates], NaN where a target has no such
    candidate (fewer were proposed, or its simulation failed).

    first / best / all: the spread (n, median, PERCENTILES) of the rank-1 candidate, of each target's closest candidate
    and of every candidate. best_within_1 / best_within_2: share of ALL targets whose closest candidate is within 1x / 2x
    the seed noise (a target without candidates is a miss). With `right` (bool, same shape: the candidate has the true
    family): first_right_family, any_right_family and closest_is_right_family, again over all targets. With `groups`
    (one label per target, e.g. the site-type pattern): by_group[label] = n, first_median, best_median, best_within_2."""
    ratio = np.atleast_2d(np.asarray(ratio, float))
    have = np.isfinite(ratio)
    best, col = _best(ratio)
    with np.errstate(invalid="ignore"):
        within = lambda x: np.nan_to_num(best, nan=np.inf) <= x
    out = dict(n_targets=int(len(ratio)), candidates_per_target=float(have.sum() / len(ratio)),
               first=_spread(ratio[:, 0]), best=_spread(best), all=_spread(ratio),
               best_within_1=float(within(1).mean()), best_within_2=float(within(2).mean()))
    if right is not None:
        right = np.atleast_2d(np.asarray(right, bool)) & have
        out.update(first_right_family=float(right[:, 0].mean()), any_right_family=float(right.any(1).mean()),
                   closest_is_right_family=float(right[np.arange(len(ratio)), col].mean()))
    if groups is not None:
        groups = np.asarray(groups)
        out["by_group"] = {}
        for g in dict.fromkeys(groups.tolist()):
            s = groups == g
            out["by_group"][g] = dict(n=int(s.sum()), first_median=_spread(ratio[s, 0])["median"],
                                      best_median=_spread(best[s])["median"], best_within_2=float(within(2)[s].mean()))
    return out
