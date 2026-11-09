"""Bayesian active learning for the inverse model (docs/superpowers/specs/2026-11-04-active-learning-design.md).

New labelled training draws are simulated from a prior tilted towards the cells where the model is weak AND still
learning; the model is then retrained on all draws with importance weights that bring the training set back to the prior.
Simulation only makes training data: nothing here runs after the model.

A cell is the family with edge (`reward.FAMILY_OF`, the 156 states without the unit type) times a width bin. Within a cell
the proposal is exactly the prior, so the importance weight of a row is a per-cell constant.
"""
import numpy as np

from shazam.inverse.encoding import state_index, state_of
from shazam.inverse.reward import FAMILY_OF, N_FAMILIES
from tbribbon.parent_prior import sample_prior

WIDTH_EDGES = (4, 8, 13, 18, 23)            # width bins 4-7, 8-12, 13-17, 18-22
N_WBINS = len(WIDTH_EDGES) - 1
N_CELLS = N_FAMILIES * N_WBINS
GAIN_WEIGHTS = dict(family=0.5, width=0.3, log_s=0.2)
PRIOR_SHARE = 0.5                           # q = PRIOR_SHARE * prior + (1 - PRIOR_SHARE) * tilted, so p / q <= 2
TOP_SHARE = 0.6                             # the tilted half puts at least this much mass in the top-quartile-gain cells
MAX_TRIES = 100_000                         # proposal draws per accepted draw; reaching it means a broken accept table


def width_bin(w):
    w = int(w)
    if not WIDTH_EDGES[0] <= w < WIDTH_EDGES[-1]:
        raise ValueError(f"width {w} outside {WIDTH_EDGES[0]}..{WIDTH_EDGES[-1] - 1}")
    return int(np.searchsorted(WIDTH_EDGES, w, side="right")) - 1


def cell_of(draw):
    return int(FAMILY_OF[state_index(state_of(draw))]) * N_WBINS + width_bin(draw.width)


def family_of_cell(cell):
    return np.asarray(cell) // N_WBINS


def shrink_rates(k, n, groups):
    """Beta-binomial empirical Bayes: posterior-mean rates of k errors in n draws, the prior of each group fitted to its
    cells by the method of moments (a group whose cells look alike pools them hard)."""
    k, n, groups = (np.asarray(a) for a in (k, n, groups))
    out = np.zeros(len(k), float)
    for g in np.unique(groups):
        m = groups == g
        mean = (k[m].sum() + 0.5) / (n[m].sum() + 1.0)
        raw = (k[m] + 0.5) / (n[m] + 1.0)
        var = np.average((raw - mean) ** 2, weights=n[m]) if m.sum() > 1 else 0.0
        binom = mean * (1 - mean) * np.mean(1.0 / np.maximum(n[m], 1))
        between = max(var - binom, 1e-6)
        strength = np.clip(mean * (1 - mean) / between - 1.0, 1e-3, 1e6)        # alpha + beta of the group's Beta prior
        out[m] = (k[m] + strength * mean) / (n[m] + strength)
    return out


def shrink_means(est, se, groups):
    """Normal-normal empirical Bayes: each estimate pulled towards its group's precision-weighted mean in proportion to
    its noise variance against the group's between-cell variance."""
    est, se, groups = (np.asarray(a, float) for a in (est, se, groups))
    out = np.zeros(len(est))
    for g in np.unique(groups):
        m = groups == g
        v = np.maximum(se[m] ** 2, 1e-12)
        mu = np.sum(est[m] / v) / np.sum(1 / v)
        tau2 = max(np.mean((est[m] - mu) ** 2) - np.mean(v), 0.0)
        shrink = v / (v + tau2) if tau2 > 0 else np.ones_like(v)
        out[m] = mu + (1 - shrink) * (est[m] - mu)
    return out


def expected_gain(err, slope, weights=GAIN_WEIGHTS):
    """Per cell: sum over metrics of weight * error now * max(learning slope, 0). A cell that is wrong but no longer
    improves with data gets nothing: its errors look like what the spectrum cannot tell apart."""
    return sum(w * np.asarray(err[k]) * np.maximum(np.asarray(slope[k]), 0.0) for k, w in weights.items())


GAMMA_MAX = 20.0


def _tilt(p, gain, gamma):
    t = p * (gain / gain[gain > 0].mean()) ** gamma          # gamma 0: the prior; large gamma: the highest-gain cells
    return t / t.sum()


def tilted_proposal(p, gain, prior_share=PRIOR_SHARE, top_share=TOP_SHARE):
    """(q, gamma): the cell distribution of the proposal. tilted ∝ p (gain / mean gain)^gamma, with the smallest gamma
    that puts `top_share` of the tilted mass in the top-quartile-gain cells (gamma 0 if the prior already does);
    q = prior_share p + (1 - prior_share) tilted, so p / q <= 1 / prior_share everywhere."""
    p, gain = np.asarray(p, float), np.asarray(gain, float)
    p = p / p.sum()
    if not np.any(gain > 0):
        return p.copy(), 0.0
    top = gain >= np.quantile(gain[p > 0], 0.75)
    mass = lambda g: _tilt(p, gain, g)[top].sum()
    lo, hi = 0.0, GAMMA_MAX
    if mass(lo) >= top_share:
        gamma = lo
    else:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if mass(mid) < top_share else (lo, mid)
        gamma = hi
    return prior_share * p + (1 - prior_share) * _tilt(p, gain, gamma), float(gamma)


def acceptance(p, q):
    """Per-cell acceptance probabilities that turn prior draws into proposal draws: (q / p) / max(q / p)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    ratio = np.where(p > 0, q / np.where(p > 0, p, 1.0), 0.0)
    return ratio / ratio.max()


def sample_proposal(rng, accept, sampler=sample_prior, cell=cell_of):
    """One draw of the proposal: prior draws (from `rng`) kept with probability accept[cell]."""
    for _ in range(MAX_TRIES):
        d = sampler(rng)
        if rng.random() < accept[cell(d)]:
            return d
    raise RuntimeError(f"no proposal draw accepted in {MAX_TRIES} tries")


def pooled_weights(cells, prior_cells, n_cells=N_CELLS):
    """Per-row weights p(cell) / m(cell) that bring the pooled rows (`cells`) back to the cell distribution of the
    prior-drawn rows (`prior_cells`, a subset drawn from the prior). Both are estimated from the rows themselves, so the
    generator's own rejections cancel. Mean 1 by construction."""
    cells, prior_cells = np.asarray(cells), np.asarray(prior_cells)
    p = np.bincount(prior_cells, minlength=n_cells) / len(prior_cells)
    m = np.bincount(cells, minlength=n_cells) / len(cells)
    w = np.where(m > 0, p / np.where(m > 0, m, 1.0), 0.0)
    return w[cells]


def cells_of_arrays(arrays):
    """The cell of every row of a loaded split (`shazam.inverse.data.load_split` arrays)."""
    from shazam.inverse import data
    return np.array([cell_of(d) for d in data.draws(arrays, slice(None))], dtype=np.int64)
