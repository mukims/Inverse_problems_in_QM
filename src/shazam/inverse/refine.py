"""Trust-region refinement of a candidate host (spec section 8, step 3).

`trust_region_minimise` is a small Gaussian-process Bayesian optimiser confined to a box around the incumbent best point.
`refine` applies it to the continuous parameters of a host whose simulator recreation check failed.

Ruling for this task: the search runs over the ACTIVE continuous slots of `encode(draw)` (unbounded, logit units) with
the discrete state fixed and the lead hopping fixed at 1 (the lead's slot is inactive in the encoding). The impurity
count `n_imp` is kept FIXED as given (it is part of the candidate; the density moves only through the device length), so
the `log_c` slot is excluded from the search; the `w` slot is excluded because the width is an integer. Only the host
parameters (log_s, on-site energies, Delta, hoppings) are searched; inactive and excluded slots stay as encoded.

One evaluation simulates the decoded draw on all `seeds` (common random numbers, so the objective is smooth), takes the
per-channel median over seeds and returns its misfit to the target. Per ruling R5, when the budget runs out without
passing the threshold `refine` simply reports the best point; the final check on fresh seeds is the pipeline's job.
"""
import warnings
from typing import Callable, NamedTuple

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from shazam.arrangement import misfit
from shazam.inverse.encoding import SLOT, decode, encode
from shazam.inverse.verify import OPT_SEEDS, simulate_inputs
from tbribbon.parent_prior import Draw

RADIUS0 = 0.5                    # initial per-coordinate half-width of the trust box (unbounded units)
RADIUS_RANGE = (0.01, 4.0)       # the box never collapses to a point or grows past the sensible range of a logit
N_CANDIDATES = 512               # uniform points in the trust box scored by expected improvement
LENGTH_SCALE = (2.0, (0.5, 30.0))  # GP length-scale start and bounds, in unbounded units: keeps 3-point fits from collapsing
PATIENCE = 2                     # consecutive successes (failures) before the radius doubles (halves)
Z_MAX = 8.0                      # proposals stay where the logit decode still responds (|z| > ~7 saturates the box)
_EXCLUDED = (SLOT["log_c"], SLOT["w"])


class RefineResult(NamedTuple):
    draw: Draw
    misfit: float
    n_simulations: int
    stopped_early: bool


def _expected_improvement(gp, points, best):
    mu, sd = gp.predict(points, return_std=True)
    sd = np.maximum(sd, 1e-12)
    gain = best - mu
    u = gain / sd
    return gain * norm.cdf(u) + sd * norm.pdf(u)


def _propose(x_obs, y_obs, centre, radius, rng):
    points = np.clip(centre + radius * rng.uniform(-1.0, 1.0, size=(N_CANDIDATES, centre.size)), -Z_MAX, Z_MAX)
    if len(y_obs) < 2:
        return points[0]
    kernel = Matern(LENGTH_SCALE[0] * np.ones(centre.size), LENGTH_SCALE[1], nu=2.5) + WhiteKernel(1e-3, (1e-8, 1.0))
    gp = GaussianProcessRegressor(kernel, normalize_y=True, random_state=int(rng.integers(2**31)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(x_obs, y_obs)
    return points[int(np.argmax(_expected_improvement(gp, points, float(np.min(y_obs)))))]


def trust_region_minimise(f: Callable[[np.ndarray], float], x0, n_evals: int, threshold: float | None, rng):
    """Minimise f from x0 with at most n_evals calls (the first is f(x0)); returns (best x, best f, history, stopped_early).

    Each step fits a GP (Matern 5/2 + white noise, normalised targets) to everything seen and evaluates the expected-
    improvement maximiser among 512 uniform points in the box of half-width `radius` around the best point. The radius
    starts at 0.5, doubles after 2 consecutive improvements of the best and halves after 2 consecutive non-improvements.
    Stops early when the best value drops below `threshold` (None = never).
    """
    x0 = np.asarray(x0, float)

    def value(x):                # a failed simulation (nan/inf) becomes a large finite penalty instead of poisoning the GP
        v = float(f(x))
        return v if np.isfinite(v) else 10.0 * max([abs(y) for y in ys] + [1.0])

    ys = []
    xs, ys = [x0], [value(x0)]
    best_i, radius, wins, losses = 0, RADIUS0, 0, 0
    stopped = threshold is not None and ys[0] < threshold
    while len(ys) < n_evals and not stopped:
        x = _propose(np.array(xs), np.array(ys), xs[best_i], radius, rng)
        xs.append(x)
        ys.append(value(x))
        if ys[-1] < ys[best_i]:
            best_i, wins, losses = len(ys) - 1, wins + 1, 0
        else:
            wins, losses = 0, losses + 1
        if wins == PATIENCE:
            radius, wins = min(radius * 2, RADIUS_RANGE[1]), 0
        elif losses == PATIENCE:
            radius, losses = max(radius / 2, RADIUS_RANGE[0]), 0
        stopped = threshold is not None and ys[best_i] < threshold
    return xs[best_i], ys[best_i], ys, stopped


def refine(draw, n_imp, target_x, threshold, budget=200, seeds=OPT_SEEDS, n_jobs=1, n_cells=None, rng=None):
    """Tune the continuous host parameters of `draw` (see the module docstring) to minimise the median misfit to target_x.

    n_evals = budget // len(seeds) evaluations of len(seeds) simulations each; stops early below `threshold`.
    `rng` is a numpy Generator (default: seed 0, for determinism). Returns the best draw found.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    state, z0, mask = encode(draw)
    free = np.flatnonzero(mask & ~np.isin(np.arange(mask.size), _EXCLUDED))
    target_x = np.asarray(target_x)

    def to_draw(x):
        z = z0.copy()
        z[free] = x
        return decode(state, z)

    def objective(x):
        sims = simulate_inputs(to_draw(x), n_imp, seeds, n_jobs=n_jobs, n_cells=n_cells)
        return float(misfit(np.median(sims, axis=0), target_x))

    x, best, history, stopped = trust_region_minimise(objective, z0[free], budget // len(seeds), threshold, rng)
    return RefineResult(to_draw(x), best, len(history) * len(seeds), stopped)
