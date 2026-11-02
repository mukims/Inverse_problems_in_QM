"""Simulator check and likelihood of a candidate host (spec section 8, steps 1, 2 and 4).

A candidate host is recreated by simulating fresh disordered ribbons of it (`simulate_inputs`). The target recreates if its
misfit to the ensemble median is no larger than the 95th percentile of the ensemble's own leave-one-out misfits
(`recreation_check`); `log_likelihood` scores the target under the ensemble's per-channel median and spread.

Simulation and statistics only: no model or torch imports.
"""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from typing import NamedTuple

import numpy as np

from atlaslib.lookup import S0
from shazam.arrangement import misfit
from tbribbon import parent_data
from tbribbon.parent_data import simulate_raw, to_input

VERIFY_SEEDS = tuple(range(10**9, 10**9 + 50))             # the 50-seed check ensemble; never used in training or tuning
FINAL_SEEDS = tuple(range(10**9 + 50, 10**9 + 100))        # the one final check after refinement
OPT_SEEDS = tuple(range(2 * 10**9, 2 * 10**9 + 20))        # the 20 common-random-number seeds of the refinement
QUANTILE = 0.95


def _limit_blas():
    """One BLAS thread per worker, so n_jobs workers do not oversubscribe the cores (as generate_parent.py)."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1)
    except ImportError:
        return None


_LIMIT = None
_HOST = None          # (ribbon, LeadCache) of the host being simulated: set in the parent before the workers fork


def _init_worker():
    global _LIMIT
    _LIMIT = _limit_blas()


def _one(args):
    draw, n_imp, seed, n_cells = args
    rb, lc = _HOST
    return to_input(simulate_raw(draw, n_imp, seed, n_cells=n_cells, rb=rb, lc=lc))


def simulate_inputs(draw, n_imp, seeds, n_jobs=1, n_cells=None):
    """v4 inputs (len(seeds), 416) float32 of the disordered device of `draw` with `n_imp` impurities, one per seed, in order.

    n_cells None = the device length of the host (device_cells_for). The ribbon and its lead self-energies depend on the
    host only, so they are built once per call (one BLAS thread, as in the workers: bit-identical to building them per
    seed) and shared with the forked workers. n_jobs > 1 uses forked workers with one BLAS thread each; a worker that dies
    raises BrokenProcessPool instead of hanging the call.
    """
    global _HOST
    jobs = [(draw, int(n_imp), int(s), n_cells) for s in seeds]
    with _limit_blas() or nullcontext():
        _HOST = parent_data.host_leads(draw)
    try:
        if n_jobs <= 1 or len(jobs) <= 1:
            with _limit_blas() or nullcontext():
                rows = [_one(j) for j in jobs]
        else:
            with ProcessPoolExecutor(min(n_jobs, len(jobs)), mp_context=mp.get_context("fork"),
                                     initializer=_init_worker) as pool:
                rows = list(pool.map(_one, jobs))                      # map keeps the order of `jobs`
    finally:
        _HOST = None
    return np.asarray(rows, dtype=np.float32).reshape(len(jobs), -1)


class CheckResult(NamedTuple):
    misfit: float          # target against the ensemble median
    threshold: float       # 95th percentile of the members' leave-one-out misfits
    p_value: float         # (1 + #{members with misfit >= the target's}) / (n + 1)
    recreated: bool        # misfit <= threshold


def recreation_check(target_x, ensemble_x):
    """Does the ensemble (n, 416) recreate the target (416,)? Member j is judged against the median of the other n - 1.

    With n = 50 the 0.95 quantile sits between the 47th and 48th member misfits, so a target that really is one more
    member fails about 6% of the time, not 5% (measured on Gaussian vectors); the p-value is close to uniform."""
    ens = np.asarray(ensemble_x, dtype=np.float64)
    n = len(ens)
    if n < 3:
        raise ValueError("need at least 3 ensemble members")
    target_misfit = misfit(np.median(ens, axis=0), target_x)
    member = np.array([misfit(np.median(np.delete(ens, j, axis=0), axis=0), ens[j]) for j in range(n)])
    threshold = float(np.quantile(member, QUANTILE))
    p = (1 + int(np.sum(member >= target_misfit))) / (n + 1)
    return CheckResult(target_misfit, threshold, float(p), bool(target_misfit <= threshold))


def log_likelihood(target_x, ensemble_x):
    """Per channel mu = median, sigma = sqrt(std^2 + S0^2); the sum over channels of -((x - mu) / sigma)^2 / 2 - log(sigma)."""
    ens = np.asarray(ensemble_x, dtype=np.float64)
    mu = np.median(ens, axis=0)
    sigma = np.sqrt(ens.std(axis=0) ** 2 + S0 ** 2)
    z = (np.asarray(target_x, dtype=np.float64) - mu) / sigma
    return float(np.sum(-0.5 * z ** 2 - np.log(sigma)))
