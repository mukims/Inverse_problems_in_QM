"""Simulator reward for phase 2 of reward fine-tuning (§6 of
docs/superpowers/specs/2026-11-04-reward-finetuning-design.md): simulate a candidate once with a fresh disorder
seed and score how well it recreates the target spectrum.

    r = -misfit(smooth(simulated x), smooth(target x)) / m0(width, density)

misfit is the v4-space MSE of the recreation check (shazam.arrangement.misfit). Smoothing over SMOOTH_CHANNELS channels
leans on the ergodic hypothesis (ERGO-1): if fluctuations decorrelate within a channel, averaging k channels cuts
the seed-to-seed noise by about sqrt(k), so one seed per candidate is enough. m0 is the typical seed-to-seed
misfit of one host (M0Model, fitted by scripts/build/estimate_m0.py).
"""
import multiprocessing as mp
from dataclasses import asdict, dataclass

import numpy as np

from shazam.arrangement import misfit
from shazam.inverse.verify import simulate_inputs
from tbribbon.parent import device_cells_for
from tbribbon.parent_data import n_impurities
from tbribbon.parent_prior import in_prior

SMOOTH_CHANNELS = 5


def smooth(x, k=SMOOTH_CHANNELS):
    """Centred moving average over k channels along the last axis, edges replicated (k = 1 returns a copy)."""
    x = np.asarray(x, dtype=np.float64)
    if k <= 1:
        return x.copy()
    pad = [(0, 0)] * (x.ndim - 1) + [(k // 2, k - 1 - k // 2)]
    c = np.cumsum(np.pad(x, pad, mode="edge"), axis=-1)
    c = np.concatenate([np.zeros(x.shape[:-1] + (1,)), c], axis=-1)
    return (c[..., k:] - c[..., :-k]) / k


@dataclass
class M0Model:
    """log m0 = a + b log(width) + c log(density)."""
    coef: list

    def __call__(self, width, density):
        a, b, c = self.coef
        return float(np.exp(a + b * np.log(width) + c * np.log(density)))

    @classmethod
    def fit(cls, widths, densities, m0s):
        A = np.column_stack([np.ones(len(widths)), np.log(widths), np.log(densities)])
        coef, *_ = np.linalg.lstsq(A, np.log(m0s), rcond=None)
        return cls([float(v) for v in coef])

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, d):
        return cls(list(d["coef"]))


def reward(sim_x, target_x, m0, k=SMOOTH_CHANNELS):
    return -misfit(smooth(sim_x, k), smooth(target_x, k)) / m0


def in_box(draw):
    """Inside the prior box (tbribbon.parent_prior.in_prior): only such candidates are simulated and rewarded."""
    return in_prior(draw)


def _one(job):
    draw, seed = job
    try:
        n_imp = n_impurities(draw, device_cells_for(draw.params, draw.edge))
        return simulate_inputs(draw, n_imp, [seed], n_jobs=1)[0]
    except Exception:                                    # a failed simulation is a missing reward, not a failed run
        return None


def simulate_candidates(draws, seeds, n_jobs=1):
    """One v4 input (416,) per candidate, one disorder seed each, in order; None for a candidate outside the prior box
    or whose simulation raised. n_jobs > 1 forks workers (one BLAS thread each, as verify.simulate_inputs sets)."""
    jobs = [(d, int(s)) for d, s in zip(draws, seeds)]
    ok = [in_box(d) for d, _ in jobs]
    todo = [j for j, k in zip(jobs, ok) if k]
    if n_jobs <= 1 or len(todo) <= 1:
        done = [_one(j) for j in todo]
    else:
        with mp.get_context("fork").Pool(min(n_jobs, len(todo))) as pool:
            done = pool.map(_one, todo, chunksize=1)
    it = iter(done)
    return [next(it) if k else None for k in ok]
