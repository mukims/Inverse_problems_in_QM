import numpy as np
from atlaslib.conformal import coverage, fit_relative, intervals


def test_relative_intervals_reach_nominal_coverage(rng):
    y = rng.uniform(2, 98, 40_000)
    pred = y * (1 + rng.normal(0, 0.05, y.size))
    q = fit_relative(pred[:20_000], y[:20_000], alpha=0.1)
    lo, hi = intervals(pred[20_000:], q)
    assert abs(coverage(lo, hi, y[20_000:]) - 0.90) < 0.02
