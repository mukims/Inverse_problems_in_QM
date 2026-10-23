import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "error_vs_arrangement", Path(__file__).resolve().parents[2] / "scripts/experiments/edge_fraction/error_vs_arrangement.py")
eva = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eva)


def test_a_monotone_relation_has_rank_correlation_one():
    x = np.linspace(0, 1, 100)
    r = eva.rank_correlation(np.exp(x), x, n_boot=200)
    assert r["rho"] == pytest.approx(1.0) and r["ci"][0] > 0.99 and r["p"] < 1e-10


def test_for_unrelated_values_the_interval_covers_zero_about_95_percent_of_the_time():
    rng = np.random.default_rng(0)
    runs = [eva.rank_correlation(rng.normal(size=200), rng.normal(size=200), n_boot=200, seed=i) for i in range(60)]
    covered = np.mean([r["ci"][0] < 0 < r["ci"][1] for r in runs])
    assert covered >= 0.85 and np.mean([abs(r["rho"]) for r in runs]) < 0.1


def test_a_descriptor_that_never_varies_has_no_correlation():
    r = eva.rank_correlation(np.arange(10.0), np.zeros(10), n_boot=50)
    assert np.isnan(r["rho"]) and r["significant"] is False


def test_significance_needs_the_corrected_p_value():
    x = np.linspace(0, 1, 60)
    y = x + np.random.default_rng(1).normal(0, 0.6, 60)          # a weak relation
    loose = eva.rank_correlation(y, x, n_boot=200, n_tests=1)
    strict = eva.rank_correlation(y, x, n_boot=200, n_tests=1000)
    assert loose["p"] == strict["p"] and strict["significant"] <= loose["significant"]
