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


# ---------------------------------------------------------------- partial correlation, controlling for other descriptors

def test_partial_correlation_removes_a_link_that_only_runs_through_a_control():
    rng = np.random.default_rng(5)
    z = rng.normal(size=600)                                   # e.g. edge fraction
    err, desc = z + rng.normal(0, 0.3, 600), z + rng.normal(0, 0.3, 600)
    assert eva.rank_correlation(err, desc, n_boot=100)["rho"] > 0.8
    assert abs(eva.partial_rank_correlation(err, desc, [z], n_boot=100)["rho"]) < 0.1


def test_partial_correlation_keeps_a_link_the_controls_do_not_explain():
    rng = np.random.default_rng(6)
    z, desc = rng.normal(size=600), rng.normal(size=600)
    err = desc + 0.5 * z + rng.normal(0, 0.3, 600)
    r = eva.partial_rank_correlation(err, desc, [z], n_boot=100)
    assert r["rho"] > 0.8 and r["ci"][0] > 0.7 and r["p"] < 1e-10


def test_a_control_that_never_varies_is_ignored():
    rng = np.random.default_rng(7)
    z, desc = rng.normal(size=300), rng.normal(size=300)
    err = desc + z
    with_constant = eva.partial_rank_correlation(err, desc, [z, np.zeros(300)], n_boot=50)
    without = eva.partial_rank_correlation(err, desc, [z], n_boot=50)
    assert with_constant["rho"] == pytest.approx(without["rho"]) and with_constant["p"] == pytest.approx(without["p"])


def test_with_no_controls_the_partial_correlation_is_the_plain_one():
    rng = np.random.default_rng(8)
    x = rng.normal(size=200); y = x + rng.normal(size=200)
    assert eva.partial_rank_correlation(y, x, [], n_boot=50)["rho"] == pytest.approx(eva.rank_correlation(y, x, n_boot=50)["rho"])
