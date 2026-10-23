import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

SPEC = importlib.util.spec_from_file_location(
    "run_edge_fraction", Path(__file__).resolve().parents[2] / "scripts/experiments/edge_fraction/run_edge_fraction.py")
ef = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ef)


def ridge(X_tr, y_tr, X_va, y_va):
    return Ridge(alpha=1.0).fit(X_tr, y_tr)


def test_folds_split_by_seed_and_cover_every_seed_once():
    seeds = np.arange(1000)
    folds = ef.seed_folds(seeds, n_folds=5)
    tested = np.concatenate([te for _, _, te in folds])
    assert sorted(tested.tolist()) == list(range(1000))
    for tr, va, te in folds:
        assert not (set(seeds[tr]) & set(seeds[te])) and not (set(seeds[va]) & set(seeds[te]))
        assert not (set(seeds[tr]) & set(seeds[va]))


def test_a_label_written_in_the_spectrum_is_predicted_with_skill_near_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 20))
    y = X[:, 3] * 0.1 + 0.3
    pred = ef.out_of_fold(X, y, np.arange(1000), ridge, n_folds=5)
    assert ef.skill(pred, y) > 0.99


def test_a_label_unrelated_to_the_spectrum_has_skill_near_zero():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(1000, 20)), rng.uniform(0, 0.5, 1000)
    pred = ef.out_of_fold(X, y, np.arange(1000), ridge, n_folds=5)
    assert abs(ef.skill(pred, y)) < 0.05


def test_skill_compares_the_misfit_with_that_of_predicting_the_mean():
    y = np.array([0.0, 0.2, 0.4])
    assert ef.skill(y, y) == 1.0
    assert ef.skill(np.full(3, y.mean()), y) == pytest.approx(0.0)


def test_the_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(2)
    y = rng.uniform(0, 0.5, 500)
    pred = y + rng.normal(0, 0.05, 500)
    lo, hi = ef.skill_interval(pred, y, n_boot=300, seed=0)
    assert lo < ef.skill(pred, y) < hi
