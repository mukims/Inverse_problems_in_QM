import time
import warnings

import numpy as np

from shazam.inverse import posterior as post
from shazam.inverse.calibration import expected_calibration_error


def test_a_better_fit_gains_probability():
    p = np.array([0.3, 0.3, 0.2])
    log_L = np.array([-10.0, -50.0, -50.0])
    u = post.updated_probabilities(p, log_L, p0=0.2, kappa=1.0, log_L0=-50.0)
    assert u[0] > p[0] and u[1] < p[1] and u[2] < p[2]
    assert u[0] > 0.99


def test_when_every_candidate_fits_badly_all_fall_towards_zero():
    p = np.array([0.4, 0.3, 0.2])
    log_L0 = -100.0
    u = post.updated_probabilities(p, np.full(3, log_L0 - 200.0), p0=0.1, kappa=1.0, log_L0=log_L0)
    assert np.all(u < 0.01)


def test_updated_never_sum_above_one():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 6))
        p = rng.dirichlet(np.ones(n + 1))[:n]
        u = post.updated_probabilities(p, rng.normal(0, 30, n), 1 - p.sum(), float(rng.choice([1.0, 10.0, 300.0])),
                                       float(rng.normal(0, 30)))
        assert np.all(u >= 0) and u.sum() <= 1 + 1e-12


def test_neutral_evidence_keeps_the_model_probabilities():
    p = np.array([0.5, 0.3, 0.1])
    u = post.updated_probabilities(p, np.full(3, -37.0), p0=0.0, kappa=1.0, log_L0=-80.0)
    np.testing.assert_allclose(u, p / p.sum())


def test_zero_masses_and_empty_input_are_clean():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert post.updated_probabilities(np.array([]), np.array([]), 0.3, 2.0, -5.0).shape == (0,)
        u = post.updated_probabilities(np.array([0.0, 0.5]), np.array([-1.0, -1.0]), 0.0, 1.0, -5.0)
        np.testing.assert_allclose(u, [0.0, 1.0])
        np.testing.assert_allclose(post.updated_probabilities(np.array([0.0]), np.array([-1.0]), 0.5, 1.0, -1.0), [0.0])
        np.testing.assert_allclose(post.updated_probabilities(np.array([0.0]), np.array([-1.0]), 0.0, 1.0, -1.0), [0.0])


def _synthetic(n_records, seed=0):
    """Overconfident model probabilities; the likelihood is higher for the hits, so it carries signal."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_records):
        n = int(rng.integers(1, 6))
        p = rng.dirichlet(np.ones(n + 1) * 2)[:n] * 0.5 + 0.4 / n
        p0 = 1 - p.sum()
        hits = rng.random(n) < 0.25
        log_L = np.where(hits, rng.normal(-40, 8, n), rng.normal(-90, 25, n))
        out.append(dict(p=p, log_L=log_L, p0=p0, hits=hits))
    return out


def test_fit_update_does_no_worse_than_kappa_one():
    recs = _synthetic(200)
    fit = post.fit_update(recs)
    grid = np.quantile(np.concatenate([r["log_L"] for r in recs]), np.linspace(0, 1, 41))
    hits = np.concatenate([r["hits"] for r in recs])

    def ece(kappa, log_L0):
        u = np.concatenate([post.updated_probabilities(r["p"], r["log_L"], r["p0"], kappa, log_L0) for r in recs])
        return expected_calibration_error(u, hits)

    best_k1 = min(ece(1.0, g) for g in grid)
    assert fit["ece"] <= best_k1 + 1e-12
    assert abs(ece(fit["kappa"], fit["log_L0"]) - fit["ece"]) < 1e-12
    assert fit["ece"] < fit["ece_model"]
    assert all(type(fit[k]) is float for k in ("kappa", "log_L0", "ece", "ece_model"))


def test_fit_update_is_deterministic_and_ties_go_to_the_smallest():
    # no evidence at all: every (kappa, log_L0) gives the same updated probabilities when p0 = 0
    recs = [dict(p=np.array([0.35, 0.65]), log_L=np.array([-3.0, -3.0]), p0=0.0, hits=np.array([True, False]))]
    fit = post.fit_update(recs, kappas=np.array([5.0, 1.0, 20.0]), log_L0_grid=np.array([2.0, -7.0]))
    assert (fit["kappa"], fit["log_L0"]) == (1.0, -7.0)


def test_fit_time_on_500_records_is_reported():
    recs = _synthetic(500, seed=1)
    t0 = time.time()
    fit = post.fit_update(recs)
    dt = time.time() - t0
    print(f"fit_update on 500 records: {dt:.2f} s", fit)
    assert dt < 60
