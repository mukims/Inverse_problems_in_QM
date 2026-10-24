"""Simulator check and likelihood (spec section 8, steps 1, 2 and 4)."""
import numpy as np
import pytest

from shazam.inverse import verify
from shazam.inverse.verify import (FINAL_SEEDS, OPT_SEEDS, VERIFY_SEEDS, CheckResult, log_likelihood,
                                   recreation_check, simulate_inputs)
from tbribbon.parent import named_material
from tbribbon.parent_data import n_impurities
from tbribbon.parent_prior import Draw

# pytest's own process already runs BLAS threads, which makes fork() warn in the parallel test; test-process noise.
pytestmark = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")

N_CELLS = 20                      # short devices keep the tests fast
# The seeds below are small integers on purpose: they are test seeds, not the verification seeds of VERIFY_SEEDS.
SEEDS = tuple(range(1, 52))       # 51 independent disorder realisations: 50 ensemble members + 1 target
HOST = Draw(named_material("graphene-ideal"), "armchair", 6, 0.0125)
N_IMP = n_impurities(HOST, N_CELLS)


@pytest.fixture(scope="module")
def sims():
    return simulate_inputs(HOST, N_IMP, SEEDS, n_cells=N_CELLS)


def test_seed_tuples_are_the_ones_of_the_global_constraints():
    assert VERIFY_SEEDS == tuple(range(10**9, 10**9 + 50))
    assert FINAL_SEEDS == tuple(range(10**9 + 50, 10**9 + 100))
    assert OPT_SEEDS == tuple(range(2 * 10**9, 2 * 10**9 + 20))
    assert not set(VERIFY_SEEDS) & set(FINAL_SEEDS) and not set(VERIFY_SEEDS + FINAL_SEEDS) & set(OPT_SEEDS)


def test_simulate_inputs_shape_dtype_and_seed_order(sims):
    assert sims.shape == (51, 416) and sims.dtype == np.float32
    assert 0.0 <= sims.min() and sims.max() <= 1.0
    assert not np.allclose(sims[0], sims[1])                      # different seeds, different disorder
    again = simulate_inputs(HOST, N_IMP, (SEEDS[3], SEEDS[0]), n_cells=N_CELLS)
    assert np.array_equal(again[0], sims[3]) and np.array_equal(again[1], sims[0])


def test_parallel_simulation_matches_serial_in_seed_order(sims):
    par = simulate_inputs(HOST, N_IMP, SEEDS[:6], n_jobs=2, n_cells=N_CELLS)
    assert np.array_equal(par, sims[:6])


def test_a_member_of_the_ensemble_is_recreated(sims):
    r = recreation_check(sims[50], sims[:50])
    print("member of the ensemble:", r)
    assert isinstance(r, CheckResult)
    assert r.recreated and r.p_value >= 0.05
    assert r.misfit <= r.threshold


def test_a_much_denser_ribbon_is_not_recreated(sims):
    dense = simulate_inputs(HOST, 4 * N_IMP, (SEEDS[50],), n_cells=N_CELLS)[0]
    r = recreation_check(dense, sims[:50])
    print("4x impurities:", r)
    assert not r.recreated and r.misfit > r.threshold


def test_the_matching_host_has_the_higher_likelihood(sims):
    target = sims[50]
    other = simulate_inputs(Draw(named_material("hbn"), "armchair", 6, 0.0125), N_IMP, SEEDS[:50], n_cells=N_CELLS)
    denser = simulate_inputs(HOST, 4 * N_IMP, SEEDS[:50], n_cells=N_CELLS)
    ll = log_likelihood(target, sims[:50])
    print("log-likelihood matching / hBN / 4x denser:", ll, log_likelihood(target, other), log_likelihood(target, denser))
    assert ll > log_likelihood(target, other)
    assert ll > log_likelihood(target, denser)


def test_rank_arithmetic_on_synthetic_vectors():
    rng = np.random.default_rng(0)
    ens = rng.uniform(0.2, 0.8, size=(50, 416))
    r = recreation_check(np.median(ens, axis=0), ens)
    assert r.misfit == 0.0 and r.p_value == 1.0 and r.recreated
    far = recreation_check(np.median(ens, axis=0) + 0.5, ens)
    assert far.p_value == pytest.approx(1 / 51) and not far.recreated


def test_check_matches_a_hand_computation():
    ens = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])
    target = np.array([1.0, 3.0])
    r = recreation_check(target, ens)
    med = np.median(ens, axis=0)                                         # [1.5, 1.5]
    own = [np.mean((ens[j] - np.median(np.delete(ens, j, axis=0), axis=0)) ** 2) for j in range(4)]
    assert r.misfit == pytest.approx(np.mean((target - med) ** 2))      # 1.125
    assert r.threshold == pytest.approx(np.quantile(own, 0.95))
    assert r.p_value == pytest.approx((1 + sum(m >= r.misfit for m in own)) / 5)
    assert r.recreated == (r.misfit <= r.threshold)


def test_log_likelihood_formula_and_spread_floor():
    ens = np.array([[0.1, 0.5], [0.3, 0.5], [0.2, 0.5]])
    x = np.array([0.2, 0.6])
    mu, sd = np.median(ens, axis=0), ens.std(axis=0)
    sig = np.sqrt(sd ** 2 + verify.S0 ** 2)
    want = float(np.sum(-0.5 * ((x - mu) / sig) ** 2 - np.log(sig)))
    assert log_likelihood(x, ens) == pytest.approx(want)
    assert np.isfinite(log_likelihood(x, ens))                           # channel 2 has zero spread: the floor keeps it finite
