"""Trust-region refinement (spec section 8, step 3)."""
import numpy as np
import pytest

from shazam.inverse import refine as refine_mod
from shazam.inverse.encoding import encode
from shazam.inverse.refine import RefineResult, refine, trust_region_minimise
from shazam.inverse.verify import OPT_SEEDS
from tbribbon.parent import named_material
from tbribbon.parent_prior import Draw

STAR = np.array([0.6, -0.5, 0.6])                     # minimum of the analytic test function
X0 = STAR + np.array([1.0, 0.0, 0.0])                 # start at distance 1


def quad(x):
    return float(np.sum((np.asarray(x) - STAR) ** 2))


def test_finds_the_minimum_of_a_quadratic_in_ten_evaluations():
    calls = []

    def f(x):
        calls.append(np.array(x))
        return quad(x)

    x, best, history, stopped = trust_region_minimise(f, X0, 10, None, np.random.default_rng(0))
    assert len(calls) == len(history) == 10 and not stopped
    assert history[0] == quad(X0) == 1.0
    assert best < 0.25 * quad(X0)
    assert best == min(history) and np.isclose(quad(x), best)


def test_stops_early_below_the_threshold():
    calls = []

    def f(x):
        calls.append(0)
        return quad(x)

    x, best, history, stopped = trust_region_minimise(f, X0, 10, 0.5, np.random.default_rng(0))
    assert stopped and best < 0.5 and len(history) == len(calls) < 10


def test_exhausted_budget_is_reported_not_raised():
    x, best, history, stopped = trust_region_minimise(quad, X0, 10, -1.0, np.random.default_rng(0))
    assert not stopped and len(history) == 10
    assert np.isclose(quad(x), best) and best == min(history)


def test_refine_keeps_the_discrete_state(monkeypatch):
    host = Draw(named_material("hbn"), "zigzag", 6, 0.0125)
    state, z, mask = encode(host)
    target = np.full(416, 0.3, np.float32)
    seen = []

    def spectra(draw, seeds):
        rng = np.random.default_rng([int(s) for s in seeds])          # stands in for the disorder seeds
        return (0.3 + 0.05 * np.tanh(draw.params.s) + 0.01 * rng.standard_normal((len(seeds), 416))).astype(np.float32)

    def fake_simulate(draw, n_imp, seeds, n_jobs=1, n_cells=None):
        seen.append((draw, n_imp, tuple(seeds)))
        return spectra(draw, seeds)

    monkeypatch.setattr(refine_mod, "simulate_inputs", fake_simulate)
    res = refine(host, 3, target, threshold=-1.0)
    assert isinstance(res, RefineResult) and not res.stopped_early
    assert len(seen) == 10 and res.n_simulations == 200
    assert all(n == 3 and s == OPT_SEEDS for _, n, s in seen)
    for d, _, _ in seen:
        assert (d.params.pattern, d.edge, d.params.lead) == (host.params.pattern, host.edge, host.params.lead)
        assert d.width == host.width and d.params.t_AB == 1.0
    out = res.draw
    assert (out.params.pattern, out.edge, out.params.lead) == (host.params.pattern, host.edge, host.params.lead)
    assert out.width == host.width and res.misfit == pytest.approx(min(
        float(np.mean((np.median(spectra(d, OPT_SEEDS), axis=0) - target) ** 2)) for d, _, _ in seen), rel=1e-5)


def test_a_failed_evaluation_is_penalised_not_propagated():
    f = lambda x: float("nan") if x[0] > 0.2 else float(np.sum(x ** 2))       # simulations fail in half the box
    best_x, best_f, history, _ = trust_region_minimise(f, np.array([0.0, 1.0, 0.0]), 10, None, np.random.default_rng(1))
    assert np.all(np.isfinite(history)) and best_f == min(history) and best_x[0] <= 0.2


def test_refine_uses_posterior_samples_and_recovers_shifted_host(monkeypatch):
    # Tests that refine accepts z_samples and scales the search accordingly
    host = Draw(named_material("hbn"), "zigzag", 6, 0.0125)
    state, z0, mask = encode(host)
    target = np.full(416, 0.3, np.float32)

    # Synthetic cluster of z_samples around shifted center (s increased by 0.3)
    rng = np.random.default_rng(42)
    z_cluster = np.tile(z0, (50, 1))
    z_cluster[:, 0] += rng.normal(0.3, 0.08, size=50)

    seen = []

    def spectra(draw, seeds):
        rng = np.random.default_rng([int(s) for s in seeds])
        # Minimum misfit when s is near 0.3 above host.params.s
        diff = (draw.params.s - (host.params.s + 0.3)) ** 2
        return (0.3 + 0.01 * diff + 0.001 * rng.standard_normal((len(seeds), 416))).astype(np.float32)

    def fake_simulate(draw, n_imp, seeds, n_jobs=1, n_cells=None):
        seen.append(draw)
        return spectra(draw, seeds)

    monkeypatch.setattr(refine_mod, "simulate_inputs", fake_simulate)
    res = refine(host, 3, target, threshold=-1.0, budget=80, rng=np.random.default_rng(0),
                 z_samples=z_cluster)
    assert isinstance(res, RefineResult) and not res.stopped_early
    assert len(seen) == 4  # 80 budget // 20 seeds = 4 evals
    # The refined draw moves towards the shifted target s
    assert abs(res.draw.params.s - (host.params.s + 0.3)) < abs(host.params.s - (host.params.s + 0.3))

