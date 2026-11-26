import dataclasses

import numpy as np
import pytest

from shazam.inverse import sim_reward as sr
from tbribbon.parent_prior import sample_prior

pytestmark = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")


def test_reward_of_the_target_itself_is_zero():
    x = np.random.default_rng(0).random(416)
    assert sr.reward(x, x, m0=1e-3) == 0.0
    assert sr.reward(x + 0.1, x, m0=1e-3) < 0


def test_smoothing_cuts_independent_noise_by_sqrt_k():
    noise = np.random.default_rng(1).normal(size=(200, 416))
    ratio = sr.smooth(noise, 9)[:, 10:-10].std() / noise.std()
    assert ratio == pytest.approx(1 / 3, rel=0.1)
    assert sr.smooth(noise, 1) is not noise and np.allclose(sr.smooth(noise, 1), noise)
    assert sr.smooth(np.ones(416), 5).shape == (416,) and np.allclose(sr.smooth(np.ones(416), 5), 1)


def test_failed_or_out_of_box_candidates_are_none(monkeypatch):
    rng = np.random.default_rng(2)
    draws = [sample_prior(rng)._replace(width=4) for _ in range(3)]
    bad = draws[1]

    def fake(draw, n_imp, seeds, n_jobs=1, n_cells=None):
        if draw == bad:
            raise RuntimeError("simulation died")
        return np.full((len(seeds), 416), 0.5, np.float32)

    monkeypatch.setattr(sr, "simulate_inputs", fake)
    for n_jobs in (1, 2):
        out = sr.simulate_candidates(draws, [7, 8, 9], n_jobs=n_jobs)
        assert out[1] is None and out[0].shape == (416,) and out[2].shape == (416,)
    huge = draws[0]._replace(params=dataclasses.replace(draws[0].params, s=5.0))       # s above the box (4 eV)
    assert sr.in_box(draws[0]) and not sr.in_box(huge)
    assert sr.simulate_candidates([huge], [1], n_jobs=1) == [None]


def test_m0_model_fits_and_round_trips_json():
    w = np.array([5, 9, 15, 21] * 5, float)
    c = np.repeat([0.003, 0.01, 0.02, 0.03, 0.05], 4)
    m0 = np.exp(-6 + 0.5 * np.log(w) + 0.8 * np.log(c))
    model = sr.M0Model.fit(w, c, m0)
    assert model(9, 0.01) == pytest.approx(np.exp(-6 + 0.5 * np.log(9) + 0.8 * np.log(0.01)), rel=1e-6)
    again = sr.M0Model.from_json(model.to_json())
    assert again(12, 0.02) == pytest.approx(model(12, 0.02))


def test_host_m0_is_the_median_leave_one_out_misfit():
    em = __import__("importlib.util").util
    spec = em.spec_from_file_location("estimate_m0", __import__("pathlib").Path(__file__).resolve().parents[2]
                                      / "scripts" / "build" / "estimate_m0.py")
    mod = em.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.host_m0(np.ones((5, 416)), k=1) == 0.0
    noisy = np.random.default_rng(3).normal(size=(10, 416))
    assert mod.host_m0(noisy, k=9) < mod.host_m0(noisy, k=1)
