import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from shazam.inverse import calibration as cal, data
from shazam.inverse.candidates import PRIOR_SCALE, candidates, full_vector
from shazam.inverse.encoding import STATES, active_slots, encode
from shazam.inverse.model import InverseModel
from tbribbon.parent import TYPES
from tbribbon.parent_prior import EDGES, sample_prior
from test_inverse_candidates import HOST_A, HOST_B, samples

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(folder, name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / folder / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def two_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def test_calibrated_samples_pass_and_overconfident_fail():
    rng = np.random.default_rng(0)
    n_trials, n_samples = 1000, 1000
    truth = rng.normal(size=n_trials)
    samples_ = rng.normal(size=(n_trials, n_samples))                  # the same Normal the truths come from
    ranks = np.array([cal.sbc_rank(t, s, rng) for t, s in zip(truth, samples_)])
    assert cal.ks_uniform_p(ranks, n_samples) > 0.01
    lo, hi = np.percentile(samples_, [5, 95], axis=1)
    assert 0.87 <= cal.coverage(lo, hi, truth) <= 0.93
    lo, hi = np.percentile(0.5 * samples_, [5, 95], axis=1)            # overconfident: spread halved
    assert cal.coverage(lo, hi, truth) < 0.7
    ranks = np.array([cal.sbc_rank(t, 0.5 * s, rng) for t, s in zip(truth, samples_)])
    assert cal.ks_uniform_p(ranks, n_samples) < 0.01


def test_sbc_rank_breaks_ties_uniformly():
    rng = np.random.default_rng(1)
    s = np.array([0.0, 1.0, 1.0, 1.0, 2.0])                            # 1 below, 3 equal: rank in 1..4
    r = [cal.sbc_rank(1.0, s, rng) for _ in range(400)]
    assert set(r) == {1, 2, 3, 4} and cal.sbc_rank(-1.0, s, rng) == 0 and cal.sbc_rank(3.0, s, rng) == 5
    # an integer-valued truth against integer samples is still uniform once ties are broken at random
    n = 1000
    ranks = np.array([cal.sbc_rank(t, rng.integers(0, 6, n).astype(float), rng) for t in rng.integers(0, 6, 800)])
    assert cal.ks_uniform_p(ranks, n, rng) > 0.01


def test_ece_is_small_when_probabilities_are_honest():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20000)
    hits = rng.uniform(size=20000) < p
    assert cal.expected_calibration_error(p, hits) < 0.03
    assert cal.expected_calibration_error(np.full(20000, 0.9), rng.uniform(size=20000) < 0.5) > 0.35
    assert cal.expected_calibration_error(np.array([1.0]), np.array([True])) == 0.0        # p = 1 falls in the last bin
    assert np.isnan(cal.expected_calibration_error(np.array([]), np.array([])))


def test_captures_requires_nearest_and_inside_the_99_percent_ellipsoid():
    states, z = samples([(HOST_A, 600), (HOST_B, 400)], seed=2)
    (ca, cb), _ = candidates(states, z)
    assert ca.draw.params.s > 2.0 and cb.draw.params.s < 2.0                     # ca is host A's cluster
    key = (HOST_A.edge, HOST_A.params.pattern)
    truth = full_vector(HOST_A)
    all_c = [ca, cb]
    assert cal.captures(ca, truth, all_c, key) and not cal.captures(cb, truth, all_c, key)   # cb is not the nearest
    assert cal.captures(ca, HOST_A, all_c) and cal.captures(cb, HOST_B, all_c)                # a Draw carries its own key
    assert not cal.captures(ca, truth, all_c, ("zigzag", HOST_A.params.pattern))              # wrong edge
    assert not cal.captures(ca, truth, all_c, (HOST_A.edge, (True, True, False)))             # wrong pattern
    # nearest to cb's cluster in prior units, yet outside cb's own 99% ellipsoid
    far = full_vector(HOST_B)
    far[0] += 0.3                                                                  # log s shifted by 0.3: nearer to B than A
    assert np.linalg.norm((far - ca.mean) / PRIOR_SCALE) > np.linalg.norm((far - cb.mean) / PRIOR_SCALE)
    assert cal.mahalanobis(far, cb.mean, cb.cov) > cb.mahal_99
    assert not cal.captures(cb, far, all_c, key) and not cal.captures(ca, far, all_c, key)
    near = full_vector(HOST_B)
    near[0] += 0.005
    assert cal.mahalanobis(near, cb.mean, cb.cov) <= cb.mahal_99 and cal.captures(cb, near, all_c, key)


def test_mahalanobis_matches_the_definition():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 5))
    cov = a @ a.T + np.eye(5)
    x, m = rng.normal(size=5), rng.normal(size=5)
    assert abs(cal.mahalanobis(x, m, cov) - np.sqrt((x - m) @ np.linalg.inv(cov) @ (x - m))) < 1e-9
    xs = rng.normal(size=(7, 5))                                       # rows are accepted too
    assert np.allclose(cal.mahalanobis(xs, m, cov), [cal.mahalanobis(r, m, cov) for r in xs])


def _arrays(n, seed=0):
    """Test-split rows (index % 20 == 19) from prior draws, with random spectra."""
    draws = [sample_prior(np.random.default_rng(seed + i)) for i in range(n)]
    rng = np.random.default_rng(seed)
    return dict(index=19 + 20 * np.arange(n), x=rng.normal(size=(n, 416)).astype(np.float32),
                on=np.array([[d.params.on_C, d.params.on_H, d.params.on_K] for d in draws], np.int8),
                lead=np.array([TYPES.index(d.params.lead) for d in draws], np.int8),
                params=np.array([[getattr(d.params, c) for c in data.PARAM_COLUMNS] for d in draws]),
                edge=np.array([EDGES.index(d.edge) for d in draws], np.int8),
                width=np.array([d.width for d in draws], np.int16), density=np.array([d.density for d in draws]))


def test_cli_writes_a_well_formed_calibration_json(tmp_path, monkeypatch):
    ev = _load("evaluate", "eval_inverse_calibration")
    torch.manual_seed(0)
    model = InverseModel(len(STATES), state_masks=np.stack([active_slots(s) for s in STATES]))
    model.save(tmp_path / "model.pt")
    np.savez(tmp_path / "shard_00000.npz", **_arrays(4))
    out = tmp_path / "calibration.json"
    ev.main(["--model", str(tmp_path / "model.pt"), "--data", str(tmp_path), "--n-test", "4", "--n-samples", "60",
             "--steps", "4", "--threads", "2", "--out", str(out)])
    r = json.loads(out.read_text())
    assert r["n_test"] == 4 and r["n_samples"] == 60 and isinstance(r["pass"], bool) and r["seconds_per_draw"] > 0
    assert set(r["sbc_unconditional"]) == {"log_s", "log_c", "w"}
    for g in (*r["sbc_unconditional"].values(), *r["sbc_conditional"].values()):
        assert set(g) >= {"ks_p", "n", "pass"}
    assert set(r["coverage"]) >= {"per_parameter", "pooled"} and "value" in r["top5_recall"] and "value" in r["ece"]
    assert isinstance(r["pass"], bool) and r["pass"] == all(r["gates"].values())
    assert {"sbc_unconditional", "coverage", "top5_recall", "ece"} <= set(r["gates"])
