import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats

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


def test_sbc_rank_needs_an_explicit_rng():
    with pytest.raises(TypeError):
        cal.sbc_rank(1.0, np.array([1.0, 1.0]))


class _TwoClusterModel:
    """A perfectly calibrated 'model' whose posterior holds two well-separated clusters (60 / 40 %) in the same state, hence
    the same (edge, pattern): it samples the mixture, and conditioned on the state it is the same mixture."""

    def __init__(self, state, centres, weight, rng, sigma=0.1):
        self.state, self.centres, self.weight, self.rng, self.sigma = state, centres, weight, rng, sigma
        self.mask = active_slots(state)

    def sample(self, x, n, steps=64, state=None):
        pick = self.rng.random(n) >= self.weight
        z = np.where(pick[:, None], self.centres[1], self.centres[0]) + self.sigma * self.rng.normal(size=(n, 14))
        return torch.full((n,), self.state, dtype=torch.long), torch.as_tensor(z * self.mask)


def test_coverage_is_conditioned_on_the_nearest_right_key_candidate(monkeypatch):
    from shazam.inverse import candidates as cmod
    from shazam.inverse.encoding import decode
    monkeypatch.setattr(cmod, "_n_imp", lambda draw, atoms: 1)           # the impurity counts play no part here
    rng = np.random.default_rng(0)
    k, z0, mask = encode(sample_prior(np.random.default_rng(3))._replace(width=9))
    shift = np.where(mask, 2.0, 0.0)                                     # a second cluster 2 logits away on every active slot
    records = []
    for _ in range(80):
        base = z0 + 0.3 * rng.normal(size=14) * mask
        model = _TwoClusterModel(k, (base, base + shift), 0.6, rng)
        truth = decode(k, model.sample(None, 1)[1][0].numpy())              # the truth comes from the same posterior
        records.append(cal.evaluate_draw(model, np.zeros(416, np.float32), truth, 400, 1, 0, rng))
    s = cal.summarize(records, 400)
    old, new = s["coverage"]["key_only_diagnostic"]["pooled"], s["coverage"]["pooled"]
    assert old["coverage"] < 0.8                                         # the truth often belongs to the other cluster
    assert 0.85 <= new["coverage"] <= 0.95 and new["n"] < old["n"]       # top and nearest: the top cluster's own 90% intervals
    assert s["coverage"]["n_draws"] < s["coverage"]["key_only_diagnostic"]["n_draws"]


def _records(n, coverage_hits, rng, n_samples=100):
    """Fabricated per-draw records: uniform SBC ranks and a coverage of `coverage_hits` / n for c and one more parameter."""
    return [dict(uncond={k: int(rng.integers(0, n_samples + 1)) for k in ("log_s", "log_c", "w")},
                 cond={"eps_C": int(rng.integers(0, n_samples + 1))},
                 covered={"c": i < coverage_hits, "s": True, "w": False}, covered_key_only=None,
                 recall=True, probs=[0.5], hits=[bool(i % 2)]) for i in range(n)]


def test_noise_aware_verdicts_are_reported_without_changing_the_gates():
    rng = np.random.default_rng(0)
    s = cal.summarize(_records(500, 450, rng), 100)
    k = len(s["sbc_unconditional"]) + len(s["sbc_conditional"])
    assert k == 4 and s["noise_aware"]["ks_gates"] == 4
    for g in (*s["sbc_unconditional"].values(), *s["sbc_conditional"].values()):
        assert abs(g["ks_p_bonferroni"] - min(1.0, k * g["ks_p"])) < 1e-12
        assert g["pass"] == (g["ks_p"] > 0.01)                           # the spec flag is untouched
    c = s["coverage"]["per_parameter"]["c"]
    assert c["coverage"] == 0.9 and c["pass"] and abs(c["binomial_p"] - stats.binomtest(450, 500, 0.9).pvalue) < 1e-12
    assert s["coverage"]["per_parameter"]["s"]["coverage"] == 1.0 and not s["coverage"]["per_parameter"]["s"]["pass"]
    assert s["coverage"]["per_parameter"]["w"]["gated"] is False and s["coverage"]["per_parameter"]["w"]["pass"] is None
    assert "ruling R19" in s["coverage"]["note_w"]
    assert s["noise_aware"]["coverage_gates"] == 2 and abs(c["binomial_p_bonferroni"] - min(1.0, 2 * c["binomial_p"])) < 1e-12
    assert s["pass"] is False and s["coverage_consistent_bonferroni"] is False        # coverage of s is 1.0 over 500 draws
    # a borderline coverage: 0.93 passes the spec flag; Bonferroni says whether it is consistent with 0.90
    t = cal.summarize(_records(100, 93, rng), 100)["coverage"]["per_parameter"]["c"]
    assert t["pass"] and t["binomial_p_bonferroni"] > 0.01


def test_cli_resumes_from_the_jsonl_and_gives_the_same_result(tmp_path, monkeypatch):
    ev = _load("evaluate", "eval_inverse_calibration")
    torch.manual_seed(0)
    InverseModel(len(STATES), state_masks=np.stack([active_slots(s) for s in STATES])).save(tmp_path / "model.pt")
    np.savez(tmp_path / "shard_00000.npz", **_arrays(4))
    args = ["--model", str(tmp_path / "model.pt"), "--data", str(tmp_path), "--n-test", "4", "--n-samples", "60",
            "--steps", "4", "--threads", "2", "--out", str(tmp_path / "calibration.json")]
    ev.main(args)
    first = json.loads((tmp_path / "calibration.json").read_text())
    log = tmp_path / "calibration.jsonl"
    lines = log.read_text().splitlines()
    assert len(lines) == 4 and sorted(json.loads(l)["index"] for l in lines) == [19, 39, 59, 79]
    log.write_text("\n".join(lines[:2]) + "\n" + lines[2][:20])             # two done, a third cut short by a crash
    calls = []
    orig = cal.evaluate_draw
    monkeypatch.setattr(cal, "evaluate_draw", lambda *a, **k: calls.append(1) or orig(*a, **k))
    ev.main(args)
    assert len(calls) == 2                                               # only the missing draws are evaluated
    second = json.loads((tmp_path / "calibration.json").read_text())
    for key in ("sbc_unconditional", "sbc_conditional", "coverage", "top5_recall", "ece", "gates", "pass"):
        assert second[key] == first[key]                                 # (index, seed) fix each draw's randomness
    good = [json.loads(l) for l in log.read_text().splitlines() if l.endswith("}")]            # the cut line stays, ignored
    assert second["resumed_draws"] == 2 and sorted(r["index"] for r in good) == [19, 39, 59, 79]
    calls.clear()
    ev.main(args)
    assert not calls
    with pytest.raises(SystemExit):
        ev.main(args[:args.index("--n-samples")] + ["--n-samples", "50"] + args[args.index("--n-samples") + 2:])
