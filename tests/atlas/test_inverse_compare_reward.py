import importlib.util
from pathlib import Path

import numpy as np
import pytest

from shazam.inverse import reward as rw
from shazam.inverse.encoding import STATES
from test_inverse_reward import LEAD_C, LEAD_K, _flip, _state

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(folder, name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / folder / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cr = _load("evaluate", "compare_reward")


def _onehot(*ks, n=156):
    p = np.zeros(n)
    for k, v in ks:
        p[k] += v
    return p


def test_family_probs_sum_out_the_lead():
    p = _onehot((_state(LEAD_C), 0.5), (_state(LEAD_K), 0.5))[None]
    fam = cr.family_probs(p)
    assert fam.shape == (1, 74) and fam[0, rw.FAMILY_OF[_state(LEAD_C)]] == pytest.approx(1.0)


def test_family_top1_on_known_probs():
    a, b = _state(LEAD_C), _flip(_state(LEAD_C), edge="armchair")
    probs = np.stack([_onehot((a, 0.6), (b, 0.4)), _onehot((a, 0.3), (b, 0.7)), _onehot((a, 0.9), (b, 0.1))])
    m = cr.family_metrics(probs, np.array([a, a, a]), decoded=None, physics_rows=None)
    assert m["family_top1_hits"].tolist() == [1, 0, 1]


def test_physics_decode_prefers_the_cheap_hedge():
    a = _state(LEAD_C)
    b, c = 10, 11                                             # any two other states: the cost table is hand-built
    cost = np.ones((156, 156)) - np.eye(156)
    cost[b, c] = cost[c, b] = 0.1
    probs = _onehot((a, 0.45), (b, 0.30), (c, 0.25))[None]
    assert cr.decode_states(probs)[0] == a
    assert cr.decode_states(probs, cost)[0] == b              # expected cost 0.475, against 0.55 for a


def test_ece_of_perfectly_calibrated_bins_is_zero():
    conf = np.repeat([0.25, 0.75], 400)
    hits = np.concatenate([np.arange(400) < 100, np.arange(400) < 300])
    assert cr.family_ece(conf, hits) == pytest.approx(0.0, abs=1e-12)


def test_metrics_are_lead_invariant():
    ka, kb = _state(LEAD_C), _state(LEAD_K)
    probs = np.stack([_onehot((ka, 0.7), (_flip(ka, edge="armchair"), 0.3))] * 2)
    rows = np.zeros((2, 156))
    ma = cr.family_metrics(probs, np.array([ka, ka]), decoded=None, physics_rows=rows)
    mb = cr.family_metrics(probs, np.array([kb, kb]), decoded=None, physics_rows=rows)
    for k in ("family_top1", "family_top3", "family_nll", "family_ece", "physics_cost"):
        assert ma[k] == pytest.approx(mb[k])


def test_width_and_coverage_metrics():
    truth_w = np.array([9, 9, 12])
    samples = np.array([[9] * 10, [14] * 10, [11] * 5 + [13] * 5])
    m = cr.width_metrics(samples, truth_w)
    assert m["width_within1"] == pytest.approx(2 / 3) and m["width_over3"] == pytest.approx(1 / 3)
    assert m["width_in90"] == pytest.approx(2 / 3)


def test_recreation_rate_is_one_when_the_simulator_returns_the_target():
    import torch
    from test_inverse_finetune import _big_model
    model = _big_model(2).eval()
    rng = np.random.default_rng(4)
    target = rng.random((3, 416)).astype(np.float32)

    def ensemble(draw, seeds, row):
        return target[row] + 0.01 * np.random.default_rng(row).normal(size=(len(seeds), 416))

    rate, rows = cr.recreation_rate(model, torch.as_tensor(target), target, n_samples=20, simulate_ensemble=ensemble)
    assert rate == 1.0 and len(rows) == 3 and all(r["recreated"] for r in rows)


def test_ensemble_members_takes_the_first_k_models_and_keeps_the_old_name_for_all():
    names = ["s0", "s1", "s2", "s3"]
    assert cr.ensemble_members(names, sizes=[2, 3, 4], everything=False) == {
        "ensemble2": ["s0", "s1"], "ensemble3": ["s0", "s1", "s2"], "ensemble4": names}
    assert cr.ensemble_members(names, sizes=[], everything=True) == {"ensemble": names}
    assert cr.ensemble_members(["s0"], sizes=[], everything=True) == {}
    with pytest.raises(ValueError):
        cr.ensemble_members(names, sizes=[5], everything=False)
    with pytest.raises(ValueError):
        cr.ensemble_members(names, sizes=[1], everything=False)


def test_model_outputs_give_a_bandtop_scale_model_the_band_top_of_each_row():
    import torch
    from shazam.inverse.encoding import STATES, active_slots
    from shazam.inverse.model import InverseModel
    torch.manual_seed(0)
    masks = torch.as_tensor(np.stack([active_slots(s) for s in STATES]))
    model = InverseModel(len(STATES), channels=417, input="bandtop", extra_conv=1, state_masks=masks, scale="bandtop").eval()
    x = torch.cat([torch.rand(3, 416), torch.tensor([[0.2], [0.5], [0.9]])], 1)
    probs, states, phys = cr.model_outputs(model, x, n_samples=4, seed=0, batch=2)
    assert probs.shape == (3, len(STATES)) and states.shape == (3, 4) and np.isfinite(phys).all()
    torch.manual_seed(0)
    again = cr.model_outputs(model, x, n_samples=4, seed=0, batch=2)[2]
    assert np.allclose(phys, again)
    # row by row the samples are those of sample_batch with that row's band top
    h = model.encode(x[2:3])
    gen = torch.Generator().manual_seed(0)
    p0 = torch.softmax(model.discrete(model.encode(x[:2])), 1)
    torch.multinomial(p0, 4, replacement=True, generator=gen); torch.randn(8, model.n_cont, generator=gen)
    s = torch.multinomial(torch.softmax(model.discrete(h), 1), 4, replacement=True, generator=gen).reshape(-1)
    z = model.sample_batch(h.repeat_interleave(4, 0), s, torch.randn(4, model.n_cont, generator=gen), top=x[2:3, -1].expand(4))
    assert np.allclose(phys[2], rw.physical_torch(s, z).detach().numpy(), atol=1e-5)
