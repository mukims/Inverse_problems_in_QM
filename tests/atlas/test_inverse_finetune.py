import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from shazam.inverse import reward as rw
from shazam.inverse.encoding import encode
from tbribbon.parent_prior import sample_prior

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(folder, name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / folder / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")


@pytest.fixture(autouse=True)
def two_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def test_reward_tables_on_synthetic_draws():
    rt = _load("build", "reward_tables")
    rng = np.random.default_rng(0)
    draws = [sample_prior(rng) for _ in range(40)]
    weights = rt.band_weights_of(draws, n_jobs=2)
    np.testing.assert_allclose(weights.sum(1), 1, atol=1e-9)
    scales = rw.fit_physics_scales(draws, weights)
    t = rt.build_tables(draws, weights, scales)
    states = np.array([encode(d)[0] for d in draws])
    for arm in ("label", "physics"):
        c = t[arm]
        assert c.shape == (40, 156) and c.min() >= 0 and c.max() <= 1
        assert np.all(c[np.arange(40), states] == 0)


# ---------------------------------------------------------------- finetune_inverse (Task 5)

from shazam.inverse import data  # noqa: E402
from shazam.inverse.model import InverseModel  # noqa: E402
from shazam.inverse.runio import sha256_of  # noqa: E402
from tbribbon import parent_data  # noqa: E402

gp = _load("generate", "generate_parent")
ti = _load("build", "train_inverse")
fi = _load("build", "finetune_inverse")


@pytest.fixture(scope="module")
def shard(tmp_path_factory):
    """200 draws (180 train, 10 val, 10 test) on small hosts, as in test_inverse_training."""
    root = tmp_path_factory.mktemp("finetune_smoke")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(parent_data, "device_cells_for", lambda params, edge: 10)
        orig = gp.sample_prior
        mp.setattr(gp, "sample_prior", lambda rng: (lambda d: d._replace(width=4 + d.width % 3))(orig(rng)))
        gp.write_shard(root, 0, range(200), n_jobs=4)
    return root


@pytest.fixture(scope="module")
def reference(shard, tmp_path_factory):
    """A one-epoch band-top reference model, saved like a real one."""
    out = tmp_path_factory.mktemp("ref")
    ti.main(["--data", str(shard), "--out", str(out), "--epochs", "1", "--input", "bandtop", "--extra-conv", "1",
             "--threads", "2"])
    return out / "model.pt"


def _tables(arrays):
    rt = _load("build", "reward_tables")
    draws = data.draws(arrays, slice(None))
    weights = rt.band_weights_of(draws, n_jobs=1)
    return draws, weights


def test_classifier_loss_matches_expected_cost_when_beta_and_lambda_are_zero():
    torch.manual_seed(0)
    logits, cost = torch.randn(5, 156), torch.rand(5, 156)
    y = torch.randint(0, 156, (5,))
    loss, _ = fi.classifier_loss(logits, logits + torch.randn(5, 156), cost, y, beta=0.0, lam=0.0)
    want = (torch.softmax(logits, 1) * cost).sum(1).mean()
    assert loss.item() == pytest.approx(want.item(), abs=1e-6)


def test_kl_term_is_zero_at_the_reference():
    torch.manual_seed(1)
    logits, cost = torch.randn(4, 156), torch.rand(4, 156)
    y = torch.randint(0, 156, (4,))
    with_kl, parts = fi.classifier_loss(logits, logits.clone(), cost, y, beta=1.0, lam=0.0)
    assert parts["kl"] == pytest.approx(0.0, abs=1e-6)
    assert with_kl.item() == pytest.approx(parts["expected_cost"], abs=1e-6)


def test_lead_swapped_truth_gives_the_same_loss():
    from test_inverse_reward import LEAD_C, LEAD_K
    from tbribbon.parent_prior import Draw
    a, b = Draw(LEAD_C, "zigzag", 9, 0.01), Draw(LEAD_K, "zigzag", 9, 0.01)
    w = np.array([[0.5, 0.0, 0.5]] * 2)
    from test_inverse_reward import SCALES
    rows = torch.as_tensor(rw.physics_family_cost_table([a, b], w, SCALES))
    assert torch.equal(rows[0], rows[1])
    logits = torch.randn(1, 156).expand(2, -1)
    la, _ = fi.classifier_loss(logits[:1], logits[:1], rows[:1], torch.tensor([encode(a)[0]]), beta=0.1, lam=0.0)
    lb, _ = fi.classifier_loss(logits[:1], logits[:1], rows[1:], torch.tensor([encode(b)[0]]), beta=0.1, lam=0.0)
    assert la.item() == pytest.approx(lb.item())


def test_reward_none_is_the_original_loss(shard, reference):
    train = data.load_split(shard, "train", fi.NEEDED)
    cfg = fi.FinetuneConfig(reward="none", ref=str(reference), batch=64, lr=1e-4)
    model = InverseModel.load(reference)
    tensors = fi.prepare_rows(train, None, model.cfg["input"])
    idx = torch.arange(64)
    torch.manual_seed(3)
    loss, _ = fi.step_loss(model, None, tensors, idx, cfg)
    other = InverseModel.load(reference)          # loaded before seeding: building a model draws random weights
    torch.manual_seed(3)
    want, _ = other.loss(tensors["x"][idx], tensors["state"][idx], tensors["z"][idx], tensors["mask"][idx])
    assert loss.item() == pytest.approx(want.item(), rel=1e-6)


@pytest.mark.parametrize("arm", ["label", "physics"])
def test_finetune_smoke_writes_model_and_history(shard, reference, tmp_path, arm):
    ref_sha, ref_mtime = sha256_of(reference), reference.stat().st_mtime
    train, val = (data.load_split(shard, s, fi.NEEDED) for s in ("train", "val"))
    tables = {}
    for name, arrays in (("train", train), ("val", val)):
        draws, weights = _tables(arrays)
        scales = rw.fit_physics_scales(draws, weights)
        states = np.array([encode(d)[0] for d in draws])
        cost = (rw.label_family_cost_table(states) if arm == "label"
                else rw.physics_family_cost_table(draws, weights, scales))
        tables[name] = dict(cost=cost, weights=weights)
    tables["scales"] = scales
    cfg = fi.FinetuneConfig(reward=arm, ref=str(reference), epochs=1, batch=64, flow_rows=8, val_subset=10, steps=8)
    model, history = fi.finetune(train, val, tables, cfg)
    fi.write_run(tmp_path, model, history)
    assert InverseModel.load(tmp_path / "model.pt").cfg["input"] == "bandtop"
    h = __import__("json").loads((tmp_path / "history.json").read_text())
    assert h["ref_sha256"] == ref_sha and h["cfg"]["reward"] == arm and len(h["val"]) == 1
    assert {"family_top1", "family_nll"} <= set(h["val"][0])
    assert sha256_of(reference) == ref_sha and reference.stat().st_mtime == ref_mtime


# ---------------------------------------------------------------- finetune_simulator (Task 9)

fs = _load("build", "finetune_simulator")


def test_group_advantages_normalise_and_skip_small_groups():
    nan = np.nan
    a = fs.group_advantages(np.array([[1, 2, 3, nan], [5, nan, nan, nan]]))
    np.testing.assert_allclose(a[0, :3], [-1.2247449, 0, 1.2247449], atol=1e-6)
    assert np.isnan(a[0, 3]) and np.all(np.isnan(a[1]))
    assert np.all(fs.group_advantages(np.array([[2.0, 2.0, 2.0]])) == 0)


def _big_model(seed=0):
    from shazam.inverse.encoding import STATES, active_slots
    torch.manual_seed(seed)
    masks = torch.as_tensor(np.stack([active_slots(s) for s in STATES]))
    return InverseModel(len(STATES), channels=416, state_masks=masks)


def test_grpo_step_raises_logprob_of_the_better_candidate():
    model, ref = _big_model(), _big_model()
    ref.load_state_dict(model.state_dict())
    ref.requires_grad_(False)
    x = torch.randn(1, 416)
    states = torch.tensor([[3, 40]])
    z = torch.randn(1, 2, 14) * model.state_masks[states].float()
    cfg = fs.SimConfig(start="", group=2, targets_per_step=1, beta=0.0, anchor=0.0, lr=1e-3)
    opt = torch.optim.AdamW(model.parameters(), cfg.lr)
    before = torch.log_softmax(model.discrete(model.encode(x)), 1)[0, 3].item()
    fs.grpo_step(model, ref, opt, x, states, z, np.array([[1.0, 0.0]]), None, cfg)
    after = torch.log_softmax(model.discrete(model.encode(x)), 1)[0, 3].item()
    assert after > before


def _stub_simulate(draws, seeds, n_jobs=1):
    return [np.full(416, d.width / 22.0, np.float32) for d in draws]


def _run(tmp_path, out, steps=None, stop_after=None, resume=False):
    start = tmp_path / "start.pt"
    if not start.exists():
        _big_model(1).save(start)
    rng = np.random.default_rng(0)
    targets = dict(x=torch.as_tensor(rng.random((6, 416)), dtype=torch.float32),
                   target_x=rng.random((6, 416)).astype(np.float32))
    cfg = fs.SimConfig(start=str(start), group=4, targets_per_step=2, max_sims=6 * 8, checkpoint_every=2, anchor=0.0,
                       lr=1e-3, steps=8)

    def on_step(step):
        if stop_after is not None and step == stop_after:
            (out / "STOP").write_text("")

    return fs.run(cfg, out, targets, None, simulate=_stub_simulate, on_step=on_step, resume=resume)


def test_resume_continues_the_same_stream(tmp_path):
    straight = _run(tmp_path, tmp_path / "a")
    stopped = _run(tmp_path, tmp_path / "b", stop_after=3)
    assert stopped["stopped"] and stopped["step"] == 3
    (tmp_path / "b" / "STOP").unlink()
    resumed = _run(tmp_path, tmp_path / "b", resume=True)
    assert resumed["step"] == straight["step"] == 6 and resumed["sims_used"] == straight["sims_used"] == 48
    a = InverseModel.load(tmp_path / "a" / "model.pt").state_dict()
    b = InverseModel.load(tmp_path / "b" / "model.pt").state_dict()
    for k in a:
        assert torch.allclose(a[k].float(), b[k].float(), atol=1e-7), k
