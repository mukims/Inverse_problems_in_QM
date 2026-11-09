import numpy as np
import pytest
import torch

from shazam.inverse import active
from shazam.inverse.encoding import STATES, active_slots, encode, state_index, state_of
from shazam.inverse.model import InverseModel
from shazam.inverse.reward import FAMILY_OF, N_FAMILIES
from tbribbon.parent_prior import sample_prior


def test_cell_is_the_family_with_edge_times_the_width_bin():
    rng = np.random.default_rng(0)
    for _ in range(300):
        d = sample_prior(rng)
        c = active.cell_of(d)
        assert 0 <= c < active.N_CELLS == N_FAMILIES * 4
        fam, wb = divmod(c, active.N_WBINS)
        assert fam == FAMILY_OF[state_index(state_of(d))]
        assert wb == active.width_bin(d.width)
    assert [active.width_bin(w) for w in (4, 7, 8, 12, 13, 17, 18, 22)] == [0, 0, 1, 1, 2, 2, 3, 3]
    with pytest.raises(ValueError):
        active.width_bin(23)


def test_shrunk_rates_pull_small_cells_towards_their_group():
    groups = np.array([0, 0, 0, 0, 1, 1])
    n = np.array([1000, 1000, 1000, 2, 1000, 1000])
    k = np.array([100, 120, 80, 2, 500, 520])            # the 2-draw cell saw 2 errors out of 2
    r = active.shrink_rates(k, n, groups)
    assert np.all((r > 0) & (r < 1))
    assert abs(r[0] - 0.1) < 0.01                         # big cells stay near their raw rate
    assert r[3] < 0.5                                     # 2/2 is pulled hard towards the group's ~0.1
    assert abs(r[4] - 0.5) < 0.02


def test_shrunk_means_follow_the_standard_errors():
    groups = np.zeros(4, int)
    est = np.array([1.0, -1.0, 0.0, 4.0])
    exact = active.shrink_means(est, np.full(4, 1e-6), groups)
    assert np.allclose(exact, est, atol=1e-3)
    noisy = active.shrink_means(est, np.full(4, 1e3), groups)
    assert np.allclose(noisy, est.mean(), atol=1e-2)


def test_gain_needs_both_an_error_and_a_positive_slope():
    err = dict(family=np.array([0.5, 0.5, 0.0]), width=np.zeros(3), log_s=np.zeros(3))
    slope = dict(family=np.array([0.1, -0.1, 0.1]), width=np.zeros(3), log_s=np.zeros(3))
    g = active.expected_gain(err, slope)
    assert g[0] > 0 and g[1] == 0 and g[2] == 0


def test_tilted_proposal_keeps_half_the_prior_and_targets_the_top_quartile():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(40))
    gain = rng.gamma(0.5, 1.0, 40)
    q, lam = active.tilted_proposal(p, gain)
    assert q.sum() == pytest.approx(1.0) and lam >= 0
    assert np.all(q >= 0.5 * p - 1e-12)                   # p / q <= 2 everywhere
    tilted = (q - 0.5 * p) / 0.5
    top = gain >= np.quantile(gain, 0.75)
    assert tilted[top].sum() >= 0.6 - 1e-6
    q0, lam0 = active.tilted_proposal(p, np.zeros(40))    # no gain anywhere: the proposal is the prior
    assert np.allclose(q0, p)


def test_rejection_sampler_reproduces_the_designed_cell_frequencies():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    q = np.array([0.1, 0.2, 0.3, 0.4])
    accept = active.acceptance(p, q)
    rng = np.random.default_rng(2)
    cell = lambda d: d                                    # a fake prior whose "draws" are their own cells
    draws = [active.sample_proposal(rng, accept, sampler=lambda r: int(r.choice(4, p=p)), cell=cell)
             for _ in range(40_000)]
    freq = np.bincount(draws, minlength=4) / len(draws)
    se = np.sqrt(q * (1 - q) / len(draws))
    assert np.all(np.abs(freq - q) < 4 * se)


def test_pooled_weights_restore_the_prior_cell_histogram():
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(12))
    prior_cells = rng.choice(12, 20_000, p=p)
    q = rng.dirichlet(np.ones(12) * 0.3)
    pooled = np.concatenate([prior_cells, rng.choice(12, 5_000, p=q)])
    w = active.pooled_weights(pooled, prior_cells, n_cells=12)
    assert w.mean() == pytest.approx(1.0)
    hist_w = np.bincount(pooled, weights=w, minlength=12) / w.sum()
    hist_p = np.bincount(prior_cells, minlength=12) / len(prior_cells)
    assert np.allclose(hist_w, hist_p)
    assert np.allclose(active.pooled_weights(prior_cells, prior_cells, n_cells=12), 1.0)


def test_weighted_loss_with_unit_weights_equals_the_unweighted_loss():
    torch.manual_seed(0)
    rng = np.random.default_rng(4)
    enc = [encode(sample_prior(rng)) for _ in range(16)]
    state, z, mask = zip(*enc)
    state = torch.as_tensor(np.array(state)); z = torch.as_tensor(np.stack(z), dtype=torch.float32)
    mask = torch.as_tensor(np.stack(mask))
    masks = torch.as_tensor(np.stack([active_slots(s) for s in STATES]))
    model = InverseModel(len(STATES), channels=416, state_masks=masks)
    x = torch.randn(16, 416)
    with torch.random.fork_rng():
        a, _ = model.loss(x, state, z, mask)
    with torch.random.fork_rng():
        b, _ = model.loss(x, state, z, mask, weight=torch.ones(16))
    assert torch.allclose(a, b, atol=1e-6)
    _, c = model.loss(x, state, z, mask, weight=torch.cat([torch.ones(8), torch.zeros(8)]))   # zero-weight rows do not count
    _, d = model.loss(x[:8], state[:8], z[:8], mask[:8])
    assert c["ce"] == pytest.approx(d["ce"], rel=1e-5)


# ---------------------------------------------------------------- generation and training with a proposal

import importlib.util
import json
from pathlib import Path

from shazam.inverse import data
from tbribbon import parent_data

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(folder, name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / folder / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _small_hosts(mp, gp):
    mp.setattr(parent_data, "device_cells_for", lambda params, edge: 10)
    orig = gp.sample_prior
    mp.setattr(gp, "sample_prior", lambda rng: (lambda d: d._replace(width=4 + d.width % 3))(orig(rng)))


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_proposal_shards_hold_only_accepted_cells_at_disjoint_indices_and_train_with_weights(tmp_path):
    gp, ti = _load("generate", "generate_parent"), _load("build", "train_inverse")
    k_cells = np.array([c for c in range(active.N_CELLS)
                        if STATES[int(np.flatnonzero(FAMILY_OF == c // active.N_WBINS)[0])].pattern == (False, False, True)])
    accept = np.zeros(active.N_CELLS)
    accept[k_cells] = 1.0                                       # a proposal that keeps only kagome hosts
    start = 3_000_000_000
    with pytest.MonkeyPatch.context() as mp:
        _small_hosts(mp, gp)
        gp.write_shard(tmp_path / "v1", 0, range(120), n_jobs=4)
        mp.setattr(gp, "PROPOSAL", accept)
        gp.write_shard(tmp_path / "al", 0, range(start, start + 40), n_jobs=4, shard_size=40, index_start=start,
                       proposal=dict(file="test", accept=accept.tolist()))
        with pytest.raises(ValueError, match="started at index"):
            gp.write_shard(tmp_path / "al", 0, range(40), n_jobs=1, index_start=0)
    with np.load(tmp_path / "al" / "shard_00000.npz") as z:
        assert z["index"].min() == start and len(z["index"]) == 40
        recomputed = active.cells_of_arrays({k: z[k] for k in z.files})
        assert np.array_equal(z["cell"], recomputed)
        assert set(z["cell"]) <= set(k_cells)
    manifest = json.loads((tmp_path / "al" / "manifest.json").read_text())
    assert manifest["index_start"] == start and manifest["proposal"]["accept"] == accept.tolist()
    assert "index_start" not in json.loads((tmp_path / "v1" / "manifest.json").read_text())

    out = tmp_path / "model"
    ti.main(["--data", str(tmp_path / "v1"), "--extra-data", str(tmp_path / "al"), "--out", str(out), "--epochs", "1",
             "--batch", "32", "--input", "bandtop", "--threads", "2"])
    h = json.loads((out / "history.json").read_text())
    n_v1 = int(data.split_mask(np.arange(120), "train").sum())
    n_al = int(data.split_mask(np.arange(start, start + 40), "train").sum())
    assert h["n_train"] == n_v1 + n_al and h["extra_data"][0]["n_train"] == n_al
    assert h["weight_stats"]["min"] < 1 < h["weight_stats"]["max"] or h["weight_stats"]["min"] == 0
