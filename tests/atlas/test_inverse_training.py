import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from shazam.inverse import data
from shazam.inverse.encoding import STATES, encode
from shazam.inverse.model import InverseModel
from tbribbon import parent_data
from tbribbon.parent import TYPES
from tbribbon.parent_prior import EDGES, excluded, sample_prior

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(folder, name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / folder / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")

gp = _load("generate", "generate_parent")
ti = _load("build", "train_inverse")


@pytest.fixture(scope="module")
def shard(tmp_path_factory):
    """200 draws (180 train, 10 validation, 10 test) on small hosts; the workers are forked, so they see the patches."""
    root = tmp_path_factory.mktemp("parent_smoke")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(parent_data, "device_cells_for", lambda params, edge: 10)
        orig = gp.sample_prior
        mp.setattr(gp, "sample_prior", lambda rng: (lambda d: d._replace(width=4 + d.width % 3))(orig(rng)))
        gp.write_shard(root, 0, range(200), n_jobs=4)
    return root


@pytest.fixture(autouse=True)
def two_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def test_training_on_smoke_shards_lowers_the_validation_loss(shard, tmp_path):
    # Fixed seed 0 and the default batch (512 = one step per epoch on 180 rows). With 156 states and 180 rows most
    # validation states are unseen, so the loss only falls reliably in this early regime: seeds 0-2 all fall each epoch,
    # while many small-batch steps overfit and the 10-row validation loss turns around within three epochs.
    ti.main(["--data", str(shard), "--out", str(tmp_path), "--epochs", "3", "--seed", "0", "--threads", "2"])
    h = json.loads((tmp_path / "history.json").read_text())
    assert len(h["val_loss"]) == 3 and len(h["train_loss"]) == 3
    assert h["val_loss"][-1] < h["val_loss"][0]
    assert h["n_train"] == 180 and h["n_val"] == 10 and h["excluded"] == {"hide": "none", "train": 0, "val": 0}
    assert h["best_epoch"] in (0, 1, 2) and h["val_loss"][h["best_epoch"]] == min(h["val_loss"])
    from shazam.inverse.runio import code_version
    assert h["cfg"]["batch"] == 512 and h["code_version"] == code_version()            # the trainer's own version
    assert h["data_provenance"]["code_versions"] == [data.load_manifest(shard)["code_version"]]   # and the data's
    assert h["data_provenance"]["n_draws"] == 200
    model = InverseModel.load(tmp_path / "model.pt")
    assert model.cfg["masks_fixed"] and bool(model.mask_known.all())


def test_exclude_removes_the_hidden_neighbourhood(shard, tmp_path, monkeypatch):
    train, val = data.load_split(shard, "train"), data.load_split(shard, "val")
    for split, n in ((train, 6), (val, 2)):                    # make some rows hBN-like (H only, delta near 1.576)
        split["on"][:n] = (False, True, False)
        split["lead"][:n] = 1
        split["params"][:n, data.PARAM_COLUMNS.index("delta")] = 1.576
    bad = {"train": np.array([excluded(d, "hbn") for d in data.draws(train, slice(None))]),
           "val": np.array([excluded(d, "hbn") for d in data.draws(val, slice(None))])}
    assert bad["train"].sum() >= 6 and bad["val"].sum() >= 2
    seen = []
    orig = InverseModel.loss

    def spy(self, x, state, z, mask):
        seen.append(x.detach().clone())
        return orig(self, x, state, z, mask)
    monkeypatch.setattr(InverseModel, "loss", spy)
    cfg = ti.TrainConfig(epochs=1, batch=64, exclude="hbn")
    _, h = ti.train(train, val, cfg)
    seen = torch.cat(seen).numpy()
    for split, flag in ((train, bad["train"]), (val, bad["val"])):
        for row in split["x"][flag]:
            assert not (seen == row).all(1).any()               # no excluded row was trained on or validated on
        for row in split["x"][~flag][:5]:
            assert (seen == row).all(1).any()                   # kept rows were
    assert h["excluded"] == {"hide": "hbn", "train": int(bad["train"].sum()), "val": int(bad["val"].sum())}
    assert h["n_train"] == len(train["x"]) - h["excluded"]["train"]
    assert h["n_val"] == len(val["x"]) - h["excluded"]["val"]


def _synthetic(n, first, informative=True, seed=0):
    """Arrays as `train` consumes them, from prior draws; x = [one_hot(state), z] @ P / 10 + noise (or pure noise)."""
    P = torch.randn(len(STATES) + 14, 416, generator=torch.Generator().manual_seed(123))
    draws = [sample_prior(np.random.default_rng(first + i)) for i in range(n)]
    enc = [encode(d) for d in draws]
    k = torch.tensor([e[0] for e in enc]); z = torch.tensor(np.stack([e[1] for e in enc]), dtype=torch.float32)
    noise = 0.01 * torch.randn(n, 416, generator=torch.Generator().manual_seed(seed + first))
    x = (torch.cat([torch.nn.functional.one_hot(k, len(STATES)).float(), z], 1) @ P / 10 if informative else 0) + noise
    return dict(index=np.arange(first, first + n), x=x.numpy().astype(np.float32),
                on=np.array([[d.params.on_C, d.params.on_H, d.params.on_K] for d in draws]),
                lead=np.array([TYPES.index(d.params.lead) for d in draws]),
                params=np.array([[getattr(d.params, c) for c in data.PARAM_COLUMNS] for d in draws]),
                edge=np.array([EDGES.index(d.edge) for d in draws]), width=np.array([d.width for d in draws]),
                density=np.array([d.density for d in draws]))


def test_training_learns_a_synthetic_inverse_problem_and_noise_does_not():
    cfg = ti.TrainConfig(epochs=4, batch=64, seed=0)
    _, h = ti.train(_synthetic(1500, 0), _synthetic(200, 10**6), cfg)
    assert all(b < a for a, b in zip(h["train_loss"], h["train_loss"][1:]))
    assert h["val_loss"][-1] < 0.8 * h["val_loss"][0]
    _, hn = ti.train(_synthetic(1500, 0, informative=False), _synthetic(200, 10**6, informative=False), cfg)
    assert hn["val_loss"][-1] > 0.8 * hn["val_loss"][0]               # the same threshold rejects spectra without information


def test_cli_records_the_excluded_counts(shard, tmp_path):
    ti.main(["--data", str(shard), "--out", str(tmp_path), "--exclude", "hbn", "--epochs", "1", "--threads", "2"])
    h = json.loads((tmp_path / "history.json").read_text())
    ex = {d: sum(excluded(r, "hbn") for r in data.draws(data.load_split(shard, d), slice(None))) for d in ("train", "val")}
    assert h["excluded"] == {"hide": "hbn", **ex} and h["n_train"] == 180 - ex["train"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["history.json", "model.pt"]          # no .tmp left behind


def test_a_non_finite_loss_stops_the_run_and_bad_epochs_are_refused():
    tr, va = _synthetic(40, 0), _synthetic(10, 10**6)
    with pytest.raises(ValueError, match="epochs"):
        ti.train(tr, va, ti.TrainConfig(epochs=0))
    tr["x"][0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="epoch 0"):
        ti.train(tr, va, ti.TrainConfig(epochs=2, batch=64))


def test_an_excluded_run_never_writes_into_model_v1(shard, tmp_path, monkeypatch):
    import paths
    assert ti.default_out("none") == ti.DEFAULT_OUT == paths.results_dir("inverse_design", "model_v1")
    assert ti.default_out("kagome") == paths.results_dir("inverse_design", "hidden_kagome")
    monkeypatch.setattr(ti.paths, "results_dir", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(ti, "DEFAULT_OUT", tmp_path / "inverse_design" / "model_v1")
    ti.main(["--data", str(shard), "--exclude", "hbn", "--epochs", "1", "--threads", "2"])        # no --out
    h = json.loads((tmp_path / "inverse_design" / "hidden_hbn" / "history.json").read_text())
    assert h["excluded"]["hide"] == "hbn" and not (tmp_path / "inverse_design" / "model_v1").exists()
    with pytest.raises(SystemExit, match="model_v1"):
        ti.main(["--data", str(shard), "--exclude", "hbn", "--epochs", "1", "--out", str(ti.DEFAULT_OUT)])
    assert not (tmp_path / "inverse_design" / "model_v1").exists()
