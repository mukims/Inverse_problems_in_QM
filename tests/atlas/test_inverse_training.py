import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from shazam.inverse import data
from shazam.inverse.model import InverseModel
from tbribbon import parent_data
from tbribbon.parent_prior import excluded

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
    assert h["cfg"]["batch"] == 512 and h["code_version"] == data.load_manifest(shard)["code_version"]
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
