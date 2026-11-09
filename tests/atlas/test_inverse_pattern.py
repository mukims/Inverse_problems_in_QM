import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from shazam.inverse import data
from shazam.inverse.pattern import NAMES, N_PATTERNS, PatternModel, pattern_labels
from tbribbon import parent_data
from tbribbon.parent_prior import PATTERNS

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
pytestmark = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")


def _load(folder, name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / folder / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def shard(tmp_path_factory):
    """200 draws (180 train, 10 validation, 10 test) on small hosts."""
    gp = _load("generate", "generate_parent")
    root = tmp_path_factory.mktemp("pattern_smoke")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(parent_data, "device_cells_for", lambda params, edge: 10)
        orig = gp.sample_prior
        mp.setattr(gp, "sample_prior", lambda rng: (lambda d: d._replace(width=4 + d.width % 3))(orig(rng)))
        gp.write_shard(root, 0, range(200), n_jobs=4)
    return root


def test_labels_are_the_seven_site_type_patterns():
    assert N_PATTERNS == 7 and sorted(NAMES) == sorted(["C", "H", "K", "CH", "CK", "HK", "CHK"])
    on = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.int8)
    assert [NAMES[k] for k in pattern_labels(on)] == ["C", "H", "K", "CHK", "HK"]
    assert all(PATTERNS[k] == tuple(bool(v) for v in r) for k, r in zip(pattern_labels(on), on))


def test_model_gives_a_probability_per_pattern_and_round_trips(tmp_path):
    torch.manual_seed(0)
    model = PatternModel(embed=32)
    x = torch.rand(5, model.cfg["channels"])
    p = model.predict_proba(x)
    assert p.shape == (5, N_PATTERNS) and torch.allclose(p.sum(1), torch.ones(5))
    model.save(tmp_path / "m.pt")
    again = PatternModel.load(tmp_path / "m.pt")
    assert again.cfg == model.cfg and torch.allclose(again.predict_proba(x), p)


def test_training_writes_the_model_history_and_val_scores(shard, tmp_path):
    tp = _load("build", "train_pattern")
    tp.main(["--data", str(shard), "--out", str(tmp_path), "--epochs", "3", "--threads", "2", "--embed", "32"])
    h = json.loads((tmp_path / "history.json").read_text())
    assert len(h["val_loss"]) == 3 and h["n_train"] == 180 and h["n_val"] == 10
    assert h["val_loss"][-1] < h["val_loss"][0]
    assert h["best_epoch"] == int(np.argmin(h["val_loss"]))
    s = h["val_scores"]
    assert 0 <= s["top1"] <= 1 and set(s["by_pattern"]) <= set(NAMES)
    assert sum(sum(r) for r in s["confusion"]) == 10
    assert PatternModel.load(tmp_path / "model.pt").cfg["embed"] == 32
    va = data.load_split(shard, "val", ("on",))
    assert s["n"] == len(va["on"])


def test_dropout_acts_only_in_training_and_old_checkpoints_still_load(tmp_path):
    torch.manual_seed(0)
    model = PatternModel(embed=32, dropout=0.5)
    x = torch.rand(8, model.cfg["channels"])
    assert torch.allclose(model.predict_proba(x), model.predict_proba(x))
    model.train()
    assert not torch.allclose(model.logits(x), model.logits(x))
    plain = PatternModel(embed=32)
    ck = {"cfg": {k: v for k, v in plain.cfg.items() if k != "dropout"}, "state_dict": plain.state_dict()}
    torch.save(ck, tmp_path / "old.pt")
    assert PatternModel.load(tmp_path / "old.pt").cfg["dropout"] == 0.0


def test_input_noise_scales_with_each_spectrum_and_leaves_the_band_top_alone():
    tp = _load("build", "train_pattern")
    x = torch.cat([torch.ones(2, 416) * torch.tensor([[1.0], [10.0]]), torch.tensor([[0.3], [0.7]])], 1)
    noisy = tp.add_noise(x, 0.05, torch.Generator().manual_seed(0))
    assert torch.equal(noisy[:, -1], x[:, -1])
    spread = (noisy - x)[:, :-1].std(1)
    assert torch.allclose(spread, torch.tensor([0.05, 0.5]), rtol=0.15)
    assert torch.equal(tp.add_noise(x, 0.0, None), x)


def test_regularised_training_records_its_settings(shard, tmp_path):
    tp = _load("build", "train_pattern")
    tp.main(["--data", str(shard), "--out", str(tmp_path), "--epochs", "2", "--threads", "2", "--embed", "32",
             "--dropout", "0.3", "--label-smoothing", "0.1", "--noise", "0.03"])
    h = json.loads((tmp_path / "history.json").read_text())
    assert (h["cfg"]["dropout"], h["cfg"]["label_smoothing"], h["cfg"]["noise"]) == (0.3, 0.1, 0.03)
    assert PatternModel.load(tmp_path / "model.pt").cfg["dropout"] == 0.3


def test_scoring_without_a_pattern_zeroes_it_and_renormalises_the_rest():
    ep = _load("evaluate", "eval_pattern")
    p = np.array([[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]])
    q = ep.without(p, [2])
    assert np.allclose(q, [[2 / 3, 1 / 3, 0], [0.5, 0.5, 0]]) and np.allclose(p[0], [0.5, 0.25, 0.25])
