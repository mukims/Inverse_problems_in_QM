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


def test_regime_accuracy_bins_width_edge_density_and_the_cross_hopping_of_two_type_lattices():
    ep = _load("evaluate", "eval_pattern")
    from tbribbon.parent_prior import FIELDS
    n = 8
    params = np.zeros((n, len(FIELDS)))
    params[:, FIELDS.index("t_CH")] = [0.1, 0.1, 0.9, 0.9, 0, 0, 0, 0]
    va = dict(width=np.array([4, 5, 20, 21, 10, 10, 10, 10]), edge=np.array([0, 0, 1, 1, 0, 1, 0, 1]),
              density=np.array([0.003, 0.003, 0.04, 0.04, 0.003, 0.003, 0.003, 0.003]), params=params)
    y = np.array([NAMES.index("CH")] * 4 + [NAMES.index("H")] * 4)
    hit = np.array([0, 0, 1, 1, 1, 1, 1, 1], bool)
    r = ep.regime_accuracy(hit, va, y)
    assert r["all"]["width"]["4-8"] == dict(n=2, top1=0.0) and r["all"]["width"]["19-22"] == dict(n=2, top1=1.0)
    assert r["two_type"]["cross"]["0.05-0.3"] == dict(n=2, top1=0.0)
    assert r["two_type"]["cross"]["0.6-1"] == dict(n=2, top1=1.0)
    assert r["all"]["edge"]["armchair"]["n"] == 4 and r["two_type"]["width"]["9-13"]["n"] == 0
    assert r["all"]["density"]["0.0025-0.005"]["n"] == 6


def test_ensembles_average_the_members_state_probabilities():
    ep = _load("evaluate", "eval_pattern")
    a, b = np.array([[0.8, 0.2], [0.1, 0.9]]), np.array([[0.2, 0.8], [0.3, 0.7]])
    out = ep.ensemble_probs({"m0": a, "m1": b, "m2": None}, ["m0", "m1"])
    assert np.allclose(out, [[0.5, 0.5], [0.2, 0.8]])
    with pytest.raises(ValueError):
        ep.ensemble_probs({"m0": a, "m2": None}, ["m0", "m2"])                     # a lattice-type-only model


def test_cli_distils_from_teacher_folders_and_scores_ensembles_with_regimes(shard, tmp_path):
    ti, ep = _load("build", "train_inverse"), _load("evaluate", "eval_pattern")
    common = ["--data", str(shard), "--input", "bandtop", "--extra-conv", "1", "--epochs", "1", "--threads", "2"]
    for s in (0, 1):
        ti.main(common + ["--out", str(tmp_path / f"t{s}"), "--seed", str(s)])
    ti.main(common + ["--out", str(tmp_path / "student"), "--seed", "4",
                      "--teacher", str(tmp_path / "t0"), "--teacher", str(tmp_path / "t1")])
    h = json.loads((tmp_path / "student" / "history.json").read_text())
    assert h["cfg"]["teachers"] == [str(tmp_path / f"t{s}" / "model.pt") for s in (0, 1)]
    assert (h["cfg"]["distill_alpha"], h["cfg"]["distill_temp"]) == (0.7, 2.0) and len(h["train_kd"]) == 1
    out = tmp_path / "eval.json"
    ep.main([str(tmp_path / m) for m in ("t0", "t1", "student")]
            + ["--data", str(shard), "--out", str(out), "--ensemble-sizes", "2"])
    r = json.loads(out.read_text())["models"]
    assert set(r) == {str(tmp_path / m) for m in ("t0", "t1", "student")} | {"ensemble2"}
    assert "mcnemar_vs_ref" in r["ensemble2"] and set(r["ensemble2"]["regimes"]) == {"all", "two_type"}
