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


def test_optimizer_is_adamw_and_the_plateau_schedule_halves_the_rate_when_validation_stalls():
    model = InverseModel(3)
    opt, sched = ti.make_optimizer(model, ti.TrainConfig(lr=1e-3, weight_decay=1e-2, schedule="plateau"))
    assert isinstance(opt, torch.optim.AdamW) and opt.param_groups[0]["weight_decay"] == 1e-2
    sched.step(1.0)
    for _ in range(ti.PLATEAU_PATIENCE + 1):               # no improvement for longer than the schedule's patience
        sched.step(1.0)
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-4)
    # default: weight decay 0.01 and a constant rate; weight_decay=0 reproduces model_v1's plain Adam
    opt, sched = ti.make_optimizer(model, ti.TrainConfig())
    assert opt.param_groups[0]["weight_decay"] == 0.01 and sched is None
    assert ti.make_optimizer(model, ti.TrainConfig(weight_decay=0.0))[0].param_groups[0]["weight_decay"] == 0


def test_linear_input_is_the_cleaned_spectrum_without_the_log_and_training_can_use_it(shard):
    va = data.load_split(shard, "val", ti.NEEDED + ("raw",))
    lin = parent_data.to_input_linear(va["raw"])
    assert lin.dtype == np.float32 and lin.shape == va["raw"].shape and lin.min() >= 0
    # same rounding, despike and clip as the log input: undoing the log gives the same T (to float32 precision)
    assert np.allclose(np.expm1(va["x"].astype(np.float64) * np.log1p(parent_data.V4.cap)), lin, atol=2e-5 * 64)
    tensors, _ = ti.prepare(va, input="linear")
    assert torch.equal(tensors["x"], torch.as_tensor(lin))
    with pytest.raises(ValueError):
        ti.prepare(va, input="sqrt")


def test_bandtop_input_resamples_each_spectrum_to_its_own_band_top_and_keeps_the_top(shard):
    va = data.load_split(shard, "val", ti.NEEDED + ("raw",))
    lin = parent_data.to_input_linear(va["raw"])
    bt = parent_data.to_input_bandtop(va["raw"])
    assert bt.dtype == np.float32 and bt.shape == (len(lin), 417)
    for row_lin, row_bt in zip(lin, bt):
        k = np.nonzero(row_lin > 0)[0].max() + 1                      # channels up to the band top
        assert row_bt[-1] == pytest.approx(k / 416)                   # band top as a fraction of the window
        assert row_bt[0] == row_lin[0] and row_bt[415] == pytest.approx(row_lin[k - 1])
        assert row_bt[:416].max() <= row_lin.max() + 1e-6           # interpolation adds no new extremes
    # a spectrum whose signal fills the window is resampled onto itself
    full = np.linspace(1, 3, 416)[None]
    assert np.allclose(parent_data.to_input_bandtop(full)[0, :416], parent_data.to_input_linear(full)[0])
    tensors, _ = ti.prepare(va, input="bandtop")
    assert tensors["x"].shape == (len(lin), 417)


def test_pristine_target_uses_the_inputs_band_top(shard):
    va = data.load_split(shard, "val", ti.NEEDED + ("raw", "clean"))
    before = parent_data.to_input_bandtop(va["raw"])
    tgt = parent_data.pristine_target_bandtop(va["clean"], va["raw"])
    assert tgt.dtype == np.float32 and tgt.shape == (len(va["raw"]), 416)
    k = parent_data.bandtop_k(parent_data.to_input_linear(va["raw"]))
    for i in range(len(tgt)):
        expect = np.interp(np.linspace(0, k[i] - 1, 416), np.arange(k[i]), va["clean"][i, :k[i]].astype(float))
        assert np.allclose(tgt[i], expect, atol=1e-5)
    assert tgt.min() >= 0 and tgt.max() <= va["clean"].max()
    assert np.array_equal(parent_data.to_input_bandtop(va["raw"]), before)
    assert np.array_equal(parent_data.bandtop_k(np.zeros((1, 416))), [416])


def test_autoencoder_training_reads_only_train_rows_and_writes_its_files(shard, tmp_path, monkeypatch):
    ta = _load("build", "train_autoencoder")
    seen = {}
    orig = ta.make_arrays

    def spy(arrays):
        seen.setdefault("index", []).append(arrays["index"].copy())
        return orig(arrays)
    monkeypatch.setattr(ta, "make_arrays", spy)
    out = tmp_path / "ae"
    hist = ta.main(["--data", str(shard), "--out", str(out), "--encoder", "mlp", "--mlp-hidden", "32", "16",
                    "--epochs", "2", "--batch", "32", "--threads", "2"])
    tr_idx, va_idx = seen["index"]
    assert np.all(np.isin(tr_idx % 20, range(18))) and np.all(va_idx % 20 == 18)
    assert (out / "ae.pt").exists() and (out / "history.json").exists()
    h = json.loads((out / "history.json").read_text())
    assert len(h["val_loss"]) == 2 and h["baseline_val_mse"] > 0 and h["cfg"]["encoder"] == "mlp"
    assert hist["best_epoch"] in (0, 1)


def test_init_encoder_copies_and_freeze_keeps_the_weights(shard, tmp_path):
    from shazam.inverse.autoencoder import SpectrumAutoencoder
    torch.manual_seed(3)
    ae = SpectrumAutoencoder(encoder="mlp", mlp_hidden=(32, 16), dropout=0.1)
    ae.save(tmp_path / "ae.pt")
    tr, va = data.load_split(shard, "train", ti.NEEDED + ("raw",)), data.load_split(shard, "val", ti.NEEDED + ("raw",))
    base = dict(epochs=1, batch=64, input="bandtop", encoder="mlp", mlp_hidden=(32, 16), dropout=0.1,
                init_encoder=str(tmp_path / "ae.pt"))
    frozen, h = ti.train(tr, va, ti.TrainConfig(**base, freeze_encoder=True))
    assert h["cfg"]["freeze_encoder"] and h["cfg"]["init_encoder"].endswith("ae.pt")
    for a, b in zip(ae.encoder.state_dict().values(), frozen.encoder.state_dict().values()):
        assert torch.equal(a, b)                                            # frozen: bitwise unchanged
    tuned, _ = ti.train(tr, va, ti.TrainConfig(**base))
    assert any(not torch.equal(a, b) for a, b in zip(ae.encoder.state_dict().values(),
                                                     tuned.encoder.state_dict().values()))
    with pytest.raises(ValueError, match="mlp_hidden"):
        ti.train(tr, va, ti.TrainConfig(**dict(base, mlp_hidden=(64, 16))))


def test_embed_sets_the_encoder_output_both_heads_read_and_is_recorded(shard, tmp_path):
    ti.main(["--data", str(shard), "--out", str(tmp_path), "--input", "bandtop", "--extra-conv", "1", "--embed", "832",
             "--epochs", "1", "--threads", "2"])
    model = InverseModel.load(tmp_path / "model.pt")
    assert model.cfg["embed"] == 832
    assert model.encode(torch.zeros(3, model.cfg["channels"])).shape == (3, 832)
    assert model.discrete[0].in_features == 832
    assert json.loads((tmp_path / "history.json").read_text())["cfg"]["embed"] == 832


def test_drop_pattern_removes_those_lattices_from_train_and_val_and_leaves_their_states_unknown(shard, tmp_path):
    from shazam.inverse.pattern import NAMES, pattern_labels
    tr, va = (data.load_split(shard, s, ("on",)) for s in ("train", "val"))
    chk = NAMES.index("CHK")
    n_tr, n_va = (int((pattern_labels(s["on"]) != chk).sum()) for s in (tr, va))
    assert n_tr < len(tr["on"])                                   # the smoke shard has CHK rows to drop
    ti.main(["--data", str(shard), "--out", str(tmp_path), "--input", "bandtop", "--extra-conv", "1",
             "--drop-pattern", "CHK", "--epochs", "1", "--threads", "2"])
    h = json.loads((tmp_path / "history.json").read_text())
    assert (h["n_train"], h["n_val"]) == (n_tr, n_va)
    assert h["drop_pattern"] == ["CHK"]
    model = InverseModel.load(tmp_path / "model.pt")
    is_chk = torch.tensor([s.pattern == (True, True, True) for s in STATES])
    assert not model.mask_known[is_chk].any() and model.mask_known[~is_chk].any()


def test_teachers_give_their_tempered_average_as_soft_targets_and_distilled_training_records_them(tmp_path):
    tr, va = _synthetic(600, 0), _synthetic(100, 10**6)
    paths_ = []
    for seed in (0, 1):
        t, _ = ti.train(tr, va, ti.TrainConfig(epochs=1, batch=64, seed=seed))
        t.save(tmp_path / f"t{seed}.pt"); paths_.append(str(tmp_path / f"t{seed}.pt"))
    x = torch.as_tensor(tr["x"][:50])
    teachers = [InverseModel.load(p) for p in paths_]
    soft = ti.teacher_targets(teachers, x, temp=2.0)
    with torch.no_grad():
        want = torch.stack([torch.softmax(m.discrete(m.encode(x)).masked_fill(~m.mask_known, -torch.inf) / 2.0, 1)
                            for m in teachers]).mean(0)
    assert soft.shape == (50, len(STATES)) and torch.allclose(soft, want, atol=1e-6)
    assert torch.allclose(soft.sum(1), torch.ones(50), atol=1e-5)
    cfg = ti.TrainConfig(epochs=2, batch=64, seed=2, teachers=tuple(paths_), distill_alpha=0.7, distill_temp=2.0)
    _, h = ti.train(tr, va, cfg)
    assert list(h["cfg"]["teachers"]) == paths_ and h["cfg"]["distill_alpha"] == 0.7
    assert len(h["train_kd"]) == 2 and all(np.isfinite(h["train_kd"]))
    with pytest.raises(ValueError):
        ti.train(tr, va, ti.TrainConfig(epochs=1, batch=64, teachers=tuple(paths_), input="bandtop"))   # input mismatch


def test_cli_trains_with_the_bandtop_scale_coordinate_and_refuses_it_without_the_bandtop_input(shard, tmp_path):
    ti.main(["--data", str(shard), "--out", str(tmp_path), "--input", "bandtop", "--extra-conv", "1", "--scale", "bandtop",
             "--epochs", "1", "--threads", "2"])
    h = json.loads((tmp_path / "history.json").read_text())
    assert h["cfg"]["scale"] == "bandtop" and np.isfinite(h["val_loss"]).all()
    model = InverseModel.load(tmp_path / "model.pt")
    assert model.cfg["scale"] == "bandtop"
    with pytest.raises((SystemExit, ValueError)):
        ti.main(["--data", str(shard), "--out", str(tmp_path / "bad"), "--scale", "bandtop", "--epochs", "1", "--threads", "2"])
