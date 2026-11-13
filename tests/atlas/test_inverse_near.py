"""generate_parent.py --near: training draws restricted to the neighbourhood of known materials."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from shazam.inverse import data
from shazam.inverse import realism as rl
from shazam.inverse.data import PARAM_COLUMNS
from tbribbon import parent_data
from tbribbon.parent import NAMED

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
START = 3_200_000_000


def _load(folder, name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / folder / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _small_hosts(mp, gp):
    mp.setattr(parent_data, "device_cells_for", lambda params, edge: 10)
    orig = gp.sample_prior
    mp.setattr(gp, "sample_prior", lambda rng: (lambda d: d._replace(width=4 + d.width % 3))(orig(rng)))


def _distances(z, name):
    arrays = {k: z[k] for k in ("on", "params")}
    names, d = rl.nearest_material_rows(arrays, materials=(name,))
    return d


def test_near_spec_keeps_the_block_order_and_refuses_bad_plans():
    gp = _load("generate", "generate_parent")
    spec = gp.near_spec([("hbn", 3), ("kagome", 2)], radius=2.0)
    assert spec == dict(materials=["hbn", "kagome"], counts=[3, 2], radius=2.0, rel_tol=rl.REL_TOL, abs_tol=rl.ABS_TOL)
    assert [gp.near_material(spec, k) for k in range(5)] == ["hbn"] * 3 + ["kagome"] * 2
    with pytest.raises(ValueError, match="outside"):
        gp.near_material(spec, 5)
    with pytest.raises(ValueError, match="unknown material"):
        gp.near_spec([("diamond", 3)])
    with pytest.raises(ValueError, match="count"):
        gp.near_spec([("hbn", 0)])
    with pytest.raises(ValueError, match="twice"):
        gp.near_spec([("hbn", 2), ("hbn", 2)])


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_near_shards_hold_each_materials_block_inside_the_radius_and_record_the_plan(tmp_path):
    gp = _load("generate", "generate_parent")
    spec = gp.near_spec([("hbn", 25), ("kagome", 15)], radius=2.0)
    with pytest.MonkeyPatch.context() as mp:
        _small_hosts(mp, gp)
        gp.write_shard(tmp_path, 0, range(START, START + 40), n_jobs=4, shard_size=40, index_start=START, near=spec)
        with pytest.raises(ValueError, match="near-material plan"):
            gp.write_shard(tmp_path, 0, range(START, START + 40), n_jobs=1, shard_size=40, index_start=START,
                           near=gp.near_spec([("hbn", 40)]))
        with pytest.raises(ValueError, match="near-material plan"):
            gp.write_shard(tmp_path, 0, range(START, START + 40), n_jobs=1, shard_size=40, index_start=START)
        assert gp.NEAR is None                                       # a plain call leaves no plan behind
    with np.load(tmp_path / "shard_00000.npz") as z:
        assert list(z["index"]) == list(range(START, START + 40))
        code = {n: NAMED.index(n) for n in NAMED}
        assert list(z["near"]) == [code["hbn"]] * 25 + [code["kagome"]] * 15
        assert (_distances(z, "hbn")[:25] <= 2.0).all() and (_distances(z, "kagome")[25:] <= 2.0).all()
        assert (z["tries"] >= z["attempts"]).all() and z["tries"].sum() > 40 * 20        # these regions are rare
        s = z["params"][:, PARAM_COLUMNS.index("s")]
        assert s[:25].min() >= 0.8 * 2.30 - 1e-9 and s[:25].max() <= 1.2 * 2.30 + 1e-9
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["near"] == spec and manifest["index_start"] == START and manifest["proposal"] is None
    tr = data.load_split(tmp_path, "train", ("index", "x", "near"))
    assert len(tr["index"]) == 36 and set(tr["index"] % 20) <= set(range(18))


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_a_near_draw_is_reproducible_from_its_index(tmp_path):
    gp = _load("generate", "generate_parent")
    spec = gp.near_spec([("silicene", 4)])
    with pytest.MonkeyPatch.context() as mp:
        _small_hosts(mp, gp)
        gp.write_shard(tmp_path / "a", 0, range(START, START + 4), n_jobs=1, shard_size=4, index_start=START, near=spec)
        gp.write_shard(tmp_path / "b", 0, range(START, START + 4), n_jobs=2, shard_size=4, index_start=START, near=spec)
    with np.load(tmp_path / "a" / "shard_00000.npz") as a, np.load(tmp_path / "b" / "shard_00000.npz") as b:
        for key in ("x", "params", "width", "density", "near", "tries"):
            assert np.array_equal(a[key], b[key])


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_cli_near_sets_the_number_of_draws_resumes_and_refuses_a_proposal_with_it(tmp_path):
    gp = _load("generate", "generate_parent")
    args = ["--out", str(tmp_path), "--near", "triangular=6", "kagome=4", "--index-start", str(START),
            "--shard-size", "6", "--n-jobs", "2"]
    with pytest.MonkeyPatch.context() as mp:
        _small_hosts(mp, gp)
        gp.main(args)
        first = {p.name: p.stat().st_mtime_ns for p in tmp_path.glob("shard_*.npz")}
        gp.main(args)                                                            # resumes: nothing is redone
        assert first == {p.name: p.stat().st_mtime_ns for p in tmp_path.glob("shard_*.npz")}
        with pytest.raises(SystemExit, match="--near and --proposal"):
            gp.main(args + ["--proposal", str(tmp_path / "cells.json")])
        with pytest.raises(SystemExit, match="NAME=COUNT"):
            gp.main(["--out", str(tmp_path / "x"), "--near", "triangular"])
        with pytest.raises(SystemExit, match="near-material plan"):
            gp.main(["--out", str(tmp_path), "--near", "triangular=10", "--index-start", str(START),
                     "--shard-size", "6", "--n-jobs", "1"])
    near = np.concatenate([np.load(p)["near"] for p in sorted(tmp_path.glob("shard_*.npz"))])
    assert list(near) == [NAMED.index("triangular")] * 6 + [NAMED.index("kagome")] * 4
    assert json.loads((tmp_path / "manifest.json").read_text())["counts"]["n_draws"] == 10


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_training_adds_near_rows_unweighted_and_keeps_validation_on_the_prior_data(tmp_path):
    gp, ti = _load("generate", "generate_parent"), _load("build", "train_inverse")
    with pytest.MonkeyPatch.context() as mp:
        _small_hosts(mp, gp)
        gp.write_shard(tmp_path / "v1", 0, range(120), n_jobs=4)
        gp.write_shard(tmp_path / "near", 0, range(START, START + 40), n_jobs=4, shard_size=40, index_start=START,
                       near=gp.near_spec([("kagome", 25), ("triangular", 15)]))
    base = ["--data", str(tmp_path / "v1"), "--epochs", "1", "--batch", "32", "--input", "bandtop", "--threads", "2"]
    out = tmp_path / "model"
    ti.main(base + ["--add-data", str(tmp_path / "near"), "--out", str(out)])
    h = json.loads((out / "history.json").read_text())
    n_v1 = int(data.split_mask(np.arange(120), "train").sum())
    n_near = int(data.split_mask(np.arange(START, START + 40), "train").sum())
    assert h["n_train"] == n_v1 + n_near and h["n_val"] == int(data.split_mask(np.arange(120), "val").sum())
    assert h["added_data"] == [dict(root=str(tmp_path / "near"), n_train=n_near,
                                    provenance=data.data_provenance(tmp_path / "near"))]
    assert "weight_stats" not in h and "extra_data" not in h
    with pytest.raises(SystemExit, match="--add-data"):
        ti.main(base + ["--add-data", str(tmp_path / "near"), "--extra-data", str(tmp_path / "near"),
                        "--out", str(tmp_path / "m2")])
    with pytest.raises(SystemExit, match="--add-data"):
        ti.main(base + ["--add-data", str(tmp_path / "near"), "--drop-pattern", "K", "--out", str(tmp_path / "m3")])
    with pytest.raises(SystemExit, match="--add-data"):
        ti.main(base + ["--add-data", str(tmp_path / "v1"), "--out", str(tmp_path / "m4")])      # the same draws twice
