import importlib.util
from pathlib import Path

import numpy as np
import pytest

from shazam.inverse import data
from tbribbon import parent_data

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "generate"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gp = _load("generate_parent")
check_parent = _load("check_parent")


@pytest.fixture(autouse=True)
def short_devices(monkeypatch):
    """Ten cells per device keep the draws fast; the workers are forked, so they see the patch."""
    monkeypatch.setattr(parent_data, "device_cells_for", lambda params, edge: 10)


def test_shards_round_trip_and_split_by_index(tmp_path):
    gp.write_shard(tmp_path, 0, range(0, 40), n_jobs=2)       # small hosts: monkeypatch n_cells to 10 for speed
    a = data.load_split(tmp_path, "train"); b = data.load_split(tmp_path, "val"); c = data.load_split(tmp_path, "test")
    assert sorted(np.r_[a["index"], b["index"], c["index"]]) == list(range(40))
    assert set(b["index"] % 20) <= {18} and set(c["index"] % 20) <= {19}
    assert a["x"].shape[1] == 416 and a["params"].shape[1] == 12


def test_a_draw_is_reproducible_from_its_index():
    d1, d2 = gp.draw_with_resampling(7), gp.draw_with_resampling(7)
    assert d1[0] == d2[0] and np.array_equal(d1[2], d2[2])


def test_rerun_after_an_interrupted_shard_rewrites_it_without_duplicates(tmp_path):
    (tmp_path / "shard_00000.npz.tmp").write_bytes(b"partial")
    gp.main(["--out", str(tmp_path), "--n-draws", "30", "--shard-size", "20", "--n-jobs", "2"])
    gp.main(["--out", str(tmp_path), "--n-draws", "30", "--shard-size", "20", "--n-jobs", "2"])   # resumes, nothing redone
    idx = np.concatenate([np.load(p)["index"] for p in sorted(tmp_path.glob("shard_*.npz"))])
    assert sorted(idx) == list(range(30)) and not list(tmp_path.glob("*.tmp"))


def test_check_reports_ok_on_generated_shards(tmp_path):
    gp.write_shard(tmp_path, 0, range(20), n_jobs=2)
    assert check_parent.check(tmp_path)["ok"]


def test_cheap_rules_run_before_the_expensive_work(monkeypatch):
    """A host whose type is invisible is redrawn without computing channels or transport; the cap raises clearly."""
    def boom(*a, **k):
        raise AssertionError("expensive step ran for a rejected draw")
    monkeypatch.setattr(parent_data, "band_weights", lambda params: {t: 1.0 if i else 0.001 for i, t in enumerate(params.active_types)})
    monkeypatch.setattr(parent_data, "clean_channels", boom)
    monkeypatch.setattr(parent_data, "simulate_raw", boom)
    # single-type hosts give {C: 0.001} and are also rejected, so every attempt fails and the cap is reached
    with pytest.raises(RuntimeError, match=r"no acceptable host in 100 attempts.*invisible_type': 100"):
        gp.draw_with_resampling(3)


def test_stored_fields_manifest_and_resume(tmp_path):
    gp.main(["--out", str(tmp_path), "--n-draws", "4", "--shard-size", "2", "--n-jobs", "2"])
    files = sorted(tmp_path.glob("shard_*.npz"))
    stamps = [p.stat().st_mtime_ns for p in files]
    gp.main(["--out", str(tmp_path), "--n-draws", "4", "--shard-size", "2", "--n-jobs", "2"])
    assert [p.stat().st_mtime_ns for p in files] == stamps                 # finished shards are not redone
    m = data.load_manifest(tmp_path)
    assert [e["shard"] for e in m["shards"]] == [0, 1] and m["counts"]["n_draws"] == 4 and m["code_version"]
    assert m["prior_ranges"]["w"] == [4, 22] and (tmp_path / "shard_00001_rejections.json").exists()
    a = data.load_split(tmp_path, "train")
    assert list(a["index"]) == [0, 1, 2, 3]
    assert a["x"].dtype == np.float32 and a["clean"].dtype == np.int16 and a["params"].dtype == np.float64
    for r, i in enumerate(a["index"]):
        draw, n_imp, raw, clean, attempts, _ = gp.draw_with_resampling(int(i))
        assert data.draws(a, [r]) == [draw]                                 # the loader rebuilds the exact draw
        assert a["n_imp"][r] == n_imp and a["attempts"][r] == attempts and a["n_cells"][r] == 10
        assert np.array_equal(a["clean"][r], clean) and np.allclose(a["raw"][r], raw, atol=1e-6)
        assert a["active_atoms"][r] == 10 * gp.active_atoms_per_cell(draw.params, draw.edge, draw.width)
