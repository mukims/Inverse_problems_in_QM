import importlib.util
import os
import subprocess
import sys
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


# pytest's own process already runs BLAS threads, which makes fork() warn; the generator script itself forks before
# it has any thread (checked separately), so the warning is test-process noise.
pytestmark = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")

gp = _load("generate_parent")
check_parent = _load("check_parent")


@pytest.fixture(autouse=True)
def small_hosts(monkeypatch):
    """Ten cells per device and widths 4 to 6 keep the draws fast; the workers are forked, so they see the patches."""
    monkeypatch.setattr(parent_data, "device_cells_for", lambda params, edge: 10)
    orig = gp.sample_prior
    monkeypatch.setattr(gp, "sample_prior", lambda rng: (lambda d: d._replace(width=4 + d.width % 3))(orig(rng)))


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
    assert m["code_version"][:7].isalnum() and m["prior_ranges"]["w"] == [4, 22] and (tmp_path / "shard_00001_rejections.json").exists()
    a = data.load_split(tmp_path, "train")
    assert list(a["index"]) == [0, 1, 2, 3]
    assert a["x"].dtype == np.float32 and a["clean"].dtype == np.int16 and a["params"].dtype == np.float64
    for r, i in enumerate(a["index"]):
        draw, n_imp, raw, clean, attempts, _ = gp.draw_with_resampling(int(i))
        assert data.draws(a, [r]) == [draw]                                 # the loader rebuilds the exact draw
        assert a["n_imp"][r] == n_imp and a["attempts"][r] == attempts and a["n_cells"][r] == 10
        assert np.array_equal(a["clean"][r], clean) and np.allclose(a["raw"][r], raw, atol=1e-6)
        assert a["active_atoms"][r] == 10 * gp.active_atoms_per_cell(draw.params, draw.edge, draw.width)


def test_a_worker_that_raises_stops_the_shard(monkeypatch, tmp_path):
    def broken(index):
        raise ValueError(f"boom {index}")
    monkeypatch.setattr(gp, "_record", broken)
    with pytest.raises(RuntimeError, match=r"shard 3: draw index \d+ failed in a worker[\s\S]*boom"):
        gp.write_shard(tmp_path, 3, range(0, 4), n_jobs=2)
    assert not list(tmp_path.glob("shard_*.npz"))


def test_a_dead_worker_is_noticed_while_the_others_keep_working(monkeypatch):
    import os
    import time

    def draw(index):
        time.sleep(0.2)                                  # slow-ish stub draws: 3 workers stream results meanwhile
        if index == 1:
            os._exit(3)
        return {"index": index}
    monkeypatch.setattr(gp, "_record", draw)
    t0 = time.time()
    with pytest.raises(RuntimeError, match=r"shard 0: a worker exited with code 3 while computing draw index 1$"):
        gp._compute(range(60), 3, label="shard 0")
    assert time.time() - t0 < 5          # the other draws alone would take 60 * 0.2 / 2 = 6 s


def test_a_stalled_shard_stops(monkeypatch):
    import time
    monkeypatch.setattr(gp, "_record", lambda index: time.sleep(60))
    monkeypatch.setattr(gp, "STALL_SECONDS", 1)
    with pytest.raises(RuntimeError, match=r"no result for 1 s.*\[[01], [01]\]"):
        gp._compute(range(2), 2, label="shard 0")


def test_the_process_exits_after_a_failed_large_shard(tmp_path):
    """A dead worker leaves thousands of undelivered tasks; the parent must still exit instead of hanging."""
    code = (
        "import importlib.util, os, sys\n"
        f"spec = importlib.util.spec_from_file_location('generate_parent', {str(SCRIPTS / 'generate_parent.py')!r})\n"
        "gp = importlib.util.module_from_spec(spec); spec.loader.exec_module(gp)\n"
        "gp._record = lambda index: os._exit(3)\n"
        "gp._compute(range(200000), 2, label='shard 0')\n")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert res.returncode != 0 and "exited with code 3" in res.stderr


def test_write_shard_refuses_an_empty_shard(tmp_path):
    with pytest.raises(ValueError, match="no draw indices"):
        gp.write_shard(tmp_path, 0, range(0), n_jobs=2)


def test_write_shard_keeps_the_shard_size_of_an_existing_data_set(tmp_path):
    (tmp_path / "manifest.json").write_text('{"shard_size": 20, "shards": []}')
    with pytest.raises(ValueError, match="shard size 20"):
        gp.write_shard(tmp_path, 0, range(20), n_jobs=2, shard_size=10)
    with pytest.raises(ValueError, match="at most 20"):
        gp.write_shard(tmp_path, 0, range(30), n_jobs=2)
    with pytest.raises(ValueError, match="must start at 20"):
        gp.write_shard(tmp_path, 1, range(30, 40), n_jobs=2)
    assert not list(tmp_path.glob("shard_*"))


# ------------------------------------------------------------------------------- provenance and backward compatibility

REPO = Path(__file__).resolve().parents[2]
BASE = "96d65f8f3"            # the last commit whose generator wrote shards without per-shard provenance
report_parent = _load("report_parent")


def _base_module(rel, name, folder):
    res = subprocess.run(["git", "show", f"{BASE}:{rel}"], cwd=REPO, capture_output=True, text=True)
    if res.returncode:
        pytest.skip(f"base commit {BASE} is not available")
    path = folder / f"{name}.py"
    path.write_text(res.stdout)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def old_gp(tmp_path_factory, monkeypatch):
    """The generator and parent_data of the base commit (the code of the running data step), on the same small hosts."""
    folder = tmp_path_factory.mktemp("base_code")
    pd = _base_module("src/tbribbon/parent_data.py", "parent_data_base", folder)
    g = _base_module("scripts/generate/generate_parent.py", "generate_parent_base", folder)
    g.parent_data = pd
    monkeypatch.setattr(pd, "device_cells_for", lambda params, edge: 10)
    orig = g.sample_prior
    monkeypatch.setattr(g, "sample_prior", lambda rng: (lambda d: d._replace(width=4 + d.width % 3))(orig(rng)))
    return g


def test_the_new_generator_writes_bit_identical_draws_and_only_adds_fields(tmp_path, old_gp):
    old_gp.write_shard(tmp_path / "old", 0, range(12), n_jobs=2)
    gp.write_shard(tmp_path / "new", 0, range(12), n_jobs=2)
    with np.load(tmp_path / "old" / "shard_00000.npz") as a, np.load(tmp_path / "new" / "shard_00000.npz") as b:
        assert set(b.files) == set(a.files) | {"code_version"}
        for k in a.files:
            assert b[k].dtype == a[k].dtype and b[k].shape == a[k].shape
            if k != "seconds":                                          # wall time differs, nothing else may
                assert np.array_equal(a[k], b[k]), k
        assert b["code_version"].shape == () and str(b["code_version"]) == gp.CODE_VERSION
    assert ((tmp_path / "old" / "shard_00000_rejections.json").read_text()
            == (tmp_path / "new" / "shard_00000_rejections.json").read_text())
    (eo,), (en,) = data.load_manifest(tmp_path / "old")["shards"], data.load_manifest(tmp_path / "new")["shards"]
    assert set(en) == set(eo) | {"code_version"} and en["code_version"] == gp.CODE_VERSION
    assert {k: v for k, v in en.items() if k not in ("seconds", "code_version")} == {k: v for k, v in eo.items() if k != "seconds"}


def test_the_new_generator_extends_a_folder_the_old_one_started(tmp_path, old_gp):
    argv = ["--out", str(tmp_path), "--shard-size", "2", "--n-jobs", "2"]
    old_gp.main(argv + ["--n-draws", "4"])
    old_files = sorted(tmp_path.glob("shard_*.npz"))
    stamps = [p.stat().st_mtime_ns for p in old_files]
    gp.main(argv + ["--n-draws", "6"])
    assert [p.stat().st_mtime_ns for p in old_files] == stamps            # the old shards are kept as they are
    m = data.load_manifest(tmp_path)
    assert [e["shard"] for e in m["shards"]] == [0, 1, 2] and m["counts"]["n_draws"] == 6
    assert "code_version" not in m["shards"][0] and "code_version" not in m["shards"][1]
    assert m["shards"][2]["code_version"] == m["code_version"] == gp.CODE_VERSION
    for split in ("train", "val", "test"):                             # every key: the 0-d code_version is not a column
        got = data.load_split(tmp_path, split)
        assert "code_version" not in got and all(len(v) == len(got["index"]) for v in got.values())
    assert list(data.load_split(tmp_path, "train")["index"]) == list(range(6))
    assert check_parent.check(tmp_path)["ok"] and report_parent.report(tmp_path)["n_draws"] == 6
    prov = data.data_provenance(tmp_path)
    assert prov["n_draws"] == 6 and set(prov["code_versions"]) == {"unknown", gp.CODE_VERSION}
    assert prov["manifest_code_version"] == gp.CODE_VERSION and prov["root"] == str(tmp_path)


def test_the_code_version_is_fixed_when_the_generator_starts(tmp_path, monkeypatch):
    """A commit made while a multi-day run is going must not relabel the shards it writes afterwards."""
    monkeypatch.setattr(gp, "CODE_VERSION", "0123abc-at-start")
    gp.main(["--out", str(tmp_path), "--n-draws", "2", "--shard-size", "2", "--n-jobs", "2"])
    m = data.load_manifest(tmp_path)
    assert m["code_version"] == m["shards"][0]["code_version"] == "0123abc-at-start"
    with np.load(tmp_path / "shard_00000.npz") as z:
        assert str(z["code_version"]) == "0123abc-at-start"
    assert data.data_provenance(tmp_path)["code_versions"] == ["0123abc-at-start"]


def test_seeded_rows_are_a_sorted_reproducible_choice():
    a = data.seeded_rows(100, 10)
    assert list(a) == sorted(a) and len(set(a)) == 10 and list(a) == list(data.seeded_rows(100, 10))
    assert len(data.seeded_rows(5, 10)) == 5
