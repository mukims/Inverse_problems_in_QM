import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("organise_data", Path(__file__).resolve().parents[2] / "scripts" / "data" / "organise_data.py")
od = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(od)

GRID = np.arange(6) * 0.5


def make_ribbon(path, densities, n_cells=125, eta=1.2e-5, grid=GRID, fill=0.0):
    """A ribbon directory in CloudStore layout with one tiny cloud per density."""
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "energies_t.npy", grid)
    np.save(path / "pristine.npy", np.ones_like(grid))
    clouds = {}
    for d in densities:
        key = f"d{d:.4f}"
        np.save(path / f"cloud_{key}.npy", np.full((2, grid.size), fill + d))
        np.save(path / f"cloud_{key}_seeds.npy", np.arange(2))
        clouds[key] = {"density": d, "n": 2, "formula": "caroli", "lead_eta": eta, "n_cells": n_cells}
    meta = {"pristine_formula": "caroli", "pristine_lead_eta": eta, "pristine_n_cells": n_cells, "clouds": clouds}
    (path / "meta.json").write_text(json.dumps(meta))
    return path


def row(action, src, dst, size=0):
    return dict(action=action, source=str(src), destination=str(dst), reason="test", bytes=str(size))


def log_rows(log):
    return list(csv.DictReader(open(log)))


def test_move_row_renames_a_directory(tmp_path):
    src = make_ribbon(tmp_path / "run" / "hbn" / "armchair" / "N10", [0.01])
    dst = tmp_path / "data" / "hbn" / "armchair" / "N10"
    od.execute([row("MOVE", src, dst)], tmp_path / "MOVES.csv")
    assert (dst / "cloud_d0.0100.npy").exists()
    assert not src.exists()


def test_archive_row_creates_missing_parent_directories(tmp_path):
    src = make_ribbon(tmp_path / "run" / "x" / "zigzag" / "N5", [0.01])
    dst = tmp_path / "archives" / "atlas_store_runs" / "run" / "x" / "zigzag" / "N5"
    od.execute([row("ARCHIVE", src, dst)], tmp_path / "MOVES.csv")
    assert (dst / "meta.json").exists() and not src.exists()


def test_copy_row_keeps_the_source_and_copies_every_byte(tmp_path):
    src = make_ribbon(tmp_path / "home" / "sq", [0.01, 0.02])
    dst = tmp_path / "data" / "square" / "legacy"
    od.execute([row("COPY", src, dst)], tmp_path / "MOVES.csv")
    assert od.tree_size(src) == od.tree_size(dst) > 0
    assert (src / "meta.json").exists()


def test_preflight_rejects_an_existing_destination_and_nothing_is_touched(tmp_path):
    a = make_ribbon(tmp_path / "run" / "a", [0.01])
    b = make_ribbon(tmp_path / "run" / "b", [0.01])
    taken = make_ribbon(tmp_path / "data" / "b", [0.02])
    rows = [row("MOVE", a, tmp_path / "data" / "a"), row("MOVE", b, taken)]
    with pytest.raises(od.PlanError, match="already exists"):
        od.execute(rows, tmp_path / "MOVES.csv")
    assert a.exists() and not (tmp_path / "data" / "a").exists()


def test_preflight_rejects_a_missing_source(tmp_path):
    with pytest.raises(od.PlanError, match="missing"):
        od.execute([row("MOVE", tmp_path / "nope", tmp_path / "data" / "x")], tmp_path / "MOVES.csv")


def test_running_the_same_plan_twice_changes_nothing_the_second_time(tmp_path):
    src = make_ribbon(tmp_path / "run" / "a", [0.01])
    cp = make_ribbon(tmp_path / "home" / "c", [0.01])
    rows = [row("MOVE", src, tmp_path / "data" / "a"), row("COPY", cp, tmp_path / "data" / "c")]
    od.execute(rows, tmp_path / "MOVES.csv")
    od.execute(rows, tmp_path / "MOVES.csv")
    statuses = [r["status"] for r in log_rows(tmp_path / "MOVES.csv")]
    assert statuses == ["done", "done", "skipped", "skipped"]
    assert (tmp_path / "data" / "a" / "meta.json").exists()


def test_merge_adds_only_the_new_density_files_and_meta_entries(tmp_path):
    canon = make_ribbon(tmp_path / "run3" / "hbn" / "armchair" / "N9", [0.0048, 0.01], fill=1.0)
    extra = make_ribbon(tmp_path / "conc_v2" / "hbn" / "armchair" / "N9", [0.01, 0.0027, 0.0049], fill=5.0)
    dst = tmp_path / "data" / "hbn" / "armchair" / "N9"
    od.execute([row("MOVE", canon, dst), row("MERGE", extra, dst)], tmp_path / "MOVES.csv")
    meta = json.loads((dst / "meta.json").read_text())
    assert sorted(meta["clouds"]) == ["d0.0027", "d0.0048", "d0.0049", "d0.0100"]
    # merged-in clouds say where they came from; the canonical ones stay unmarked
    assert meta["clouds"]["d0.0027"]["source"] == "conc_v2" and "source" not in meta["clouds"]["d0.0048"]
    assert (dst / "cloud_d0.0027.npy").exists() and (dst / "cloud_d0.0049_seeds.npy").exists()
    # the density both copies had keeps the canonical spectra, not the conc_v2 ones
    assert np.load(dst / "cloud_d0.0100.npy")[0, 0] == pytest.approx(1.01)


def test_rows_are_applied_in_dependency_order_whatever_order_they_are_listed(tmp_path):
    canon = make_ribbon(tmp_path / "run3" / "m" / "e" / "N9", [0.0048])
    extra = make_ribbon(tmp_path / "conc_v2" / "m" / "e" / "N9", [0.0027])
    dst = tmp_path / "data" / "m" / "e" / "N9"
    arch = tmp_path / "archives" / "atlas_store_runs" / "conc_v2" / "m" / "e" / "N9"
    shuffled = [row("ARCHIVE", extra, arch), row("MERGE", extra, dst), row("MOVE", canon, dst)]
    od.execute(shuffled, tmp_path / "MOVES.csv")
    assert (dst / "cloud_d0.0027.npy").exists()
    assert (arch / "cloud_d0.0027.npy").exists()


def test_directories_left_empty_by_moves_are_removed_but_the_root_stays(tmp_path):
    root = tmp_path / "atlas_store"
    src = make_ribbon(root / "run" / "hbn" / "armchair" / "N10", [0.01])
    od.execute([row("MOVE", src, tmp_path / "data" / "hbn" / "armchair" / "N10")], tmp_path / "MOVES.csv")
    od.prune_empty(root)
    assert root.exists() and list(root.iterdir()) == []


def test_prune_never_removes_a_directory_that_still_has_files(tmp_path):
    root = tmp_path / "atlas_store"
    (root / "run" / "keep").mkdir(parents=True)
    (root / "run" / "keep" / "note.txt").write_text("x")
    (root / "run" / "empty").mkdir()
    od.prune_empty(root)
    assert (root / "run" / "keep" / "note.txt").exists() and not (root / "run" / "empty").exists()


def plan(root):
    rows = []
    od.plan_atlas_store(root, lambda action, src, dst, reason, size: rows.append(dict(action=action, source=str(src), destination=str(dst), reason=reason)))
    return rows


def test_planner_takes_the_newest_run_and_archives_the_older_copy(tmp_path):
    store = tmp_path / "atlas_store"
    make_ribbon(store / "materials3_ev_full" / "hbn" / "armchair" / "N10", [0.01])
    make_ribbon(store / "materials_ev_full" / "hbn" / "armchair" / "N10", [0.01], n_cells=100)
    rows = plan(tmp_path)
    moved = [r for r in rows if r["action"] == "MOVE"]
    assert [Path(r["source"]).parts[-4] for r in moved] == ["materials3_ev_full"]
    assert Path(moved[0]["destination"]) == tmp_path / "data" / "hbn" / "armchair" / "N10"
    old = [r for r in rows if r["action"] == "ARCHIVE" and "materials_ev_full" in r["source"]]
    assert len(old) == 1 and "archives/atlas_store_runs/materials_ev_full" in old[0]["destination"]


def test_planner_skips_a_run_that_has_no_clouds_for_the_ribbon(tmp_path):
    store = tmp_path / "atlas_store"
    make_ribbon(store / "engine_v1" / "square" / "strip" / "N10", [])
    make_ribbon(store / "novelty_v1" / "square" / "strip" / "N10", [0.01, 0.02])
    rows = plan(tmp_path)
    moved = [r for r in rows if r["action"] == "MOVE"]
    assert len(moved) == 1 and "novelty_v1" in moved[0]["source"]


def test_planner_merges_conc_v2_when_device_and_grid_match_and_archives_the_source(tmp_path):
    store = tmp_path / "atlas_store"
    make_ribbon(store / "materials3_ev_full" / "hbn" / "armchair" / "N9", [0.0048])
    make_ribbon(store / "conc_v2" / "hbn" / "armchair" / "N9", [0.0027])
    rows = plan(tmp_path)
    kinds = sorted(r["action"] for r in rows if Path(r["source"]).parts[-4] == "conc_v2")
    assert kinds == ["ARCHIVE", "MERGE"]


def test_planner_archives_conc_v2_without_merging_when_device_length_differs(tmp_path):
    store = tmp_path / "atlas_store"
    make_ribbon(store / "materials3_ev_full" / "hbn" / "armchair" / "N9", [0.0048], n_cells=125)
    make_ribbon(store / "conc_v2" / "hbn" / "armchair" / "N9", [0.0027], n_cells=100)
    rows = plan(tmp_path)
    kinds = [r["action"] for r in rows if Path(r["source"]).parts[-4] == "conc_v2"]
    assert kinds == ["ARCHIVE"]
