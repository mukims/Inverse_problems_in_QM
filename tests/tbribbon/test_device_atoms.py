# tests/tbribbon/test_device_atoms.py
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from atlaslib import CloudStore, InputSpec
from tbribbon.check_store import stored_device
from tbribbon.generate_clouds import generate
from tbribbon.materials import atoms_per_cell, device_cells, make_model

REPO = Path(__file__).resolve().parents[2]
V3 = InputSpec(version="v3")


def _meta(store, m):
    return json.loads((store._dir(m.model_id) / "meta.json").read_text())


def test_tmd_atoms_count_the_chalcogens():
    m = make_model("mos2", "zigzag", 7)                 # 7 Mo per cell in the model, 21 real atoms
    assert m.sites_per_cell // m.orbitals_per_site == 7
    assert atoms_per_cell(m) == 21
    assert atoms_per_cell(make_model("hbn", "zigzag", 7)) == 14


@pytest.mark.parametrize("material, edge, cells", [
    ("mos2", "zigzag", 83), ("mos2", "armchair", 42), ("hbn", "zigzag", 125),
    ("triangular", "zigzag", 250), ("kagome", "armchair", 42), ("checkerboard", "strip", 62)])
def test_one_length_per_material_and_edge_puts_n7_to_n9_near_2000_atoms(material, edge, cells):
    assert device_cells(material, edge, 2000) == cells
    for n in (7, 9):
        atoms = cells * atoms_per_cell(make_model(material, edge, n))
        assert 1500 < atoms < 2500


def test_device_atoms_sets_the_length_and_records_it(tmp_path):
    m = make_model("triangular", "zigzag", 4)
    cells = device_cells("triangular", "zigzag", 160)
    store = CloudStore(tmp_path / "s")
    generate(store, [m], [0.05], V3, n_jobs=1, formula="caroli", seeds=range(2), device_atoms=160)
    meta = _meta(store, m)
    assert meta["pristine_n_cells"] == cells
    (cloud,) = meta["clouds"].values()
    assert cloud["n_cells"] == cells
    assert cloud["n_impurities"] == replace(m, n_cells=cells).impurities_for_density(0.05)


def test_default_length_is_recorded_as_100_cells(tmp_path):
    m = make_model("triangular", "zigzag", 4)
    store = CloudStore(tmp_path / "s")
    generate(store, [m], [], V3, n_jobs=1, formula="caroli")
    assert _meta(store, m)["pristine_n_cells"] == 100


def test_resume_refuses_a_different_length(tmp_path):
    m = make_model("triangular", "zigzag", 4)
    store = CloudStore(tmp_path / "s")
    generate(store, [m], [], V3, n_jobs=1, formula="caroli")
    with pytest.raises(ValueError, match="n_cells=100; refusing"):
        generate(store, [m], [0.05], V3, n_jobs=1, formula="caroli", seeds=range(2), device_atoms=160)


def test_check_store_reads_the_stored_length(tmp_path):
    m = make_model("triangular", "zigzag", 4)
    store = CloudStore(tmp_path / "s")
    generate(store, [m], [], V3, n_jobs=1, formula="caroli", device_atoms=160)
    assert stored_device(store, m).n_cells == device_cells("triangular", "zigzag", 160)
    old = CloudStore(tmp_path / "old")
    generate(old, [m], [], V3, n_jobs=1, formula="caroli")
    meta = _meta(old, m)
    del meta["pristine_n_cells"]                          # a store written before this change
    (old._dir(m.model_id) / "meta.json").write_text(json.dumps(meta))
    assert stored_device(old, m).n_cells == 100


def test_graphene_armchair_rejects_device_atoms(tmp_path):
    with pytest.raises(ValueError, match="device_atoms does not apply to graphene armchair"):
        generate(CloudStore(tmp_path / "s"), [make_model("graphene-ideal", "armchair", 7)], [0.01],
                 InputSpec(version="v2"), n_jobs=1, seeds=range(2), device_atoms=2000)


def test_cli_flag_reaches_the_store(tmp_path):
    env = dict(os.environ, OMP_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join([str(REPO / "notebooks" / "material_atlas"), str(REPO / "notebooks")]))
    subprocess.run([sys.executable, str(REPO / "notebooks" / "tbribbon" / "generate_clouds.py"),
                    "--store", str(tmp_path / "s"), "--grid", "models", "--models", "triangular/zigzag/N4",
                    "--densities", "0.05", "--spec-version", "v3", "--formula", "caroli", "--n-seeds", "2",
                    "--n-jobs", "1", "--device-atoms", "160"], check=True, env=env, cwd=REPO, capture_output=True)
    meta = _meta(CloudStore(tmp_path / "s"), make_model("triangular", "zigzag", 4))
    assert meta["pristine_n_cells"] == device_cells("triangular", "zigzag", 160)
