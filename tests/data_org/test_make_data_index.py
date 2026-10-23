import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("make_data_index", Path(__file__).resolve().parents[2] / "scripts" / "make_data_index.py")
mdi = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mdi)


def ribbon(path, densities, n_cells=125, eta=1.2e-5, legacy=False, seeds=3):
    path.mkdir(parents=True)
    np.save(path / "energies_t.npy", np.arange(4) * 0.5)
    np.save(path / "pristine.npy", np.ones(4))
    clouds = {f"d{d:.4f}": {"density": d, "n": seeds, "formula": "caroli"} for d in densities}
    meta = {"pristine_formula": "caroli", "pristine_lead_eta": eta, "pristine_n_cells": n_cells, "clouds": clouds}
    (path / "meta.json").write_text(json.dumps(meta))
    if legacy:
        (path / "legacy").mkdir()
    return path


def read(out):
    return list(csv.DictReader(open(out)))


def test_one_row_per_ribbon_with_its_identity_and_density_range(tmp_path):
    ribbon(tmp_path / "hbn" / "armchair" / "N10", [0.0048, 0.04])
    mdi.write_index(tmp_path, tmp_path / "INDEX.csv")
    (r,) = read(tmp_path / "INDEX.csv")
    assert (r["material"], r["edge"], r["width"], r["model_id"]) == ("hbn", "armchair", "10", "hbn/armchair/N10")
    assert (r["n_densities"], r["density_min"], r["density_max"]) == ("2", "0.0048", "0.04")
    assert (r["n_cells"], r["lead_eta"], r["formula"]) == ("125", "1.2e-05", "caroli")


def test_rows_are_sorted_by_material_edge_and_numeric_width(tmp_path):
    for w in (10, 5, 27):
        ribbon(tmp_path / "hbn" / "armchair" / f"N{w}", [0.01])
    ribbon(tmp_path / "graphene-ideal" / "zigzag" / "N7", [0.01])
    mdi.write_index(tmp_path, tmp_path / "INDEX.csv")
    assert [r["model_id"] for r in read(tmp_path / "INDEX.csv")] == [
        "graphene-ideal/zigzag/N7", "hbn/armchair/N5", "hbn/armchair/N10", "hbn/armchair/N27"]


def test_legacy_arrays_and_missing_clouds_are_reported_not_hidden(tmp_path):
    ribbon(tmp_path / "graphene-ideal" / "armchair" / "N7", [0.01], legacy=True)
    ribbon(tmp_path / "square" / "strip" / "N10", [])
    mdi.write_index(tmp_path, tmp_path / "INDEX.csv")
    rows = {r["model_id"]: r for r in read(tmp_path / "INDEX.csv")}
    assert rows["graphene-ideal/armchair/N7"]["has_legacy"] == "yes"
    assert rows["square/strip/N10"]["n_densities"] == "0" and rows["square/strip/N10"]["has_legacy"] == "no"


def test_leads_and_other_non_ribbon_folders_are_ignored(tmp_path):
    ribbon(tmp_path / "hbn" / "armchair" / "N10", [0.01])
    (tmp_path / "hbn" / "leads" / "size_10").mkdir(parents=True)
    (tmp_path / "graphene-ideal" / "leads").mkdir(parents=True)
    (tmp_path / "README.md").write_text("x")
    mdi.write_index(tmp_path, tmp_path / "INDEX.csv")
    assert [r["model_id"] for r in read(tmp_path / "INDEX.csv")] == ["hbn/armchair/N10"]


def test_counts_per_material_summarise_the_index(tmp_path):
    ribbon(tmp_path / "hbn" / "armchair" / "N10", [0.01])
    ribbon(tmp_path / "hbn" / "zigzag" / "N10", [0.01])
    ribbon(tmp_path / "mos2" / "armchair" / "N10", [0.01])
    rows = mdi.write_index(tmp_path, tmp_path / "INDEX.csv")
    assert mdi.counts_by_material(rows) == {"hbn": 2, "mos2": 1}
