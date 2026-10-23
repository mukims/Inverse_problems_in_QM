from pathlib import Path

import pytest

import paths


def test_data_root_follows_the_environment_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSMISSION_DATA", str(tmp_path / "mydata"))
    assert paths.data_root() == tmp_path / "mydata"


def test_data_root_defaults_to_a_data_folder_in_the_repository(monkeypatch):
    monkeypatch.delenv("TRANSMISSION_DATA", raising=False)
    assert paths.data_root() == paths.repo_root() / "data"


def test_the_repository_root_is_the_folder_that_holds_src():
    assert (paths.repo_root() / "src" / "paths.py").exists()


def test_ribbon_dir_follows_material_edge_and_width(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSMISSION_DATA", str(tmp_path))
    assert paths.ribbon_dir("hbn", "armchair", 10) == tmp_path / "hbn" / "armchair" / "N10"


def test_model_dir_takes_the_model_id_used_by_the_store(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSMISSION_DATA", str(tmp_path))
    assert paths.model_dir("mos2/zigzag/N9") == tmp_path / "mos2" / "zigzag" / "N9"


def test_leads_dir_is_per_material(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSMISSION_DATA", str(tmp_path))
    assert paths.leads_dir("square") == tmp_path / "square" / "leads"


def test_archives_sit_next_to_the_data_folder_unless_overridden(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    monkeypatch.setenv("TRANSMISSION_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("TRANSMISSION_ARCHIVES", raising=False)
    assert paths.archives_root() == tmp_path / "archives"
    assert paths.archive_run("smoke_v1") == tmp_path / "archives" / "atlas_store_runs" / "smoke_v1"
    monkeypatch.setenv("TRANSMISSION_ARCHIVES", str(tmp_path / "elsewhere"))
    assert paths.archives_root() == tmp_path / "elsewhere"


def test_archives_follow_a_symlinked_data_folder_to_its_real_parent(monkeypatch, tmp_path):
    real = tmp_path / "storage" / "data"
    real.mkdir(parents=True)
    link = tmp_path / "repo" / "data"
    link.parent.mkdir()
    link.symlink_to(real)
    monkeypatch.setenv("TRANSMISSION_DATA", str(link))
    monkeypatch.delenv("TRANSMISSION_ARCHIVES", raising=False)
    assert paths.archives_root() == tmp_path / "storage" / "archives"


def test_results_dir_mirrors_the_old_notebooks_layout_inside_the_repository():
    assert paths.results_dir("material_atlas", "lookup_v4") == paths.repo_root() / "results" / "material_atlas" / "lookup_v4"
    assert paths.results_dir() == paths.repo_root() / "results"


def test_results_dir_can_create_the_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    d = paths.results_dir("agnr", "multi_width", create=True)
    assert d.is_dir()


def test_legacy_array_paths_name_the_consolidated_agnr_and_square_sets(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSMISSION_DATA", str(tmp_path))
    assert paths.agnr_consolidated(7) == tmp_path / "graphene-ideal" / "armchair" / "N7" / "legacy" / "size_7.npy"
    assert paths.square_combined(10) == tmp_path / "square" / "strip" / "N10" / "legacy" / "size_10_combined"


def test_raw_output_lives_in_the_archives_unless_overridden(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    monkeypatch.setenv("TRANSMISSION_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("TRANSMISSION_ARCHIVES", raising=False)
    monkeypatch.delenv("TRANSMISSION_RAW", raising=False)
    assert paths.raw_root() == tmp_path / "archives" / "raw_simulations"
    monkeypatch.setenv("TRANSMISSION_RAW", str(tmp_path / "scratch"))
    assert paths.raw_root() == tmp_path / "scratch"


def test_runs_that_became_the_data_tree_resolve_to_it_and_the_rest_to_the_archive(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    monkeypatch.setenv("TRANSMISSION_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("TRANSMISSION_ARCHIVES", raising=False)
    for name in ("engine_v1", "materials3_ev_full", "novelty_v1"):
        assert paths.store_root(name) == tmp_path / "data"
    assert paths.store_root("materials_ev_full") == tmp_path / "archives" / "atlas_store_runs" / "materials_ev_full"


def test_the_consolidated_agnr_folder_holds_both_arrays_side_by_side(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSMISSION_DATA", str(tmp_path))
    assert paths.agnr_consolidated_dir() == tmp_path / "graphene-ideal" / "armchair" / "consolidated"


def test_require_names_the_variable_when_the_data_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSMISSION_DATA", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="TRANSMISSION_DATA"):
        paths.require(paths.ribbon_dir("hbn", "armchair", 10))


def test_require_returns_an_existing_path_unchanged(tmp_path):
    assert paths.require(tmp_path) == tmp_path


def test_square_pristine_sits_beside_the_combined_square_set(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSMISSION_DATA", str(tmp_path))
    assert paths.square_pristine(10) == tmp_path / "square" / "strip" / "N10" / "legacy" / "pristine_10.npy"
