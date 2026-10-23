from pathlib import Path
import tempfile
import numpy as np
import pytest

from atlaslib import CloudStore, MultiStore, InputSpec
from shazam import ae_data
from export_configurations import export_pairs
from tbribbon.materials import make_model


@pytest.fixture
def mini_store(tmp_path):
    root = tmp_path / "test_store"
    store = CloudStore(root)
    m = make_model("hbn", "armchair", 7)
    e_t = np.linspace(0, 3.0, 50)
    pris = np.ones(50)
    store.write_pristine(m.model_id, e_t, pris, formula="caroli", n_cells=25)

    clouds = np.ones((5, 50))
    seeds = np.array([0, 1, 2, 3, 4])
    store.write_cloud(m.model_id, 0.01, 7, clouds, seeds, e_t, formula="caroli", n_cells=25)
    return store, m


def test_cloudstore_read_configuration(mini_store):
    store, m = mini_store
    cfg = store.read_configuration(m.model_id, 0.01, 2)
    assert cfg.model_id == m.model_id
    assert cfg.seed == 2
    assert cfg.n_cells == 25
    assert cfg.n_impurities == 7
    assert cfg.coordinates.shape == (7, 2)
    assert np.all(np.isclose(np.abs(cfg.host_onsite_ev), 3.625))


def test_cloudstore_read_cloud_with_configurations(mini_store):
    store, m = mini_store
    specs, seeds, configs = store.read_cloud_with_configurations(m.model_id, 0.01)
    assert len(specs) == len(seeds) == len(configs) == 5
    for s, cfg in zip(seeds, configs):
        assert cfg.seed == s
        assert cfg.n_impurities == 7


def test_multistore_read_configuration(mini_store):
    store, m = mini_store
    multi = MultiStore(store)
    cfg = multi.read_configuration(m.model_id, 0.01, 3)
    assert cfg.seed == 3
    assert cfg.n_impurities == 7


def test_export_pairs_npz(mini_store, tmp_path):
    store, m = mini_store
    out_file = tmp_path / "exported.npz"
    res = export_pairs(store.root, [m.model_id], out_file, seeds=[0, 1, 2])
    assert out_file.exists()
    loaded = np.load(out_file, allow_pickle=True)
    assert len(loaded["spectra"]) == 3
    assert len(loaded["coords"]) == 3
    imp_t = np.asarray(loaded["impurity_onsite_t"][0], dtype=float)
    host_t = np.asarray(loaded["host_onsite_t"][0], dtype=float)
    v_t = float(loaded["defect_v_t"][0])
    assert np.allclose(imp_t, host_t + v_t)


def test_ae_data_attach_configurations():
    # Test on synthetic/mock data dict
    data = {
        "x": np.ones((3, 416), dtype=np.float32),
        "dev": np.array([0, 0, 0], dtype=np.int32),
        "dens": np.array([0.0, 0.01, 0.02], dtype=np.float32),
        "seed": np.array([-1, 10, 20], dtype=np.int32),
        "devices": np.array(["graphene-ideal/armchair/N7"]),
    }
    ae_data.attach_configurations(data)
    assert "configs" in data
    assert len(data["configs"]) == 3
    # Pristine spectrum
    assert data["configs"][0].n_impurities == 0
    # Dirty spectrum 1
    assert data["configs"][1].seed == 10
    assert data["configs"][1].n_impurities > 0
    # Dirty spectrum 2
    assert data["configs"][2].seed == 20
