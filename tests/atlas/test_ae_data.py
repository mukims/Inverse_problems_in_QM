# tests/atlas/test_ae_data.py
import numpy as np
import pytest

from atlaslib import CloudStore
from ae_data import SPEC, SPLITS, collect, load, save
from tbribbon.materials import make_model

DENS = (0.01, 0.02)


def _store(path, mid="triangular/zigzag/N4", seeds=1000, seed0=0):
    mat, edge, n = mid.split("/")
    m = make_model(mat, edge, int(n[1:]))
    e_t = SPEC.energies_t() / m.t_ev
    s = CloudStore(path)
    s.write_pristine(mid, e_t, np.full(e_t.size, 10.0))
    rng = np.random.default_rng(0)
    for d in DENS:
        s.write_cloud(mid, d, m.impurities_for_density(d), rng.uniform(0, 3, (seeds, e_t.size)), np.arange(seed0, seed0 + seeds), e_t)
    return s


@pytest.fixture
def root(tmp_path):
    _store(tmp_path / "cat")
    _store(tmp_path / "pil")
    return tmp_path


def _run(root, split, **kw):
    return collect(split, root=root, stores=("cat",), pilots=("pil",), legacy_dir=None, **kw)


def test_splits_use_only_their_own_seeds(root):
    for split, (lo, hi) in SPLITS.items():
        d = _run(root, split)
        real = d["seed"][d["seed"] >= 0]
        assert real.min() == lo and real.max() == hi
        assert d["x"].shape[1] == 416 and d["x"].dtype == np.float32
        assert d["x"].min() >= 0 and d["x"].max() <= 1


def test_clean_spectrum_joins_the_training_split_once_at_density_zero(root):
    tr = _run(root, "train")
    clean = (tr["seed"] == -1)
    assert clean.sum() == 1 and tr["dens"][clean][0] == 0 and tr["src"][clean][0] == 0
    assert (_run(root, "val")["seed"] == -1).sum() == 0 and (_run(root, "test")["seed"] == -1).sum() == 0


def test_every_cloud_is_labelled_with_its_source_and_density(root):
    d = _run(root, "val")
    assert set(d["src"]) == {0, 1} and np.allclose(np.unique(d["dens"]), DENS)
    assert list(d["devices"]) == ["triangular/zigzag/N4"] and len(d["x"]) == 2 * 2 * 150


def test_per_cloud_thins_evenly(root):
    d = _run(root, "test", per_cloud=10)
    assert len(d["x"]) == 2 * 2 * 10 and np.all(np.diff(np.sort(d["seed"][d["src"] == 0][d["dens"][d["src"] == 0] == 0.01])) > 0)


def test_the_unseen_square_strip_is_never_collected(root):
    _store(root / "cat", mid="square/strip/N10", seeds=1000)
    assert "square/strip/N10" not in list(_run(root, "train")["devices"])


def test_legacy_sets_map_counts_to_densities(tmp_path):
    mm = np.random.default_rng(0).uniform(0, 3, (3, 1000, 300))
    np.save(tmp_path / "size_7.npy", mm)
    np.save(tmp_path / "size_9.npy", mm)
    d = collect("test", root=tmp_path, stores=(), pilots=(), legacy_dir=tmp_path)
    assert set(d["src"]) == {2}
    assert np.allclose(np.unique(d["dens"]), np.unique([2 / 1400, 4 / 1400, 6 / 1400, 2 / 1800, 4 / 1800, 6 / 1800]))
    assert d["seed"].min() == 850 and d["seed"].max() == 999 and list(d["devices"]) == ["graphene-ideal/armchair/N7", "graphene-ideal/armchair/N9"]


def test_save_and_load_round_trip(root, tmp_path):
    d = _run(root, "val", per_cloud=5)
    save(d, tmp_path / "cache")
    back = load(tmp_path / "cache")
    assert all(np.array_equal(d[k], back[k]) for k in d)
