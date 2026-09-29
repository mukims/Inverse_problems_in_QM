import numpy as np
import pytest
from atlaslib import Atlas, InputSpec, Registry
try:
    from toy import E, toy_spectrum, toy_store
except ImportError:
    from tests.atlas.toy import E, toy_spectrum, toy_store

FAST = dict(latent=8, epochs=6, patience=3, k=7, refs_per_model=200, threads=2)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("atlas")
    store, models = toy_store(tmp)
    reg = Registry(models)
    return Atlas.build(store, reg, reg.ids(), InputSpec(), **FAST), tmp


def test_identifies_material_and_width_of_unseen_seeds(built):
    atlas, _ = built
    T = np.stack([toy_spectrum(3.0, 9, 0.02, 50_000 + i) for i in range(30)])
    res = atlas.locate(T, E, band_top_t=3.0)
    assert np.mean([r.material == "beta" for r in res]) >= 0.9
    assert np.median([r.width for r in res]) == pytest.approx(9, abs=1.0)


def test_width_between_neighbours_for_held_out_width(built):
    atlas, _ = built
    res = atlas.locate(np.stack([toy_spectrum(1.0, 11, 0.01, 60_000 + i) for i in range(30)]), E, 3.0)
    assert 9 <= np.median([r.width for r in res]) <= 14


def test_width_beyond_range_is_flagged(built):
    atlas, _ = built
    res = atlas.locate(np.stack([toy_spectrum(1.0, 40, 0.01, 70_000 + i) for i in range(20)]), E, 3.0)
    assert np.mean([r.width_extrapolated for r in res]) >= 0.8
    assert max(r.width for r in res) <= 14


def test_unfamiliar_spectrum_is_unknown(built):
    atlas, _ = built
    weird = np.tile(np.abs(np.sin(E * 40)) * 8, (5, 1))
    assert all(r.unknown for r in atlas.locate(weird, E, band_top_t=4.0))


def test_save_load_round_trip(built):
    atlas, tmp = built
    atlas.save(tmp / "saved")
    again = Atlas.load(tmp / "saved")
    T = toy_spectrum(1.0, 7, 0.01, 80_000)[None]
    assert atlas.locate(T, E, 3.0) == again.locate(T, E, 3.0)
