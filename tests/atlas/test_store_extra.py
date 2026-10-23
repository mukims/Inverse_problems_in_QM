import numpy as np
import pytest

from atlaslib.store import CloudStore

GRID = np.arange(4) * 0.5
MID = "hbn/armchair/N9"


def _write(store, density, fill, **meta):
    store.write_cloud(MID, density, 3, np.full((2, GRID.size), fill), np.arange(2), GRID, **meta)


@pytest.fixture
def root(tmp_path):
    s = CloudStore(tmp_path)
    s.write_pristine(MID, GRID, np.ones(GRID.size))
    _write(s, 0.0100, 0.5)                              # a catalogue density
    _write(s, 0.0027, 0.6, source="conc_v2")            # merged in from another run
    return tmp_path


def test_clouds_merged_in_from_another_run_are_hidden_by_default(root):
    store = CloudStore(root)
    assert store.densities(MID) == [0.01]
    assert store.has_cloud(MID, 0.0100) and not store.has_cloud(MID, 0.0027)


def test_extra_true_shows_every_density_including_the_merged_ones(root):
    store = CloudStore(root, extra=True)
    assert store.densities(MID) == [0.0027, 0.01]
    assert store.has_cloud(MID, 0.0027)


def test_a_hidden_density_can_still_be_read_when_asked_for_by_value(root):
    spectra, seeds = CloudStore(root).read_cloud(MID, 0.0027)
    assert spectra[0, 0] == pytest.approx(0.6) and list(seeds) == [0, 1]


def test_a_store_without_any_merged_clouds_behaves_as_before(tmp_path):
    s = CloudStore(tmp_path)
    s.write_pristine(MID, GRID, np.ones(GRID.size))
    _write(s, 0.0100, 0.5)
    assert s.densities(MID) == [0.01] and s.has_cloud(MID, 0.01)
