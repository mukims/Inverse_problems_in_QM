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


def test_models_can_be_limited_to_ribbons_that_have_clouds(root):
    store = CloudStore(root)
    store.write_pristine("hbn/zigzag/N5", GRID, np.ones(GRID.size))        # a clean spectrum and nothing else
    assert store.models() == ["hbn/armchair/N9", "hbn/zigzag/N5"]
    assert store.models(with_clouds=True) == ["hbn/armchair/N9"]


def test_only_limits_the_models_listed_to_the_given_id_prefixes(tmp_path):
    s = CloudStore(tmp_path)
    for mid in ("graphene-ideal/armchair/N7", "square/strip/N10", "hbn/armchair/N9"):
        s.write_pristine(mid, GRID, np.ones(GRID.size))
    view = CloudStore(tmp_path, only=("graphene-ideal/", "square/"))
    assert view.models() == ["graphene-ideal/armchair/N7", "square/strip/N10"]
    assert CloudStore(tmp_path).models() == ["graphene-ideal/armchair/N7", "hbn/armchair/N9", "square/strip/N10"]
    assert view.read_pristine("graphene-ideal/armchair/N7")[1].size == GRID.size


def test_a_multistore_of_two_views_accepts_what_two_overlapping_roots_would_not(tmp_path):
    from atlaslib.store import MultiStore
    s = CloudStore(tmp_path)
    for mid in ("graphene-ideal/armchair/N7", "hbn/armchair/N9"):
        s.write_pristine(mid, GRID, np.ones(GRID.size))
    both = MultiStore(CloudStore(tmp_path, only=("graphene-ideal/",)), CloudStore(tmp_path, only=("hbn/",)))
    assert both.models() == ["graphene-ideal/armchair/N7", "hbn/armchair/N9"]
