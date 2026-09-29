import numpy as np
import pytest
from atlaslib.store import CloudStore

MID = "graphene-ideal/armchair/N7"
E = np.arange(10) * 0.1


def test_round_trip_and_listing(tmp_path, rng):
    s = CloudStore(tmp_path)
    s.write_pristine(MID, E, np.ones(10) * 2)
    spectra = rng.random((5, 10))
    s.write_cloud(MID, 0.01, 14, spectra, np.arange(5), E)
    got, seeds = s.read_cloud(MID, 0.01)
    assert np.array_equal(got, spectra) and list(seeds) == [0, 1, 2, 3, 4]
    assert s.models() == [MID] and s.densities(MID) == [0.01] and s.has_cloud(MID, 0.01)


def test_energy_grid_mismatch_refused(tmp_path, rng):
    s = CloudStore(tmp_path)
    s.write_pristine(MID, E, np.ones(10))
    with pytest.raises(ValueError, match="energ"):
        s.write_cloud(MID, 0.01, 14, rng.random((3, 10)), np.arange(3), E * 2)


def test_non_finite_refused(tmp_path, rng):
    s = CloudStore(tmp_path)
    bad = rng.random((3, 10)); bad[1, 4] = np.nan
    with pytest.raises(ValueError, match="finite"):
        s.write_cloud(MID, 0.01, 14, bad, np.arange(3), E)


def test_identical_spectra_across_densities_refused(tmp_path, rng):
    """Guards against the square-lattice cache bug: same spectrum at two densities."""
    s = CloudStore(tmp_path)
    a = rng.random((4, 10))
    s.write_cloud(MID, 0.01, 14, a, np.arange(4), E)
    b = rng.random((4, 10)); b[2] = a[1]
    with pytest.raises(ValueError, match="identical"):
        s.write_cloud(MID, 0.02, 28, b, np.arange(4), E)


def test_duplicate_seeds_refused(tmp_path, rng):
    s = CloudStore(tmp_path)
    with pytest.raises(ValueError, match="seed"):
        s.write_cloud(MID, 0.01, 14, rng.random((3, 10)), np.array([0, 1, 1]), E)


def test_spike_fraction(tmp_path):
    s = CloudStore(tmp_path)
    s.write_pristine(MID, E, np.ones(10))
    c = np.ones((3, 10)) * 0.5; c[0, :3] = 5.0          # 3 of 30 points above pristine, median is 0.5
    s.write_cloud(MID, 0.01, 14, c, np.arange(3), E)
    assert s.spike_fraction(MID, 0.01) == pytest.approx(3 / 30)


def test_formula_mismatch_refused(tmp_path, rng):
    s = CloudStore(tmp_path)
    s.write_pristine(MID, E, np.ones(10), formula="agnr_lib")
    with pytest.raises(ValueError, match="formula"):
        s.write_cloud(MID, 0.01, 14, rng.random((2, 10)), np.arange(2), E, formula="caroli")


def test_unphysical_excess_cloud_refused(tmp_path):
    s = CloudStore(tmp_path)
    s.write_pristine(MID, E, np.ones(10))
    bad = np.ones((3, 10)) * 1.5
    with pytest.raises(ValueError, match="unphysical"):
        s.write_cloud(MID, 0.01, 14, bad, np.arange(3), E)

