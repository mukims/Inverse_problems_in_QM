import numpy as np
import pytest
from atlaslib.spec import InputSpec


def test_grid_is_400_channels_from_zero():
    s = InputSpec()
    assert s.n_channels == 400
    e = s.energies_t()
    assert e[0] == 0.0 and abs(e[-1] - 3.99) < 1e-12


def test_zero_fill_above_band_top_is_exact():
    s = InputSpec()
    e_t = np.arange(300) * 0.01                     # legacy AGNR grid, 0-2.99
    T = np.full((2, 300), 2.0)
    X = s.to_input(T, e_t, band_top_t=3.0)
    assert X.shape == (2, 400)
    assert np.all(X[:, 300:] == 0.0)
    assert np.allclose(X[:, :300], np.log1p(2.0) / np.log1p(20.0))


def test_refuses_to_zero_fill_real_band():
    s = InputSpec()
    e_t = np.arange(300) * 0.01
    with pytest.raises(ValueError, match="band"):
        s.to_input(np.ones((1, 300)), e_t, band_top_t=3.9)


def test_default_requires_full_window():
    s = InputSpec()
    with pytest.raises(ValueError):
        s.to_input(np.ones((1, 300)), np.arange(300) * 0.01)   # no band top given -> must cover 0-3.99


def test_rejects_grid_not_starting_at_zero_or_mismatched():
    s = InputSpec()
    with pytest.raises(ValueError):
        s.to_input(np.ones((1, 400)), np.arange(1, 401) * 0.01, band_top_t=4.0)
    with pytest.raises(ValueError):
        s.to_input(np.ones((1, 399)), np.arange(400) * 0.01, band_top_t=4.0)


def test_cap_and_rounding():
    s = InputSpec()
    e_t = np.arange(400) * 0.01
    T = np.zeros((1, 400)); T[0, 0] = 100.0; T[0, 1] = 0.0004
    X = s.to_input(T, e_t, band_top_t=4.0)
    assert X[0, 0] == pytest.approx(1.0) and X[0, 1] == 0.0


def test_other_grid_is_interpolated():
    s = InputSpec()
    e_t = np.linspace(0, 4.0, 81)                    # e.g. eV data converted with t = 2.7
    X = s.to_input(np.tile(e_t, (1, 1)), e_t, band_top_t=4.0)
    assert X[0, 100] == pytest.approx(np.log1p(1.0) / np.log1p(20.0), abs=1e-3)


def test_inputspec_v2_cap_is_64():
    s2 = InputSpec(version="v2")
    assert s2.version == "v2"
    assert s2.cap == 64.0
    assert s2.as_dict()["cap"] == 64.0


def test_no_clean_spectrum_in_grid_reaches_cap():
    from atlaslib.store import CloudStore
    store = CloudStore("~/atlas_store/smoke_v1")
    s2 = InputSpec(version="v2")
    for mid in store.models():
        _, pris = store.read_pristine(mid)
        assert np.max(pris) < s2.cap, f"{mid} pristine reaches or exceeds cap {s2.cap}"

