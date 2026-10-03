# tests/atlas/test_shape_profile.py
import numpy as np
import pytest
from meaning.distances import rms_distance, shape_profile


def _staircase(e, top, steps):
    T = np.floor(np.clip(e / top, 0, 1 - 1e-9) * steps) + 1.0
    T[e >= top] = 0.0
    return T


def test_same_shape_at_different_energy_scales_has_zero_distance():
    e = np.arange(0, 10, 0.005)
    a = shape_profile(_staircase(e, 3.0, 4), e)
    b = shape_profile(_staircase(e, 7.5, 4), e)          # same band shape, 2.5x the energy scale
    assert rms_distance(a, b) < 0.02


def test_different_shapes_are_apart():
    e = np.arange(0, 10, 0.005)
    assert rms_distance(shape_profile(_staircase(e, 3.0, 4), e), shape_profile(_staircase(e, 3.0, 9), e)) > 0.1


def test_profile_is_in_unit_range_and_has_requested_length():
    e = np.arange(0, 10, 0.01)
    p = shape_profile(_staircase(e, 5.0, 6), e, n=128)
    assert p.shape == (128,) and p.min() >= 0 and p.max() <= 1


def test_empty_spectrum_is_refused():
    e = np.arange(0, 1, 0.01)
    with pytest.raises(ValueError, match="no open channel"):
        shape_profile(np.zeros_like(e), e)
