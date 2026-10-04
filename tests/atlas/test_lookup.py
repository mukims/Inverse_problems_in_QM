# tests/atlas/test_lookup.py
import numpy as np
import pytest

from atlaslib import InputSpec
from atlaslib.lookup import window_input, window_inputs
from toy import E, toy_spectrum

SPEC = InputSpec(version="v4")

# ---------- Task 1: window-aware transform ----------

def test_window_input_matches_spec_transform_on_covered_channels():
    T = toy_spectrum(1.0, 9, 0.02, 7)
    x, mask = window_input(SPEC, E, T)
    assert mask.sum() == 200 and mask[:200].all() and not mask[200:].any()
    assert np.allclose(x[mask], SPEC.to_input(T, E, 4.0)[0][mask], atol=1e-7)
    assert np.all(x[~mask] == 0)


def test_window_inputs_batch_equals_single():
    T = np.stack([toy_spectrum(1.0, 9, 0.02, s) for s in range(3)])
    X, _ = window_inputs(SPEC, E, T)
    for i in range(3):
        assert np.array_equal(X[i], window_input(SPEC, E, T[i])[0])


def test_energies_beyond_the_axis_are_ignored():
    e = np.round(np.arange(0, 12.0001, 0.01), 6)
    T = np.interp(e, E, toy_spectrum(1.0, 9, 0.02, 7), right=0.0)
    _, mask = window_input(SPEC, e, T)
    assert mask.all()


def test_a_single_spike_is_removed():
    T = toy_spectrum(1.0, 9, 0.02, 7)
    spiked = T.copy()
    spiked[150] = 400.0
    assert np.max(np.abs(window_input(SPEC, E, spiked)[0] - window_input(SPEC, E, T)[0])) < 0.05


@pytest.mark.parametrize("energies, T, message", [
    (E - 0.01, toy_spectrum(1.0, 9, 0.02, 7), "0 eV"),
    (E[::-1], toy_spectrum(1.0, 9, 0.02, 7), "increase"),
    (E, np.full(E.size, np.nan), "finite"),
    (E[:5], np.ones(5), "at least"),
    (E, np.ones(E.size - 1), "values per spectrum"),
])
def test_bad_signatures_are_rejected(energies, T, message):
    with pytest.raises(ValueError, match=message):
        window_input(SPEC, energies, T)


def test_a_units_of_t_spec_is_refused():
    with pytest.raises(ValueError, match="eV-axis"):
        window_input(InputSpec(version="v2"), E, toy_spectrum(1.0, 9, 0.02, 7))
