import types

import numpy as np
import pytest

from atlaslib import InputSpec
from atlaslib.energy import axis_scale, generation_grid_t, on_axis
from atlaslib.registry import RibbonModel


def _m(t_ev):
    return RibbonModel("x", "armchair", 7, t_ev, 14, 3.0)


def test_units_of_t_specs_leave_energies_alone():
    e, top = on_axis(InputSpec(version="v2"), _m(2.7), np.array([0.0, 1.0]), 3.0)
    assert np.allclose(e, [0.0, 1.0]) and top == 3.0


def test_ev_spec_scales_by_hopping():
    e, top = on_axis(InputSpec(version="v3"), _m(2.7), np.array([0.0, 1.0]), 3.0)
    assert np.allclose(e, [0.0, 2.7]) and top == pytest.approx(8.1)
    assert on_axis(InputSpec(version="v3"), _m(2.7), np.array([0.0]), None)[1] is None


def test_generation_grid_lands_on_the_ev_channels():
    spec = InputSpec(version="v3")
    assert np.allclose(generation_grid_t(spec, _m(2.3)) * 2.3, spec.energies_t())


def test_non_positive_hopping_is_refused_on_ev_axis():
    bad = types.SimpleNamespace(t_ev=0.0, model_id="x/armchair/N7")
    with pytest.raises(ValueError, match="t_ev"):
        axis_scale(InputSpec(version="v3"), bad)


def test_graphene_hopping_is_physical():
    from tbribbon.materials import make_model
    assert make_model("graphene-ideal", "armchair", 7).t_ev == 2.7
