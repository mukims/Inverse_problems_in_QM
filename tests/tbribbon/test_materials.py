import numpy as np
import pytest
from tbribbon.materials import hamiltonian_for, make_model


def test_make_model_fills_geometry_and_band_top():
    m = make_model("graphene-ideal", "armchair", 9)
    assert m.model_id == "graphene-ideal/armchair/N9" and m.sites_per_cell == 18
    assert m.band_top_t == pytest.approx(1 + 2 * np.cos(np.pi / 10), abs=1e-3)
    assert hamiltonian_for(m).H0.shape == (18, 18)


def test_square_uses_strip_edge():
    m = make_model("square", "strip", 10)
    assert m.sites_per_cell == 10 and m.band_top_t < 4.0


def test_unknown_material_rejected():
    with pytest.raises(KeyError):
        make_model("unobtainium", "armchair", 7)
