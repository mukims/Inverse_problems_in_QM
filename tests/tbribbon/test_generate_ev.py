# tests/tbribbon/test_generate_ev.py
import numpy as np
import pytest

from atlaslib import CloudStore, InputSpec
from tbribbon.generate_clouds import generate
from tbribbon.materials import make_model


def test_clouds_land_on_the_ev_channels(tmp_path):
    spec = InputSpec(version="v3")
    m = make_model("triangular", "zigzag", 4)
    store = CloudStore(tmp_path / "s")
    generate(store, [m], [0.02], spec, n_jobs=1, formula="caroli", seeds=range(3))
    e_t, _ = store.read_pristine(m.model_id)
    assert np.allclose(e_t * m.t_ev, spec.energies_t())
    c, s = store.read_cloud(m.model_id, store.densities(m.model_id)[0])
    assert c.shape == (3, 416) and s.tolist() == [0, 1, 2]


def test_channels_above_the_band_top_are_exact_zeros(tmp_path):
    spec = InputSpec(version="v3")
    m = make_model("triangular", "zigzag", 4)
    store = CloudStore(tmp_path / "s")
    generate(store, [m], [0.02], spec, n_jobs=1, formula="caroli", seeds=range(2))
    e_t, pris = store.read_pristine(m.model_id)
    c, _ = store.read_cloud(m.model_id, store.densities(m.model_id)[0])
    above = e_t > m.band_top_t + 0.01
    assert above.any() and np.all(pris[above] == 0) and np.all(c[:, above] == 0)


def test_graphene_armchair_is_not_regenerated_on_the_ev_axis(tmp_path):
    with pytest.raises(ValueError, match="resampl"):
        generate(CloudStore(tmp_path / "s"), [make_model("graphene-ideal", "armchair", 7)], [0.01],
                 InputSpec(version="v3"), n_jobs=1, seeds=range(2))
