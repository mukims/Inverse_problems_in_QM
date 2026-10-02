import numpy as np
import pytest
from atlaslib.spec import InputSpec, despike


def test_despike_removes_one_and_two_channel_spikes_and_keeps_their_tail():
    T = np.zeros(20)
    T[8], T[9] = 461.73, 0.147          # legacy 7-AGNR spike just above the band top, with its decaying tail
    T[14:16] = 30.0
    out = despike(T[None])[0]
    assert out[8] == 0.0 and out[14] == 0.0 and out[15] == 0.0
    assert out[9] == pytest.approx(0.147)


@pytest.mark.parametrize("row", [
    [6, 6, 6, 8, 8, 6, 6, 6],               # triangular armchair N10: two modes open for two channels
    [8, 8, 8.267, 9.998, 6, 6.005, 6, 6],   # MoS2 armchair N14
    [5, 5, 5.002, 6.572, 3, 3, 3, 3],       # MoS2 armchair N9
    [0, 0, 1, 1, 1, 3, 3, 3],               # subband staircase
])
def test_despike_keeps_real_narrow_features(row):
    T = np.array(row, dtype=float)[None]
    assert np.array_equal(despike(T), T)


def test_v4_is_the_v3_axis_with_despiking():
    s3, s4 = InputSpec(version="v3"), InputSpec(version="v4")
    assert (s4.unit, s4.e_max_t, s4.step_t, s4.cap, s4.n_channels) == (s3.unit, s3.e_max_t, s3.step_t, s3.cap, 416)
    assert s4.despike and not s3.despike
    assert InputSpec(**s4.as_dict()) == s4


def test_v3_manifest_without_despike_key_loads_unchanged():
    d = InputSpec(version="v3").as_dict()
    d.pop("despike")
    assert InputSpec(**d).despike is False


def test_only_v4_despikes_the_input():
    e = np.arange(416) * 0.02
    T = np.zeros((1, 416))
    T[0, :100] = 2.0
    T[0, 50] = 449.0
    x3 = InputSpec(version="v3").to_input(T, e)
    x4 = InputSpec(version="v4").to_input(T, e)
    assert x3[0, 50] == pytest.approx(1.0)              # v3 unchanged: capped at 64, log-scaled
    assert x4[0, 50] == pytest.approx(x4[0, 49])        # v4: replaced by its neighbourhood median (2.0)
    assert np.array_equal(np.delete(x3, 50, 1), np.delete(x4, 50, 1))


@pytest.mark.parametrize("root", ["~/atlas_store/materials_v1", "~/atlas_store/engine_v1"])
def test_despike_leaves_every_clean_spectrum_unchanged(root):
    from atlaslib.store import CloudStore
    store = CloudStore(root)
    if not store.models():
        pytest.skip(f"{root} not present")
    for mid in store.models():
        _, pris = store.read_pristine(mid)
        p = np.round(pris, 3)[None]
        assert np.array_equal(despike(p), p), mid


def test_despike_leaves_caroli_disorder_clouds_unchanged():
    from atlaslib.store import CloudStore
    store = CloudStore("~/atlas_store/engine_v1")
    caroli = [m for m in store.models() if store._meta(m).get("pristine_formula") == "caroli"]
    if not caroli:
        pytest.skip("no Caroli models in engine_v1")
    for mid in caroli:
        for d in store.densities(mid):
            c = np.round(store.read_cloud(mid, d)[0][:100], 3)
            assert np.array_equal(despike(c), c), (mid, d)
