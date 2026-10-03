import numpy as np
from atlaslib import Atlas, InputSpec, Registry
try:
    from toy import E, toy_spectrum, toy_store
except ImportError:
    from tests.atlas.toy import E, toy_spectrum, toy_store

FAST = dict(latent=8, epochs=6, patience=3, k=7, refs_per_model=200, threads=2)


def test_add_models_keeps_old_answers_and_reports_familiarity(tmp_path):
    store, models = toy_store(tmp_path, materials=(("alpha", 1.0),))
    reg = Registry(models)
    atlas = Atlas.build(store, reg, reg.ids(), InputSpec(), **FAST)
    probe = toy_spectrum(1.0, 9, 0.01, 90_000)[None]
    before = atlas.locate(probe, E, 3.0)
    n_refs = len(atlas.refs)

    store2, new = toy_store(tmp_path / "b", materials=(("gamma", 6.0),), widths=(9,))
    for m in new:
        reg.add(m)
        e_t, p = store2.read_pristine(m.model_id); store.write_pristine(m.model_id, e_t, p)
        for d in store2.densities(m.model_id):
            c, s = store2.read_cloud(m.model_id, d); store.write_cloud(m.model_id, d, 1, c, s, e_t)
    report = atlas.add_models(store, reg, [m.model_id for m in new])

    assert set(report) == {"gamma/armchair/N9"} and 0.0 <= report["gamma/armchair/N9"] <= 1.0
    assert len(atlas.refs) > n_refs
    assert atlas.locate(probe, E, 3.0)[0].material == before[0].material


import pytest


def _gamma_into(store, reg, tmp_path):
    store2, new = toy_store(tmp_path / "g", materials=(("gamma", 6.0),), widths=(9,))
    for m in new:
        reg.add(m)
        e_t, p = store2.read_pristine(m.model_id); store.write_pristine(m.model_id, e_t, p)
        for d in store2.densities(m.model_id):
            c, s = store2.read_cloud(m.model_id, d); store.write_cloud(m.model_id, d, 1, c, s, e_t)
    return [m.model_id for m in new]


def test_add_models_takes_references_only_up_to_max_seed(tmp_path):
    store, models = toy_store(tmp_path, materials=(("alpha", 1.0),))
    reg = Registry(models)
    atlas = Atlas.build(store, reg, reg.ids(), InputSpec(), **FAST)
    new = _gamma_into(store, reg, tmp_path)
    n0 = len(atlas.refs)
    # toy seeds: d=0.01 -> 1000..1039, d=0.04 -> 4000..4039; max_seed=1019 keeps 20 cloud spectra + the pristine
    atlas.add_models(store, reg, new, max_seed=1019)
    assert len(atlas.refs) - n0 == 21


def test_add_models_caps_references_per_model(tmp_path):
    store, models = toy_store(tmp_path, materials=(("alpha", 1.0),))
    reg = Registry(models)
    atlas = Atlas.build(store, reg, reg.ids(), InputSpec(), **FAST)
    new = _gamma_into(store, reg, tmp_path)
    n0 = len(atlas.refs)
    atlas.add_models(store, reg, new, refs_per_model=30)
    assert len(atlas.refs) - n0 == 30


def test_without_models_hides_a_material_and_leaves_the_original_intact(tmp_path):
    store, models = toy_store(tmp_path)                       # alpha and beta, widths 7, 9, 14
    reg = Registry(models)
    atlas = Atlas.build(store, reg, reg.ids(), InputSpec(), **FAST)
    atlas.calibrate_novelty(store, reg, reg.ids())
    beta = [m.model_id for m in models if m.material == "beta"]
    n_refs, table = len(atlas.refs), dict(atlas.threshold_table)

    hidden = atlas.without_models(beta)
    assert {m.material for m in hidden.models} == {"alpha"}
    assert len(hidden.refs) < n_refs and hidden.ref_model.max() == len(hidden.models) - 1
    assert set(hidden.threshold_table) == {m.model_id for m in models if m.material == "alpha"}
    probe = toy_spectrum(3.0, 9, 0.01, 77_000)[None]          # a beta spectrum
    assert hidden.locate(probe, E, 3.0)[0].material == "alpha"
    assert len(atlas.refs) == n_refs and atlas.threshold_table == table   # original untouched


def test_without_models_rejects_unknown_ids(tmp_path):
    store, models = toy_store(tmp_path, materials=(("alpha", 1.0),))
    reg = Registry(models)
    atlas = Atlas.build(store, reg, reg.ids(), InputSpec(), **FAST)
    with pytest.raises(KeyError):
        atlas.without_models(["nope/armchair/N1"])


def test_with_reference_cap_keeps_at_most_n_per_model(tmp_path):
    store, models = toy_store(tmp_path)
    reg = Registry(models)
    atlas = Atlas.build(store, reg, reg.ids(), InputSpec(), **FAST)
    n_before = len(atlas.refs)
    small = atlas.with_reference_cap(10, seed=1)
    counts = np.bincount(small.ref_model, minlength=len(small.models))
    assert counts.max() <= 10 and counts.min() > 0
    assert len(atlas.refs) == n_before and [m.model_id for m in small.models] == [m.model_id for m in atlas.models]
    probe = toy_spectrum(1.0, 9, 0.01, 123)[None]
    assert small.locate(probe, E, 3.0)[0].material == "alpha"


