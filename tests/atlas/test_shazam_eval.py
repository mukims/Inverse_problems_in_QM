# tests/atlas/test_shazam_eval.py
import numpy as np
from atlaslib import Atlas, InputSpec, Registry
from shazam_eval import clean_ground_truth, identify, query_group
try:
    from toy import toy_store
except ImportError:
    from tests.atlas.toy import toy_store

FAST = dict(latent=8, epochs=6, patience=3, k=7, refs_per_model=200, threads=2)


def _atlas(tmp_path):
    store, models = toy_store(tmp_path, materials=(("alpha", 1.0), ("beta", 3.0), ("gamma", 6.0)))
    reg = Registry(models)
    atlas = Atlas.build(store, reg, reg.ids(), InputSpec(), **FAST)
    atlas.calibrate_novelty(store, reg, reg.ids(), group_by="material_edge")
    return store, reg, atlas


def test_identify_reports_per_model_and_per_material(tmp_path):
    store, reg, atlas = _atlas(tmp_path)
    res = identify(atlas, store, reg, reg.ids(), test_seed_min=0)
    assert set(res["per_model"]) == set(reg.ids())
    assert set(res["per_material"]) == {"alpha", "beta", "gamma"}
    assert res["per_material"]["alpha"]["material_accuracy"] > 90


def test_query_group_on_a_hidden_material_names_the_remaining_ones(tmp_path):
    store, reg, atlas = _atlas(tmp_path)
    gamma = [i for i in reg.ids() if i.startswith("gamma/")]
    hidden = atlas.without_models(gamma)
    q = query_group(hidden, store, reg, gamma, test_seed_min=0)
    assert set(q["nearest_material"]) <= {"alpha", "beta"}
    assert abs(sum(q["nearest_material"].values()) - 100.0) < 1e-6
    assert q["n"] > 0 and len(q["z_quantiles"]) == 3


def test_clean_ground_truth_finds_the_other_material_nearest(tmp_path):
    store, reg, atlas = _atlas(tmp_path)
    gt = clean_ground_truth(store, reg, InputSpec(), reg.ids())
    g = gt["gamma/armchair/N9"]
    assert g["nearest_eV"]["material"] != "gamma" and g["nearest_shape"]["material"] != "gamma"
    assert g["nearest_eV"]["distance"] >= 0
