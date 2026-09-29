import json
from pathlib import Path
import numpy as np
import pytest
from atlaslib import CloudStore, InputSpec, Registry, RibbonModel
from notebooks.material_atlas.build_atlas_v2 import HELD_OUT, TRAIN


def test_held_out_partition():
    for key, widths in HELD_OUT.items():
        train_pool = list(TRAIN[key])
        for w in widths:
            assert w in train_pool, f"Held-out width {w} not in train set for {key}"
        remaining = [w for w in train_pool if w not in widths]
        assert len(remaining) > 0, f"No training widths remaining for {key}"


def test_toy_generalisation_interpolation(tmp_path):
    from atlaslib import Atlas, InputSpec, Registry
    from tests.atlas.toy import toy_store

    store, models = toy_store(tmp_path, materials=(("alpha", 1.0),), widths=(6, 8, 10), n_seeds=30)
    reg = Registry(models)
    train_ids = ["alpha/armchair/N6", "alpha/armchair/N10"]
    atlas = Atlas.build(store, reg, train_ids, InputSpec(), latent=8, epochs=10, patience=3, k=5, threads=2)

    c, s = store.read_cloud("alpha/armchair/N8", 0.01)
    e_t, _ = store.read_pristine("alpha/armchair/N8")
    loc = atlas.locate(c, e_t, reg.get("alpha/armchair/N8").band_top_t)
    widths = np.array([r.width for r in loc])
    assert all(r.edge == "armchair" for r in loc)
    assert 6.0 <= np.median(widths) <= 10.0

