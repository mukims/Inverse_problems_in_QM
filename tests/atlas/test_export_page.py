# tests/atlas/test_export_page.py
import json

import numpy as np
from atlaslib import Atlas, InputSpec, Registry
from atlas_page.export_page import ENCODER_ORDER, export_page
try:
    from toy import toy_store
except ImportError:
    from tests.atlas.toy import toy_store

FAST = dict(latent=8, epochs=4, patience=2, k=7, refs_per_model=200, threads=2)


def test_export_writes_consistent_files(tmp_path):
    store, models = toy_store(tmp_path, n_seeds=40)
    reg = Registry(models)
    atlas = Atlas.build(store, reg, reg.ids(), InputSpec(), **FAST)
    out = tmp_path / "page"
    export_page(atlas, store, reg, reg.ids(), out, refs_per_model=20, n_test=12, val=(1030, 1039), test_seed_min=0)
    meta = json.loads((out / "model.json").read_text())
    n = meta["n_refs"]
    refs = np.fromfile(out / "refs.bin", dtype="<f4").reshape(n, -1)
    assert refs.shape[1] == len(meta["mu"]) == 8
    assert np.fromfile(out / "ref_model.bin", dtype="<u2").size == n == np.fromfile(out / "ref_density.bin", dtype="<f4").size
    sizes = {name: atlas.encoder.state_dict()[name].numel() for name in ENCODER_ORDER}
    assert np.fromfile(out / "encoder.bin", dtype="<f4").size == sum(sizes.values())
    tv = json.loads((out / "test_vectors.json").read_text())
    assert len(tv) == 12 and {"material", "edge", "width_vote", "unknown", "nearest_model"} <= set(tv[0]["answer"])
    assert len(meta["pca"]["components"]) == 2 and set(meta["clean"]) == set(reg.ids())


import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[2] / "notebooks/material_atlas/atlas_page"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_javascript_matches_python_on_toy_export(tmp_path):
    store, models = toy_store(tmp_path, n_seeds=40)
    reg = Registry(models)
    atlas = Atlas.build(store, reg, reg.ids(), InputSpec(), **FAST)
    out = tmp_path / "page"
    export_page(atlas, store, reg, reg.ids(), out, refs_per_model=20, n_test=40, val=(1030, 1039), test_seed_min=0)
    run = subprocess.run(["node", str(PAGE / "test_shazam.mjs"), str(out)], capture_output=True, text=True, cwd=PAGE)
    assert run.returncode == 0 and "PASS" in run.stdout, run.stdout + run.stderr

