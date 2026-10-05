# tests/tbribbon/test_lead_eta.py
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from atlaslib import CloudStore, InputSpec
from tbribbon.check_store import check_model
from tbribbon.generate_clouds import DEFAULT_LEAD_ETA, generate
from tbribbon.materials import make_model

REPO = Path(__file__).resolve().parents[2]
SMOKE4 = Path("~/atlas_store/materials2_ev_smoke").expanduser()
V3 = InputSpec(version="v3")


def _meta(store, m):
    return json.loads((store._dir(m.model_id) / "meta.json").read_text())


def test_lead_eta_reaches_the_leads_and_is_recorded(tmp_path):
    # SMOKE-4: wse2/zigzag/N14 read 1.9955 for two open channels at 0.18 eV with the default eta
    m = make_model("wse2", "zigzag", 14)
    wide, narrow = CloudStore(tmp_path / "wide"), CloudStore(tmp_path / "narrow")
    generate(wide, [m], [], V3, n_jobs=1, formula="caroli")
    generate(narrow, [m], [], V3, n_jobs=1, formula="caroli", lead_eta=1e-5)
    assert _meta(wide, m)["pristine_lead_eta"] == DEFAULT_LEAD_ETA == 1e-4
    assert _meta(narrow, m)["pristine_lead_eta"] == 1e-5
    assert check_model(wide, m, None)["clean_channels_max_err"] > 3e-3
    assert check_model(narrow, m, None)["clean_channels_max_err"] < 1e-4


@pytest.mark.skipif(not (SMOKE4 / "mose2/zigzag/N7/pristine.npy").exists(), reason="SMOKE-4 store not on this machine")
def test_default_reproduces_the_smoke4_pristine(tmp_path):
    m = make_model("mose2", "zigzag", 7)
    store = CloudStore(tmp_path / "s")
    generate(store, [m], [], V3, n_jobs=1, formula="caroli")
    assert np.allclose(store.read_pristine(m.model_id)[1], np.load(SMOKE4 / "mose2/zigzag/N7/pristine.npy"),
                       rtol=0, atol=1e-12)


def test_resume_refuses_a_different_lead_eta(tmp_path):
    m = make_model("triangular", "zigzag", 4)
    store = CloudStore(tmp_path / "s")
    generate(store, [m], [], V3, n_jobs=1, formula="caroli")
    with pytest.raises(ValueError, match=r"lead_eta=0\.0001; refusing"):
        generate(store, [m], [0.02], V3, n_jobs=1, formula="caroli", seeds=range(2), lead_eta=1e-5)
    assert store.densities(m.model_id) == []


def test_store_without_lead_eta_counts_as_the_default(tmp_path):
    m = make_model("triangular", "zigzag", 4)
    store = CloudStore(tmp_path / "s")
    generate(store, [m], [], V3, n_jobs=1, formula="caroli")
    meta = _meta(store, m)
    del meta["pristine_lead_eta"]
    (store._dir(m.model_id) / "meta.json").write_text(json.dumps(meta))
    generate(store, [m], [0.02], V3, n_jobs=1, formula="caroli", seeds=range(2))
    assert [c["lead_eta"] for c in _meta(store, m)["clouds"].values()] == [1e-4]
    with pytest.raises(ValueError, match="refusing"):
        generate(store, [m], [0.04], V3, n_jobs=1, formula="caroli", seeds=range(2), lead_eta=1e-5)


def test_graphene_armchair_rejects_a_lead_eta(tmp_path):
    with pytest.raises(ValueError, match="agnr_lib builds its own leads"):
        generate(CloudStore(tmp_path / "s"), [make_model("graphene-ideal", "armchair", 7)], [0.01],
                 InputSpec(version="v2"), n_jobs=1, seeds=range(2), lead_eta=1e-5)


def test_cli_flag_reaches_the_store(tmp_path):
    env = dict(os.environ, OMP_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join([str(REPO / "notebooks" / "material_atlas"), str(REPO / "notebooks")]))
    subprocess.run([sys.executable, str(REPO / "notebooks" / "tbribbon" / "generate_clouds.py"),
                    "--store", str(tmp_path / "s"), "--grid", "models", "--models", "triangular/zigzag/N4",
                    "--densities", "0.02", "--spec-version", "v3", "--formula", "caroli", "--n-seeds", "2",
                    "--n-jobs", "1", "--lead-eta", "1e-5"], check=True, env=env, cwd=REPO, capture_output=True)
    meta = _meta(CloudStore(tmp_path / "s"), make_model("triangular", "zigzag", 4))
    assert meta["pristine_lead_eta"] == 1e-5
    assert [c["lead_eta"] for c in meta["clouds"].values()] == [1e-5]
