"""Test for the 7/9-AGNR reference pipeline and Gate 1 benchmark artifacts."""
import json
from pathlib import Path
import numpy as np
from atlaslib import Atlas

REPO = Path(__file__).resolve().parents[2]
METRICS_PATH = REPO / "notebooks/material_atlas/reference_7_9/metrics.json"
ATLAS_PATH = REPO / "notebooks/material_atlas/reference_7_9/atlas"


def test_metrics_json_exists_and_satisfies_gate1():
    assert METRICS_PATH.exists(), f"{METRICS_PATH} does not exist"
    data = json.loads(METRICS_PATH.read_text())

    required_keys = {
        "width_accuracy",
        "end_to_end_mae",
        "mae_by_width",
        "coverage_90",
        "interval_relative_halfwidth",
        "n_test",
        "criteria_met",
    }
    assert required_keys.issubset(data.keys()), f"Missing keys: {required_keys - set(data.keys())}"

    assert data["width_accuracy"] >= 99.5, f"Width accuracy {data['width_accuracy']} < 99.5"
    assert data["end_to_end_mae"] <= 1.98, f"End-to-end MAE {data['end_to_end_mae']} > 1.98"
    assert abs(data["coverage_90"] - 90.0) <= 2.0, f"Coverage {data['coverage_90']} outside 90 +- 2%"
    assert data["n_test"] == 37350, f"Unexpected test size: {data['n_test']}"

    crit = data["criteria_met"]
    assert crit.get("width_accuracy>=99.5") is True
    assert crit.get("mae<=1.98") is True
    assert crit.get("coverage_90+-2") is True


def test_saved_reference_atlas_loads_and_locates():
    assert ATLAS_PATH.exists(), f"{ATLAS_PATH} does not exist"
    atlas = Atlas.load(ATLAS_PATH)
    assert len(atlas.models) == 2
    assert {m.width for m in atlas.models} == {7, 9}

    # Locate synthetic test spectra
    e_t = np.arange(300) * 0.01
    syn = np.zeros((2, 300))
    syn[0, 24:] = 1.0
    syn[1, 18:] = 2.0
    loc = atlas.locate(syn, e_t, band_top_t=3.0)
    assert len(loc) == 2
    assert all(hasattr(r, "width") for r in loc)
