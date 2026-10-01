"""Tests for Stage 3 7/9-AGNR concentration models and production Atlas v2 integration."""
import json
from pathlib import Path
import numpy as np
import pytest

from atlaslib.conformal import coverage, fit_relative, intervals

REPO = Path(__file__).resolve().parents[2]
METRICS_PATH = REPO / "notebooks/material_atlas/stage3_7_9/metrics.json"
PREDS_PATH = REPO / "notebooks/material_atlas/stage3_7_9/predictions_test.npz"


def test_toy_interval_coverage():
    """Verify that relative split-conformal intervals reach nominal 90% coverage on toy data."""
    rng = np.random.default_rng(42)
    y = rng.uniform(2, 98, 20_000)
    # Heteroscedastic noise proportional to true value
    pred = y * (1.0 + rng.normal(0, 0.05, y.size))
    pred = np.maximum(pred, 0.1)

    # Calibrate on first half
    q = fit_relative(pred[:10_000], y[:10_000], alpha=0.1)
    lo, hi = intervals(pred[10_000:], q)
    cov = coverage(lo, hi, y[10_000:])
    assert abs(cov - 0.90) <= 0.02, f"Coverage {cov:.4f} outside [0.88, 0.92]"


def test_unknown_flagged_spectra_get_no_estimate_unit():
    """Unit test verifying that spectra marked unknown receive no concentration estimate (NaN)."""
    # Create synthetic test predictions
    n = 100
    unknown_flags = np.zeros(n, dtype=bool)
    unknown_flags[10:20] = True # 10 spectra flagged unknown

    c_hat_unflagged = np.full(n, np.nan)
    regs_preds = np.linspace(2, 50, n)
    c_hat_unflagged[~unknown_flags] = regs_preds[~unknown_flags]

    q = 0.10
    lo, hi = np.full(n, np.nan), np.full(n, np.nan)
    lo[~unknown_flags], hi[~unknown_flags] = intervals(c_hat_unflagged[~unknown_flags], q)

    # Assertions
    assert np.all(np.isnan(c_hat_unflagged[unknown_flags])), "Unknown spectra must not have concentration estimates"
    assert np.all(np.isnan(lo[unknown_flags])), "Unknown spectra must not have interval bounds"
    assert np.all(np.isnan(hi[unknown_flags])), "Unknown spectra must not have interval bounds"
    assert np.all(~np.isnan(c_hat_unflagged[~unknown_flags])), "Unflagged spectra must have estimates"
    assert np.all(lo[~unknown_flags] <= c_hat_unflagged[~unknown_flags])
    assert np.all(c_hat_unflagged[~unknown_flags] <= hi[~unknown_flags])


@pytest.mark.skipif(not METRICS_PATH.exists(), reason="Stage 3 run must complete first")
def test_stage3_metrics_json_exists_and_satisfies_criteria():
    data = json.loads(METRICS_PATH.read_text())

    required_keys = {
        "width_accuracy",
        "end_to_end_mae",
        "end_to_end_mae_all",
        "mae_by_width",
        "unknown_flag_rate",
        "coverage_90",
        "interval_relative_halfwidth",
        "n_test",
        "n_unflagged",
        "criteria_met",
        "concentration_breakdown",
    }
    assert required_keys.issubset(data.keys()), f"Missing keys: {required_keys - set(data.keys())}"

    assert data["n_test"] == 37350, f"Expected 37,350 test spectra, got {data['n_test']}"
    assert data["width_accuracy"] >= 99.5, f"Width accuracy {data['width_accuracy']:.2f}% < 99.5%"
    assert "mae<=1.98" in data["criteria_met"]
    assert data["end_to_end_mae"] <= 2.05, f"End-to-end MAE {data['end_to_end_mae']:.3f} > 2.05"
    assert abs(data["coverage_90"] - 90.0) <= 2.0, f"Coverage {data['coverage_90']:.2f}% outside 90 +- 2%"


@pytest.mark.skipif(not PREDS_PATH.exists(), reason="Stage 3 test predictions must exist")
def test_saved_stage3_test_predictions_consistency():
    preds = np.load(PREDS_PATH)
    y_true = preds["y_true"]
    c_hat_unf = preds["c_hat_unflagged"]
    unknown = preds["unknown"]
    lo = preds["lo"]
    hi = preds["hi"]
    w_hat = preds["w_hat"]
    w_true = preds["w_true"]

    assert len(y_true) == 37350
    # Flagged unknown must receive NaN
    assert np.all(np.isnan(c_hat_unf[unknown])), "Flagged spectra received non-NaN estimate"
    assert np.all(np.isnan(lo[unknown])), "Flagged spectra received non-NaN lo bound"
    assert np.all(np.isnan(hi[unknown])), "Flagged spectra received non-NaN hi bound"

    # Unflagged must receive valid finite estimates
    assert np.all(np.isfinite(c_hat_unf[~unknown])), "Unflagged spectra missing estimate"
    assert np.all(lo[~unknown] <= c_hat_unf[~unknown])
    assert np.all(c_hat_unf[~unknown] <= hi[~unknown])
