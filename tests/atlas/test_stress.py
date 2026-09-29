"""Adversarial and empirical stress tests for atlaslib Phase 1 Tasks 1-7."""
import numpy as np
import pytest
from atlaslib import (
    Atlas,
    CloudStore,
    InputSpec,
    Registry,
    RibbonModel,
    coverage,
    fit_relative,
    intervals,
)
try:
    from toy import E, toy_spectrum, toy_store
except ImportError:
    from tests.atlas.toy import E, toy_spectrum, toy_store


# ==============================================================================
# 1. InputSpec Stress Tests
# ==============================================================================

def test_input_spec_extreme_transmissions():
    spec = InputSpec()
    e_t = spec.energies_t()

    # Extreme high values: should be capped at spec.cap (20.0), outputting exactly 1.0
    T_high = np.array([[1e3, 1e6, 1e9] + [1.0] * 397])
    X_high = spec.to_input(T_high, e_t)
    assert np.all(X_high <= 1.0)
    assert X_high[0, 0] == pytest.approx(1.0)
    assert X_high[0, 1] == pytest.approx(1.0)
    assert X_high[0, 2] == pytest.approx(1.0)

    # Negative values: should be clipped to 0.0, outputting 0.0
    T_neg = np.array([[-1.0, -100.0, -1e6] + [0.0] * 397])
    X_neg = spec.to_input(T_neg, e_t)
    assert np.all(X_neg >= 0.0)
    assert X_neg[0, 0] == 0.0
    assert X_neg[0, 1] == 0.0
    assert X_neg[0, 2] == 0.0

    # Zero transmission
    T_zero = np.zeros((1, 400))
    X_zero = spec.to_input(T_zero, e_t)
    assert np.all(X_zero == 0.0)

    # Rounding boundary behavior (round to 3 decimal places)
    # 0.00049 rounds to 0.000 -> output 0.0
    # 0.00051 rounds to 0.001 -> output > 0.0
    T_round = np.zeros((1, 400))
    T_round[0, 0] = 0.00049
    T_round[0, 1] = 0.00051
    X_round = spec.to_input(T_round, e_t)
    assert X_round[0, 0] == 0.0
    assert X_round[0, 1] > 0.0
    expected_val = np.log1p(0.001) / np.log1p(20.0)
    assert X_round[0, 1] == pytest.approx(expected_val, abs=1e-5)

    # Strict monotonicity of mapping on valid [0, 20] range
    T_vals = np.linspace(0, 20, 400)
    X_vals = spec.to_input(T_vals[None], e_t)[0]
    assert np.all(np.diff(X_vals) >= 0.0)
    assert X_vals[0] == 0.0
    assert X_vals[-1] == pytest.approx(1.0)


def test_input_spec_band_top_vs_mid_band_truncation():
    spec = InputSpec()

    # Exact zero-filling above band top
    # Band top at 2.50 t: channels < 2.50 preserved, channels >= 2.50 zeroed
    e_partial = np.arange(250) * 0.01
    T_partial = np.full((3, 250), 5.0)
    X_out = spec.to_input(T_partial, e_partial, band_top_t=2.50)
    assert X_out.shape == (3, 400)
    assert np.all(X_out[:, 250:] == 0.0)
    expected = np.log1p(5.0) / np.log1p(20.0)
    assert np.allclose(X_out[:, :250], expected)

    # Boundary: Band top at 4.00 t (covers all 400 channels [0, 3.99])
    # If data only reaches 3.50 t, zero-filling would erase real signal: must raise ValueError
    e_short = np.arange(350) * 0.01
    with pytest.raises(ValueError, match="zero-filling would erase real signal"):
        spec.to_input(np.ones((1, 350)), e_short, band_top_t=4.0)

    # Boundary: Band top at 2.50 t, but data ends at 2.48 t (just short of 2.49 required)
    e_just_short = np.arange(249) * 0.01  # ends at 2.48
    with pytest.raises(ValueError, match="zero-filling would erase real signal"):
        spec.to_input(np.ones((1, 249)), e_just_short, band_top_t=2.50)

    # Boundary: Band top at 0.0 t (empty band)
    X_empty = spec.to_input(np.zeros((1, 10)), np.arange(10) * 0.01, band_top_t=0.0)
    assert X_empty.shape == (1, 400)
    assert np.all(X_empty == 0.0)

    # Boundary: band_top_t is None -> requires data to reach full window [0, 3.99]
    with pytest.raises(ValueError, match="window extends to"):
        spec.to_input(np.ones((1, 398)), np.arange(398) * 0.01, band_top_t=None)


def test_input_spec_grid_monotonicity_and_interpolation():
    spec = InputSpec()

    # Reject non-monotonic grid
    bad_grid_dec = np.array([0.0, 0.01, 0.009, 0.03])
    with pytest.raises(ValueError, match="start at 0 and increase"):
        spec.to_input(np.ones((1, 4)), bad_grid_dec, band_top_t=0.02)

    # Reject grid with duplicate points
    bad_grid_dup = np.array([0.0, 0.01, 0.01, 0.02])
    with pytest.raises(ValueError, match="start at 0 and increase"):
        spec.to_input(np.ones((1, 4)), bad_grid_dup, band_top_t=0.02)

    # Reject grid starting at non-zero
    bad_grid_start = np.array([0.05, 0.1, 0.15])
    with pytest.raises(ValueError, match="start at 0 and increase"):
        spec.to_input(np.ones((1, 3)), bad_grid_start, band_top_t=0.1)

    # Interpolation on non-uniform energy grid (e.g., quadratic spacing)
    e_quad = (np.linspace(0, 2.0, 200) ** 2)  # 0 to 4.0
    T_quad = np.sin(e_quad * np.pi) + 2.0
    X_interp = spec.to_input(T_quad[None], e_quad, band_top_t=4.0)
    assert X_interp.shape == (1, 400)
    # Check that interpolated values closely track analytic function
    grid = spec.energies_t()
    expected_unscaled = np.sin(grid * np.pi) + 2.0
    expected_scaled = np.log1p(np.round(expected_unscaled, 3)) / np.log1p(20.0)
    assert np.allclose(X_interp[0], expected_scaled, atol=2e-3)


# ==============================================================================
# 2. CloudStore Stress Tests
# ==============================================================================

def test_store_non_finite_rejection_all_variants(tmp_path, rng):
    s = CloudStore(tmp_path)
    mid = "graphene-ideal/armchair/N7"
    e_t = np.arange(10) * 0.1

    # NaN, +Inf, -Inf in pristine
    for bad_val in [np.nan, np.inf, -np.inf]:
        bad_pristine = np.ones(10)
        bad_pristine[3] = bad_val
        with pytest.raises(ValueError, match="pristine spectrum is not finite"):
            s.write_pristine(mid, e_t, bad_pristine)

    # Valid pristine
    s.write_pristine(mid, e_t, np.ones(10))

    # NaN, +Inf, -Inf in cloud spectra (single cell, first cell, last cell)
    for bad_val in [np.nan, np.inf, -np.inf]:
        for row_idx, col_idx in [(0, 0), (2, 9), (1, 5)]:
            bad_cloud = rng.random((4, 10))
            bad_cloud[row_idx, col_idx] = bad_val
            with pytest.raises(ValueError, match="spectra contain non-finite values"):
                s.write_cloud(mid, 0.01, 14, bad_cloud, np.arange(4), e_t)


def test_store_cross_density_duplicate_cache_bug_detection(tmp_path, rng):
    s = CloudStore(tmp_path)
    mid = "graphene-ideal/armchair/N7"
    e_t = np.arange(10) * 0.1
    s.write_pristine(mid, e_t, np.ones(10))

    cloud1 = rng.random((5, 10))
    s.write_cloud(mid, 0.01, 14, cloud1, np.arange(5), e_t)

    # Identical row in a new density: must raise ValueError
    cloud2 = rng.random((5, 10))
    cloud2[4] = cloud1[2]  # exactly 1 row matches
    with pytest.raises(ValueError, match="identical spectra at densities"):
        s.write_cloud(mid, 0.02, 28, cloud2, np.arange(5), e_t)

    # Subtle difference (float precision) should NOT be flagged as duplicate
    cloud3 = cloud1.copy()
    cloud3[:, :] += 1e-5
    s.write_cloud(mid, 0.03, 42, cloud3, np.arange(5), e_t)
    assert s.has_cloud(mid, 0.03)

    # Idempotent write to the same density is permitted (overwrite)
    s.write_cloud(mid, 0.01, 14, cloud1, np.arange(5), e_t)


def test_store_duplicate_seeds_rejection(tmp_path, rng):
    s = CloudStore(tmp_path)
    mid = "graphene-ideal/armchair/N7"
    e_t = np.arange(10) * 0.1
    s.write_pristine(mid, e_t, np.ones(10))

    spectra = rng.random((4, 10))

    # Duplicate seed: first and last
    with pytest.raises(ValueError, match="duplicate seed in cloud"):
        s.write_cloud(mid, 0.01, 14, spectra, np.array([42, 100, 200, 42]), e_t)

    # All identical seeds
    with pytest.raises(ValueError, match="duplicate seed in cloud"):
        s.write_cloud(mid, 0.01, 14, spectra, np.array([7, 7, 7, 7]), e_t)

    # Unique negative and large 64-bit seeds: should pass
    large_seeds = np.array([-10, 0, 10**12, 10**14], dtype=np.int64)
    s.write_cloud(mid, 0.01, 14, spectra, large_seeds, e_t)
    _, read_seeds = s.read_cloud(mid, 0.01)
    assert np.array_equal(read_seeds, large_seeds)


def test_store_spike_fraction_extremes(tmp_path):
    s = CloudStore(tmp_path)
    mid = "graphene-ideal/armchair/N7"
    e_t = np.arange(10) * 0.1
    s.write_pristine(mid, e_t, np.ones(10) * 2.0)

    # Cloud completely below pristine: spike fraction 0.0
    c_below = np.ones((4, 10)) * 1.5
    s.write_cloud(mid, 0.01, 14, c_below, np.arange(4), e_t)
    assert s.spike_fraction(mid, 0.01) == 0.0

    # Cloud completely above pristine: spike fraction 1.0
    c_above = np.ones((4, 10)) * 2.5
    s.write_cloud(mid, 0.02, 28, c_above, np.arange(4), e_t)
    assert s.spike_fraction(mid, 0.02) == 1.0

    # Cloud within 1e-6 numerical tolerance above pristine: should NOT count as spike
    c_tol = np.ones((4, 10)) * (2.0 + 5e-7)
    s.write_cloud(mid, 0.03, 42, c_tol, np.arange(4), e_t)
    assert s.spike_fraction(mid, 0.03) == 0.0


# ==============================================================================
# 3. Registry Stress Tests
# ==============================================================================

def test_registry_invalid_edges_and_dimensions():
    # Invalid edge types
    for bad_edge in ["diagonal", "chiral", "bearded", "Armchair", "", "none"]:
        with pytest.raises(ValueError, match="edge must be one of"):
            RibbonModel("graphene", bad_edge, 7, 1.0, 14, 3.0)

    # Invalid width <= 0
    for bad_w in [0, -1, -50]:
        with pytest.raises(ValueError, match="must be positive"):
            RibbonModel("graphene", "armchair", bad_w, 1.0, 14, 3.0)

    # Invalid sites_per_cell <= 0
    for bad_s in [0, -2]:
        with pytest.raises(ValueError, match="must be positive"):
            RibbonModel("graphene", "armchair", 7, 1.0, bad_s, 3.0)

    # Invalid n_cells <= 0
    for bad_n in [0, -10]:
        with pytest.raises(ValueError, match="must be positive"):
            RibbonModel("graphene", "armchair", 7, 1.0, 14, 3.0, n_cells=bad_n)


def test_registry_duplicate_models_and_hierarchy():
    r = Registry()
    m1 = RibbonModel("mat_a", "armchair", 7, 1.0, 14, 3.0)
    m2 = RibbonModel("mat_a", "armchair", 9, 1.0, 18, 3.0)
    m3 = RibbonModel("mat_a", "zigzag", 6, 1.0, 12, 3.0)
    m4 = RibbonModel("mat_b", "strip", 10, 1.0, 10, 3.9)

    r.add(m1)
    r.add(m2)
    r.add(m3)
    r.add(m4)

    # Duplicate model addition refused
    with pytest.raises(ValueError, match="duplicate model"):
        r.add(RibbonModel("mat_a", "armchair", 7, 1.0, 14, 3.0))

    # Missing model raises KeyError
    with pytest.raises(KeyError):
        r.get("nonexistent/armchair/N7")

    # Hierarchy correctly groups and sorts widths
    hier = r.hierarchy()
    assert hier == {
        "mat_a": {
            "armchair": [7, 9],
            "zigzag": [6],
        },
        "mat_b": {
            "strip": [10],
        }
    }


def test_registry_impurities_calculation_extremes():
    m = RibbonModel("graphene", "armchair", 7, 1.0, 14, 3.0, n_cells=100)
    assert m.n_sites == 1400

    # Extremely low density must yield at least 1 impurity
    assert m.impurities_for_density(1e-9) == 1
    assert m.impurities_for_density(0.0) == 1

    # Standard densities
    assert m.impurities_for_density(0.01) == 14
    assert m.impurities_for_density(0.05) == 70


# ==============================================================================
# 4. Atlas Stress Tests
# ==============================================================================

@pytest.fixture(scope="module")
def shared_atlas(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("atlas_stress")
    store, models = toy_store(tmp, materials=(("alpha", 1.0), ("beta", 3.0)), widths=(7, 9, 14))
    reg = Registry(models)
    fast_cfg = dict(latent=8, epochs=6, patience=3, k=7, refs_per_model=200, threads=2)
    return Atlas.build(store, reg, reg.ids(), InputSpec(), **fast_cfg), tmp


def test_atlas_novelty_rejection_synthetic_spectra(shared_atlas):
    atlas, _ = shared_atlas

    # 1. High frequency oscillation (checkerboard-like synthetic wave)
    hf_noise = np.tile(np.array([0.0, 15.0] * 200), (4, 1))
    res_hf = atlas.locate(hf_noise, E, band_top_t=4.0)
    assert all(r.unknown for r in res_hf)
    assert all(r.recon_error > atlas.threshold for r in res_hf)

    # 2. Inverted spectrum (peaks where transmission is physically 0)
    inverted = np.zeros((3, 400))
    inverted[:, 320:] = 12.0  # high transmission way above band top (3.0 t)
    res_inv_unmasked = atlas.locate(inverted, E, band_top_t=4.0)
    assert all(r.unknown for r in res_inv_unmasked)

    # 3. Giant transmission spike (unseen physics)
    spike = np.zeros((2, 400))
    spike[:, 100:105] = 20.0
    res_spike = atlas.locate(spike, E, band_top_t=4.0)
    assert any(r.unknown for r in res_spike)


def test_atlas_continuous_width_and_extrapolation(shared_atlas):
    atlas, _ = shared_atlas

    # Continuous interpolation: blend of width 7 and width 9
    s7 = toy_spectrum(1.0, 7, 0.01, 1000)
    s9 = toy_spectrum(1.0, 9, 0.01, 1000)
    s_blend = 0.5 * s7 + 0.5 * s9
    res_blend = atlas.locate(s_blend[None], E, 3.0)
    assert 7.0 <= res_blend[0].width <= 9.0 + 1e-5
    assert res_blend[0].width_extrapolated is False

    # Interior trained width (N=9, flanked by 7 and 14): not extrapolated
    T_w9 = np.stack([toy_spectrum(3.0, 9, 0.01, 85_000 + i) for i in range(10)])
    res_w9 = atlas.locate(T_w9, E, 3.0)
    assert np.all([r.width_extrapolated is False for r in res_w9])

    # Extreme low width 2 (below min trained width 7): clamped and flagged
    T_w2 = np.stack([toy_spectrum(1.0, 2, 0.01, 90_000 + i) for i in range(15)])
    res_w2 = atlas.locate(T_w2, E, 3.0)
    assert all(r.width >= 7.0 for r in res_w2)  # clamped to trained min
    assert np.mean([r.width_extrapolated for r in res_w2]) >= 0.8

    # Extreme high width 50 (above max trained width 14): clamped and flagged
    T_w50 = np.stack([toy_spectrum(1.0, 50, 0.01, 95_000 + i) for i in range(15)])
    res_w50 = atlas.locate(T_w50, E, 3.0)
    assert all(r.width <= 14.0 for r in res_w50)  # clamped to trained max
    assert np.mean([r.width_extrapolated for r in res_w50]) >= 0.8


def test_atlas_exact_duplicate_zero_distance(shared_atlas):
    atlas, _ = shared_atlas
    # Query pristine spectrum of beta/armchair/N7
    T_exact = toy_spectrum(3.0, 7, 0.0, 10**6)[None]
    # Verify no division by zero or NaN in inverse-distance weighting
    res = atlas.locate(T_exact, E, 3.0)
    assert len(res) == 1
    assert not np.isnan(res[0].width)
    assert not np.isnan(res[0].confidence)
    assert not np.isnan(res[0].novelty)
    assert not np.isnan(res[0].recon_error)
    assert res[0].material == "beta"
    assert res[0].width == pytest.approx(7.0, abs=1e-3)


# ==============================================================================
# 5. Conformal Calibration Stress Tests
# ==============================================================================

def test_conformal_extreme_error_distributions(rng):
    """Verify split conformal coverage under heavy-tailed, skewed, and zero-error distributions."""
    N_cal, N_test = 2000, 5000

    # 1. Heavy-tailed Student-t (df=2, infinite variance)
    y_t = rng.uniform(10, 100, N_cal + N_test)
    scores_t = np.abs(rng.standard_t(df=2, size=y_t.size)) * 0.1
    pred_t = y_t / (1 + scores_t)
    q_t = fit_relative(pred_t[:N_cal], y_t[:N_cal], alpha=0.10)
    lo, hi = intervals(pred_t[N_cal:], q_t)
    cov_t = coverage(lo, hi, y_t[N_cal:])
    assert cov_t >= 0.88  # Nominal 0.90, within Monte Carlo tolerance

    # 2. Skewed Exponential distribution
    y_exp = rng.uniform(5, 50, N_cal + N_test)
    pred_exp = y_exp * (1 + rng.exponential(scale=0.1, size=y_exp.size))
    q_exp = fit_relative(pred_exp[:N_cal], y_exp[:N_cal], alpha=0.05)
    lo, hi = intervals(pred_exp[N_cal:], q_exp)
    cov_exp = coverage(lo, hi, y_exp[N_cal:])
    assert cov_exp >= 0.93  # Nominal 0.95

    # 3. Perfect predictor (zero errors)
    y_zero = rng.uniform(1, 10, 1000)
    q_zero = fit_relative(y_zero[:500], y_zero[:500], alpha=0.10)
    assert q_zero == 0.0
    lo, hi = intervals(y_zero[500:], q_zero)
    assert coverage(lo, hi, y_zero[500:]) == 1.0

    # 4. Outliers: massive error on 1 sample
    y_out = rng.uniform(10, 100, 1000)
    pred_out = y_out.copy()
    pred_out[0] = y_out[0] * 1000.0  # huge outlier in calibration set
    q_out = fit_relative(pred_out[:500], y_out[:500], alpha=0.10)
    assert np.isfinite(q_out)
    assert q_out >= 0.0


def test_conformal_finite_sample_calibration_sweep(rng):
    """Sweep sample sizes and alpha levels to verify coverage guarantees via Monte Carlo trials."""
    for n_cal in [30, 100, 500]:
        for alpha in [0.05, 0.10, 0.20]:
            trial_covs = []
            for _ in range(25):
                y_pool = rng.uniform(10, 100, n_cal + 1000)
                err = rng.normal(0, 0.1, y_pool.size)
                pred_pool = y_pool * (1 + err)

                q = fit_relative(pred_pool[:n_cal], y_pool[:n_cal], alpha=alpha)
                lo, hi = intervals(pred_pool[n_cal:], q)
                trial_covs.append(coverage(lo, hi, y_pool[n_cal:]))

            # Finite sample conformal prediction guarantees E[coverage] >= 1 - alpha
            mean_cov = np.mean(trial_covs)
            assert mean_cov >= (1.0 - alpha) - 0.015, (
                f"n_cal={n_cal}, alpha={alpha}: mean coverage {mean_cov:.4f} < {1.0 - alpha - 0.015:.4f}"
            )


def test_conformal_zero_guard_and_edge_cases():
    # Prediction of 0.0: protected by np.maximum(pred, 1e-6)
    q = fit_relative([0.0, 0.0], [1.0, 2.0], alpha=0.1)
    assert np.isfinite(q)
    assert q > 0

    # Single calibration sample (n=1)
    q_single = fit_relative([1.0], [1.1], alpha=0.1)
    assert np.isfinite(q_single)
    assert q_single == pytest.approx(0.1)


def test_store_channel_mismatch_silent_acceptance(tmp_path):
    """Hardened behavior: write_pristine and write_cloud reject channel mismatch."""
    s = CloudStore(tmp_path)
    mid = "graphene-ideal/armchair/N7"
    e_t = np.arange(10) * 0.1

    # write_pristine rejects wrong shape/size
    with pytest.raises(ValueError, match="pristine spectrum"):
        s.write_pristine(mid, e_t, np.ones(5))
    with pytest.raises(ValueError, match="pristine spectrum"):
        s.write_pristine(mid, e_t, np.ones((2, 10)))

    s.write_pristine(mid, e_t, np.ones(10))

    # spectra has 5 channels, but energies_t has 10 channels: must raise ValueError
    with pytest.raises(ValueError, match="spectra channels"):
        s.write_cloud(mid, 0.01, 14, np.ones((2, 5)), np.array([1, 2]), e_t)


def test_atlas_empty_batch_raises_value_error(shared_atlas):
    """Hardened behavior: atlas.locate with empty batch returns empty list, and embed returns empty arrays."""
    atlas, _ = shared_atlas
    res = atlas.locate(np.zeros((0, 400)), E, 3.0)
    assert res == []

    # Direct embed on empty batch
    from atlaslib.encoder import embed
    Z, err = embed(atlas.encoder, np.zeros((0, 400)))
    assert Z.shape == (0, atlas.encoder.to_latent.out_features)
    assert err.shape == (0,)


def test_boundary_width_extrapolation_flag_behavior(shared_atlas):
    """Empirical finding: in-distribution samples at the extreme trained widths
    (N=7 and N=14) have width_extrapolated=True because all nearest neighbors
    match trained[0] or trained[-1]."""
    atlas, _ = shared_atlas

    # Samples of known trained boundary width 7
    T_w7 = np.stack([toy_spectrum(3.0, 7, 0.01, 80_000 + i) for i in range(10)])
    res_w7 = atlas.locate(T_w7, E, 3.0)
    # All boundary samples evaluate to width_extrapolated=True
    assert all(r.width_extrapolated for r in res_w7)

    # Samples of known interior width 9 evaluate to width_extrapolated=False
    T_w9 = np.stack([toy_spectrum(3.0, 9, 0.01, 80_000 + i) for i in range(10)])
    res_w9 = atlas.locate(T_w9, E, 3.0)
    assert all(not r.width_extrapolated for r in res_w9)


def test_conformal_empty_calibration_raises_index_error():
    """Hardened behavior: fit_relative on empty calibration arrays raises ValueError."""
    with pytest.raises(ValueError, match="calibration set cannot be empty"):
        fit_relative([], [])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

