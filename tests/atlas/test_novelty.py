"""Tests for class-conditional novelty scoring (Option B)."""
import numpy as np
import pytest
from atlaslib import Atlas, InputSpec, Registry, RibbonModel
from atlaslib.store import CloudStore

try:
    from toy import E, toy_spectrum, toy_store
except ImportError:
    from tests.atlas.toy import E, toy_spectrum, toy_store


def test_class_conditional_balances_tight_and_broad_classes(tmp_path):
    """A broad class and a tight class each achieve low, controlled false alarms under Option B."""
    store = CloudStore(tmp_path / "store")
    # Model 1: tight spread; Model 2: broad spread
    m_tight = RibbonModel("synth", "armchair", 7, 1.0, 14, 3.0)
    m_broad = RibbonModel("synth", "armchair", 14, 1.0, 28, 3.0)
    reg = Registry([m_tight, m_broad])

    # Pristine with zero noise
    pris_tight = np.clip(1.0 * (1 + np.floor(E * 7 / 3.0)), 0, None)
    pris_tight[E > 3.0] = 0.0
    pris_broad = np.clip(1.0 * (1 + np.floor(E * 14 / 3.0)), 0, None)
    pris_broad[E > 3.0] = 0.0

    store.write_pristine(m_tight.model_id, E, pris_tight)
    store.write_pristine(m_broad.model_id, E, pris_broad)

    # Generate 500 seeds per model at d=0.02
    r = np.random.default_rng(42)
    def make_cloud(base, noise_std):
        specs = []
        for s in range(500):
            # Degrading multiplier so disorder scatters below pristine
            scat = 0.7 + 0.1 * r.random()
            noise = r.normal(0, noise_std, E.size)
            sp = np.clip(base * scat + noise, 0, None)
            sp[E > 3.0] = 0.0
            specs.append(sp)
        return np.array(specs)

    c_tight = make_cloud(pris_tight, 0.005) # very tight
    c_broad = make_cloud(pris_broad, 0.120) # very broad

    seeds = np.arange(500)
    store.write_cloud(m_tight.model_id, 0.02, m_tight.impurities_for_density(0.02), c_tight, seeds, E, max_excess_tol=None)
    store.write_cloud(m_broad.model_id, 0.02, m_broad.impurities_for_density(0.02), c_broad, seeds, E, max_excess_tol=None)

    # Build Atlas: seeds 0..349 train (350), 350..419 val (70), 420..499 test (80)
    atlas = Atlas.build(store, reg, reg.ids(), InputSpec(), latent=8, epochs=12, patience=4, k=7,
                        refs_per_model=250, seed=1, threads=2, max_seed=419, val_seed_min=350)

    # Calibrate Option B on validation seeds
    atlas.calibrate_novelty(store, reg, reg.ids(), val_seed_min=350, max_seed=419, n0=0)
    assert atlas.novelty == "class_conditional_v2"
    assert m_tight.model_id in atlas.threshold_table
    assert m_broad.model_id in atlas.threshold_table

    # Evaluate on test seeds 420..499 (80 samples each)
    test_tight = c_tight[420:]
    test_broad = c_broad[420:]

    loc_tight = atlas.locate(test_tight, E, band_top_t=3.0)
    loc_broad = atlas.locate(test_broad, E, band_top_t=3.0)

    fa_tight = np.mean([r.unknown for r in loc_tight])
    fa_broad = np.mean([r.unknown for r in loc_broad])

    # Both classes should have well-controlled false alarm rate (near 1%, <= 5% on 80 samples)
    assert fa_tight <= 0.05, f"fa_tight={fa_tight}"
    assert fa_broad <= 0.05, f"fa_broad={fa_broad}"


def test_out_of_class_spectrum_is_flagged(tmp_path):
    """An out-of-class spectrum has novelty_ratio > 1.0 and unknown=True."""
    tmp = tmp_path / "toy"
    store, models = toy_store(tmp, n_seeds=50)
    reg = Registry(models)
    atlas = Atlas.build(store, reg, reg.ids(), InputSpec(), latent=8, epochs=6, patience=3, k=7,
                        refs_per_model=150, seed=2, threads=2)
    atlas.calibrate_novelty(store, reg, reg.ids())

    weird = np.tile(np.abs(np.sin(E * 40)) * 8, (5, 1))
    res = atlas.locate(weird, E, band_top_t=4.0)

    assert all(r.unknown for r in res)
    assert all(r.novelty_ratio > 1.0 for r in res)
    assert all(r.unknown_recon for r in res)


def test_save_load_preserves_threshold_table(tmp_path):
    """Atlas save and load preserve the threshold table and novelty version."""
    store, models = toy_store(tmp_path / "toy", n_seeds=50)
    reg = Registry(models)
    atlas = Atlas.build(store, reg, reg.ids(), InputSpec(), latent=8, epochs=5, patience=2, k=5,
                        refs_per_model=100, seed=3, threads=2)
    atlas.calibrate_novelty(store, reg, reg.ids())

    save_dir = tmp_path / "saved_atlas"
    atlas.save(save_dir)

    loaded = Atlas.load(save_dir)
    assert loaded.novelty == "class_conditional_v2"
    assert loaded.threshold_table == atlas.threshold_table
    assert loaded.threshold_params == atlas.threshold_params
    assert loaded.z_star == atlas.z_star
    assert loaded.w_edge == atlas.w_edge
    assert loaded.n0 == atlas.n0
    assert len(loaded._model_nns) == len(atlas._model_nns)

    T = toy_spectrum(1.0, 7, 0.01, 80_000)[None]
    loc_orig = atlas.locate(T, E, 3.0)[0]
    loc_load = loaded.locate(T, E, 3.0)[0]

    assert loc_orig.unknown == loc_load.unknown
    assert pytest.approx(loc_orig.novelty_ratio, rel=1e-5) == loc_load.novelty_ratio
    assert pytest.approx(loc_orig.novelty_s, rel=1e-5) == loc_load.novelty_s


def test_per_edge_z_star_balances_different_edge_tails(tmp_path):
    """Per-edge z* balances false alarm rates when two edges have different score tails."""
    store = CloudStore(tmp_path / "store")
    m_arm = RibbonModel("synth", "armchair", 7, 1.0, 14, 3.0)
    m_zig = RibbonModel("synth", "zigzag", 6, 1.0, 12, 3.0)
    reg = Registry([m_arm, m_zig])

    pris_arm = np.clip(1.0 * (1 + np.floor(E * 7 / 3.0)), 0, None)
    pris_arm[E > 3.0] = 0.0
    pris_zig = np.clip(1.0 * (1 + np.floor(E * 6 / 3.0)), 0, None)
    pris_zig[E > 3.0] = 0.0

    store.write_pristine(m_arm.model_id, E, pris_arm)
    store.write_pristine(m_zig.model_id, E, pris_zig)

    r = np.random.default_rng(123)
    n_seeds = 300
    # Armchair has light (Gaussian) noise
    specs_arm = []
    for _ in range(n_seeds):
        sp = np.clip(pris_arm * 0.8 + r.normal(0, 0.02, E.size), 0, None)
        sp[E > 3.0] = 0.0
        specs_arm.append(sp)

    # Zigzag has heavy-tailed noise (occasional large deviations)
    specs_zig = []
    for _ in range(n_seeds):
        heavy_noise = r.normal(0, 0.02, E.size) + r.exponential(0.05, E.size) * (r.random(E.size) < 0.2)
        sp = np.clip(pris_zig * 0.8 + heavy_noise, 0, None)
        sp[E > 3.0] = 0.0
        specs_zig.append(sp)

    seeds = np.arange(n_seeds)
    store.write_cloud(m_arm.model_id, 0.02, m_arm.impurities_for_density(0.02), np.array(specs_arm), seeds, E, max_excess_tol=None)
    store.write_cloud(m_zig.model_id, 0.02, m_zig.impurities_for_density(0.02), np.array(specs_zig), seeds, E, max_excess_tol=None)

    # 150 train, 75 val (150..224), 75 test (225..299)
    atlas = Atlas.build(store, reg, reg.ids(), InputSpec(), latent=8, epochs=8, patience=3, k=5,
                        refs_per_model=120, seed=42, threads=2, max_seed=224, val_seed_min=150)
    atlas.calibrate_novelty(store, reg, reg.ids(), val_seed_min=150, max_seed=224, n0=0)

    # Zigzag's heavier tail requires higher z* than armchair
    assert isinstance(atlas.z_star, dict)
    assert atlas.z_star["zigzag"] > atlas.z_star["armchair"]

    # Both edges achieve low, balanced false alarms on test set (seeds 225..299)
    loc_arm_test = atlas.locate(np.array(specs_arm)[225:], E, band_top_t=3.0)
    loc_zig_test = atlas.locate(np.array(specs_zig)[225:], E, band_top_t=3.0)

    fa_arm = np.mean([r.unknown for r in loc_arm_test])
    fa_zig = np.mean([r.unknown for r in loc_zig_test])

    assert fa_arm <= 0.06, f"fa_arm={fa_arm}"
    assert fa_zig <= 0.06, f"fa_zig={fa_zig}"


def test_scale_shrinkage_reduces_small_sample_dispersion(tmp_path):
    """Shrinking per-class scale toward edge median reduces dispersion when sample size is small."""
    store = CloudStore(tmp_path / "store")
    # 3 widths of armchair with identical underlying noise distribution
    models = [
        RibbonModel("synth", "armchair", 6, 1.0, 12, 3.0),
        RibbonModel("synth", "armchair", 8, 1.0, 16, 3.0),
        RibbonModel("synth", "armchair", 10, 1.0, 20, 3.0)
    ]
    reg = Registry(models)
    r = np.random.default_rng(999)

    for m in models:
        pris = np.clip(1.0 * (1 + np.floor(E * m.width / 3.0)), 0, None)
        pris[E > 3.0] = 0.0
        store.write_pristine(m.model_id, E, pris)

        # 200 samples per model
        specs = []
        for _ in range(200):
            sp = np.clip(pris * 0.85 + r.normal(0, 0.05, E.size), 0, None)
            sp[E > 3.0] = 0.0
            specs.append(sp)
        store.write_cloud(m.model_id, 0.02, m.impurities_for_density(0.02), np.array(specs), np.arange(200), E, max_excess_tol=None)

    # Train on seeds 0..99
    atlas = Atlas.build(store, reg, reg.ids(), InputSpec(), latent=8, epochs=6, patience=3, k=5,
                        refs_per_model=80, seed=7, threads=2, max_seed=119, val_seed_min=100)

    # Calibrate with small validation sample (seeds 100..114, N=15) without shrinkage (n0=0)
    atlas.calibrate_novelty(store, reg, reg.ids(), val_seed_min=100, max_seed=114, n0=0)
    raw_scales = [atlas.threshold_params[m.model_id]["0.0200"]["w"] for m in models]
    raw_scale_var = float(np.var(raw_scales))

    # Calibrate with shrinkage (n0=50)
    atlas.calibrate_novelty(store, reg, reg.ids(), val_seed_min=100, max_seed=114, n0=50)
    shrunk_scales = [atlas.threshold_params[m.model_id]["0.0200"]["w"] for m in models]
    shrunk_scale_var = float(np.var(shrunk_scales))

    # Shrinkage pulls scales toward the common edge median, reducing variance across classes
    assert shrunk_scale_var < raw_scale_var

