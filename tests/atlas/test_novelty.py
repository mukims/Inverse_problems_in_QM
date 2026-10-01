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
    atlas.calibrate_novelty(store, reg, reg.ids(), val_seed_min=350, max_seed=419)
    assert atlas.novelty == "class_conditional_v1"
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
    assert loaded.novelty == "class_conditional_v1"
    assert loaded.threshold_table == atlas.threshold_table
    assert loaded.threshold_params == atlas.threshold_params
    assert loaded.z_star == atlas.z_star
    assert len(loaded._model_nns) == len(atlas._model_nns)

    T = toy_spectrum(1.0, 7, 0.01, 80_000)[None]
    loc_orig = atlas.locate(T, E, 3.0)[0]
    loc_load = loaded.locate(T, E, 3.0)[0]

    assert loc_orig.unknown == loc_load.unknown
    assert pytest.approx(loc_orig.novelty_ratio, rel=1e-5) == loc_load.novelty_ratio
    assert pytest.approx(loc_orig.novelty_s, rel=1e-5) == loc_load.novelty_s
