"""Tiny synthetic 'materials': staircases whose step count grows with width."""
import numpy as np

from atlaslib.registry import RibbonModel
from atlaslib.store import CloudStore

E = np.arange(400) * 0.01


def toy_spectrum(level, width, density, seed):
    r = np.random.default_rng(seed)
    T = level * (1 + np.floor(E * width / 3.0)) * np.exp(-density * 25 * (1 + 0.2 * r.random()))
    T[E > 3.0] = 0.0
    return np.clip(T + r.normal(0, 0.02, E.size), 0, None)


def toy_store(tmp_path, materials=(("alpha", 1.0), ("beta", 3.0)), widths=(7, 9, 14),
              densities=(0.01, 0.04), n_seeds=40):
    store, models = CloudStore(tmp_path / "store"), []
    for name, level in materials:
        for w in widths:
            m = RibbonModel(name, "armchair", w, 1.0, 2 * w, 3.0)
            models.append(m)
            store.write_pristine(m.model_id, E, toy_spectrum(level, w, 0.0, 10**6))
            for d in densities:
                seeds = np.arange(n_seeds) + int(d * 1e5)
                store.write_cloud(m.model_id, d, m.impurities_for_density(d),
                                  np.stack([toy_spectrum(level, w, d, s) for s in seeds]), seeds, E)
    return store, models
