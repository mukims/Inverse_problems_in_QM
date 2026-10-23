"""Bring legacy spectra into a CloudStore at atlas scale. Legacy grids use impurity
counts, so each target density maps to the nearest available count and the actual
density is recorded (never the target)."""
from pathlib import Path

import numpy as np


def nearest_count(target, available):
    available = np.asarray(sorted(available))
    return int(available[np.argmin(np.abs(available - target))])   # argmin picks the lower on ties


def import_consolidated_agnr(store, model, npy_path, pristine_path, densities, seeds):
    mm = np.load(npy_path, mmap_mode="r")                      # (n_conc, n_seeds, n_energies), c = 2(i+1)
    counts = 2 * np.arange(1, mm.shape[0] + 1)
    e_t = np.arange(mm.shape[2]) * 0.01
    store.write_pristine(model.model_id, e_t, np.load(pristine_path)[: mm.shape[2]])
    seeds = np.asarray(seeds)
    out = {}
    for d in densities:
        c = nearest_count(d * model.n_sites, counts)
        actual = c / model.n_sites
        store.write_cloud(model.model_id, actual, c, np.asarray(mm[c // 2 - 1, seeds]), seeds, e_t)
        out[d] = actual
    return out


def import_square_combined(store, model, combined_dir, pristine_path, densities, seeds):
    combined_dir = Path(combined_dir).expanduser()
    counts = sorted(int(p.stem.split("_")[1]) for p in combined_dir.glob("conc_*.npy"))
    e_t = np.arange(400) * 0.01
    store.write_pristine(model.model_id, e_t, np.load(Path(pristine_path).expanduser())[:400])
    seeds = np.asarray(seeds)
    out = {}
    for d in densities:
        c = nearest_count(d * model.n_sites, counts)
        rows = np.load(combined_dir / f"conc_{c}.npy", mmap_mode="r")
        cfg = np.loadtxt(combined_dir / f"conc_{c}_meta.csv", delimiter=",", skiprows=1, dtype=int)[:, 1]
        pick = [int(np.where(cfg == s)[0][0]) for s in seeds]
        actual = c / model.n_sites
        store.write_cloud(model.model_id, actual, c, np.asarray(rows[pick, :400]), seeds, e_t)
        out[d] = actual
    return out
