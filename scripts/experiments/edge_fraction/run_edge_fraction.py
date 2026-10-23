#!/usr/bin/env python
"""Can one spectrum tell how many of its impurities sit on the ribbon's edges?

Regression of the edge fraction (share of impurities on the two outermost lines of atoms) from the transmission
spectrum, one ribbon and one density at a time, so the impurity count is fixed. Every spectrum is predicted out of
fold (five folds split by configuration seed) and scored with the misfit (mean squared difference), next to two
controls: predicting the mean (misfit = variance of the label) and the same model trained on shuffled labels.

    PYTHONPATH=src python scripts/experiments/edge_fraction/run_edge_fraction.py --model hbn/armchair/N10
"""
import argparse
import json
import time

import numpy as np
import xgboost as xgb

import paths
from shazam.arrangement import across_axis, edge_fraction, misfit
from tbribbon.configuration import device_atom_coordinates

XGB = dict(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
           tree_method="hist", early_stopping_rounds=50)


def seed_folds(seeds, n_folds=5, val_share=0.15):
    """(train, validation, test) index arrays: test is one contiguous block of seeds, validation the last
    `val_share` of the remaining seeds (early stopping), train the rest. Every seed is tested once."""
    order = np.argsort(seeds, kind="stable")
    folds = []
    for te in np.array_split(order, n_folds):
        rest = order[~np.isin(order, te)]
        n_val = int(round(val_share * len(rest)))
        folds.append((rest[:-n_val], rest[-n_val:], te))
    return folds


def out_of_fold(X, y, seeds, fit, n_folds=5):
    """Prediction for every row from a model that never saw that row's seed. fit(X_tr, y_tr, X_va, y_va) -> model."""
    pred = np.empty(len(y), dtype=np.float64)
    for tr, va, te in seed_folds(seeds, n_folds):
        pred[te] = fit(X[tr], y[tr], X[va], y[va]).predict(X[te])
    return pred


def skill(pred, y):
    """1 - misfit / misfit of predicting the mean: 1 is perfect, 0 is no better than the mean, below 0 is worse."""
    return 1.0 - misfit(pred, y) / misfit(np.full(len(y), np.mean(y)), y)


def skill_interval(pred, y, n_boot=1000, seed=0, level=0.95):
    rng = np.random.default_rng(seed)
    pred, y = np.asarray(pred), np.asarray(y)
    s = [skill(pred[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return tuple(float(q) for q in np.quantile(s, [(1 - level) / 2, (1 + level) / 2]))


def fit_xgb(X_tr, y_tr, X_va, y_va, threads=8, seed=0):
    m = xgb.XGBRegressor(**XGB, n_jobs=threads, random_state=seed)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m


def features(T, clean):
    """Spectrum relative to the clean ribbon (rounded to 3 decimals first, LOGBOOK Bug #6), clipped to [0, 1]."""
    p = np.round(clean, 3)
    return np.clip(np.round(T, 3) / np.where(p > 0, p, 1.0), 0.0, 1.0)


def run_density(folder, key, info, device, across, n_boot):
    T = np.load(folder / f"cloud_{key}.npy")
    seeds = np.load(folder / f"cloud_{key}_seeds.npy")
    with np.load(folder / f"cloud_{key}_config.npz") as z:
        coords = z["coords"]
        if not np.array_equal(z["seeds"], seeds):
            raise ValueError(f"{key}: configuration rows are not in the cloud's seed order")
    X = features(T, np.load(folder / "pristine.npy"))
    y = edge_fraction(coords, device, across)
    pred = out_of_fold(X, y, seeds, fit_xgb)
    y_shuf = np.random.default_rng(0).permutation(y)
    pred_shuf = out_of_fold(X, y_shuf, seeds, fit_xgb)
    mean_pred = np.empty(len(y))
    for tr, _, te in seed_folds(seeds):
        mean_pred[te] = y[tr].mean()
    n_imp = int(info["n_impurities"])
    lo, hi = skill_interval(pred, y, n_boot=n_boot)
    return {"density": info["density"], "n_impurities": n_imp, "n_spectra": int(len(y)),
            "label_mean": float(y.mean()), "label_std": float(y.std()),
            "misfit_model": misfit(pred, y), "misfit_mean": misfit(mean_pred, y), "misfit_shuffled": misfit(pred_shuf, y_shuf),
            "skill_model": skill(pred, y), "skill_model_95ci": [lo, hi], "skill_shuffled": skill(pred_shuf, y_shuf),
            "rmse_edge_impurities_model": float(np.sqrt(misfit(pred, y)) * n_imp),
            "rmse_edge_impurities_mean": float(np.sqrt(misfit(mean_pred, y)) * n_imp)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model", default="hbn/armchair/N10")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=None, help="default: results/edge_fraction/<model>.json")
    a = ap.parse_args()
    t0 = time.time()
    folder = paths.require(paths.model_dir(a.model))
    meta = json.loads((folder / "meta.json").read_text())
    device = device_atom_coordinates(a.model, n_cells=meta.get("pristine_n_cells"))
    across = across_axis(a.model)
    clouds = {k: v for k, v in sorted(meta["clouds"].items()) if "source" not in v}    # the 4 catalogue densities
    res = {"model": a.model, "folds": "5, split by configuration seed", "features": "T / clean, rounded, clipped",
           "regressor": {"xgboost": XGB}, "densities": {}}
    print(f"[edge] {a.model}: {len(clouds)} densities, across axis {'xy'[across]}", flush=True)
    for key, info in clouds.items():
        r = run_density(folder, key, info, device, across, a.n_boot)
        res["densities"][key] = r
        print(f"[edge] {key} ({r['n_impurities']} impurities): edge fraction {r['label_mean']:.3f} +- {r['label_std']:.3f} | "
              f"misfit model {r['misfit_model']:.5f}, mean {r['misfit_mean']:.5f}, shuffled {r['misfit_shuffled']:.5f} | "
              f"skill {r['skill_model']:.3f} [{r['skill_model_95ci'][0]:.3f}, {r['skill_model_95ci'][1]:.3f}], "
              f"shuffled {r['skill_shuffled']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    out = paths.results_dir("edge_fraction", create=True) / f"{a.model.replace('/', '_')}.json" if a.out is None else a.out
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[edge] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
