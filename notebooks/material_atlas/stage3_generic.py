"""Stage 3 for any ribbon (CONC-1): impurity density from T / T_pristine, split-conformal intervals, Shazam routing."""
from collections import Counter

import numpy as np
import xgboost as xgb

from atlaslib.conformal import coverage, fit_relative, intervals
from atlaslib.energy import on_axis


def norm(T, pris):
    """T / T_pristine with the round-first guard (Data Validity Rule #3)."""
    p = np.round(pris, 3)
    return np.clip(np.round(T, 3) / np.where(p > 0, p, 1.0), 0, 1)


def split_masks(seeds, train_max=699, val_max=849):
    s = np.asarray(seeds)
    return s <= train_max, (s > train_max) & (s <= val_max), s > val_max


def _load(store, mid):
    e_t, pris = store.read_pristine(mid)
    T, y, s = [], [], []
    for d in store.densities(mid):
        c, sd = store.read_cloud(mid, d)
        T.append(c)
        y.append(np.full(len(sd), d))
        s.append(sd)
    return e_t, pris, np.concatenate(T), np.concatenate(y), np.concatenate(s)


def fit_ribbon(store, mid, train_max=699, val_max=849, n_estimators=800, threads=4):
    e_t, pris, T, y, s = _load(store, mid)
    tr, ca, _ = split_masks(s, train_max, val_max)
    F = norm(T, pris)
    reg = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=8, learning_rate=0.04, subsample=0.8,
                           colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", random_state=42, n_jobs=threads)
    reg.fit(F[tr], y[tr])
    return {"reg": reg, "q": fit_relative(reg.predict(F[ca]), y[ca], alpha=0.1), "e_t": e_t, "pris": pris}


def _scores(pred, y, q):
    lo, hi = intervals(pred, q)
    return {"n": int(len(y)), "mae_pp": round(100 * float(np.mean(np.abs(pred - y))), 4),
            "median_rel_err": round(float(np.median(np.abs(pred - y) / y)), 4),
            "coverage": round(coverage(lo, hi, y), 4)}


def evaluate_ribbon(fitted, store, mid, atlas=None, reg_registry=None, train_max=699, val_max=849, split_density=0.04):
    _, _, T, y, s = _load(store, mid)
    te = split_masks(s, train_max, val_max)[2]
    T, y = T[te], y[te]
    pred = fitted["reg"].predict(norm(T, fitted["pris"]))
    out = {"oracle": _scores(pred, y, fitted["q"]) | {
        "per_density": {f"{d:.4f}": _scores(pred[y == d], y[y == d], fitted["q"]) for d in np.unique(y)}}}
    if atlas is not None:
        m = reg_registry.get(mid)
        loc = atlas.locate(T, *on_axis(atlas.spec, m, fitted["e_t"], m.band_top_t))
        unknown = np.array([r.unknown for r in loc])
        got = np.array([r.nearest_model for r in loc])
        routed = (~unknown) & (got == mid)
        sh = {"routed_pct": round(100 * float(routed.mean()), 3), "unknown_pct": round(100 * float(unknown.mean()), 3),
              "misrouted_to": Counter(got[(~unknown) & (got != mid)].tolist()).most_common(3)}
        for name, band in (("inside", y <= split_density + 1e-9), ("above", y > split_density + 1e-9)):
            sel = routed & band
            sh[name] = (_scores(pred[sel], y[sel], fitted["q"]) | {"routed_pct": round(100 * float(routed[band].mean()), 3)}
                        if sel.any() else {"n": 0, "routed_pct": 0.0 if band.any() else None})
        out["shazam"] = sh
    return out
