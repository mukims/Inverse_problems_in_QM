#!/usr/bin/env python
"""7/9-AGNR reference solution: label-free identification -> stage-3 concentration -> intervals.
Seeds: atlas clouds use seeds 0-999 (4 densities); stage 3 trains on seeds 0-2099 of every
concentration, calibrates on 2100-2549, and is tested end to end on 2550-2999."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

from atlaslib import Atlas, CloudStore, InputSpec, Registry, RibbonModel
from atlaslib.conformal import coverage, fit_relative, intervals
from atlaslib.importers import import_consolidated_agnr


def find_repo_root() -> Path:
    cur = Path(__file__).resolve()
    for p in [cur] + list(cur.parents):
        if (p / ".git").exists() or (p / "LOGBOOK.md").exists():
            return p
    fallback = Path("/run/media/shardul/storage/machine_learning/transmission_github/transmissions")
    if fallback.exists():
        return fallback
    raise RuntimeError("Could not resolve repository root")


def find_data_dir(repo_root: Path) -> Path:
    candidates = [
        Path("/run/media/shardul/storage/machine_learning/transmission_data/transmission_results/consolidated_data"),
        repo_root / "data/consolidated_data",
        Path.home() / "transmission_data/transmission_results/consolidated_data",
    ]
    for c in candidates:
        if c.exists() and (c / "size_7.npy").exists() and (c / "size_9.npy").exists():
            return c
    raise FileNotFoundError("Could not resolve consolidated_data directory")


REPO = find_repo_root()
DATA = find_data_dir(REPO)
DEFAULT_OUT = REPO / "notebooks/material_atlas/reference_7_9"
WIDTHS = {7: "size_7.npy", 9: "size_9.npy"}
DENSITIES = [0.005, 0.01, 0.02, 0.04]


def norm(T, pris):
    p = np.round(pris, 3)
    return np.clip(np.round(T, 3) / np.where(p > 0, p, 1.0), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="~/atlas_store/reference_v1")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    a = ap.parse_args()

    # Limit threads strictly to <= 4
    threads = min(a.threads, 4)
    torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)

    n_atlas, n_tr, n_cal, n_te = (60, 120, 150, 180) if a.quick else (1000, 2100, 2550, 3000)
    n_est = 50 if a.quick else a.n_estimators
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    reg = Registry([RibbonModel("graphene-ideal", "armchair", w, 1.0, 2 * w, 3.0, source="consolidated_data") for w in WIDTHS])
    store = CloudStore(a.store)
    for m in reg:
        existing = store.densities(m.model_id)
        if len(existing) < len(DENSITIES) or any(store.read_cloud(m.model_id, d)[1].size < n_atlas for d in existing):
            import_consolidated_agnr(store, m, DATA / WIDTHS[m.width], REPO / f"{m.width}_agnr_pris.npy", DENSITIES, np.arange(n_atlas))
    atlas = Atlas.build(store, reg, reg.ids(), InputSpec(), threads=threads, epochs=6 if a.quick else 60)
    atlas.save(out / "atlas")

    e_t = np.arange(300) * 0.01
    Xs, Ws, Cs = {}, {}, {}
    for w, f in WIDTHS.items():                       # every concentration, not only the atlas densities
        mm = np.load(DATA / f, mmap_mode="r")
        conc = 2 * np.arange(1, mm.shape[0] + 1)
        for split, sl in (("tr", slice(0, n_tr)), ("cal", slice(n_tr, n_cal)), ("te", slice(n_cal, n_te))):
            T = np.asarray(mm[:, sl, :300]).reshape(-1, 300)
            Xs.setdefault(split, []).append(T)
            Ws.setdefault(split, []).append(np.full(len(T), w))
            Cs.setdefault(split, []).append(np.repeat(conc, sl.stop - sl.start))
    T_ = {k: np.concatenate(v) for k, v in Xs.items()}
    W_ = {k: np.concatenate(v) for k, v in Ws.items()}
    C_ = {k: np.concatenate(v).astype(float) for k, v in Cs.items()}
    pris = {w: np.load(REPO / f"{w}_agnr_pris.npy")[:300] for w in WIDTHS}

    regs = {w: xgb.XGBRegressor(n_estimators=n_est, max_depth=8, learning_rate=0.04, subsample=0.8,
                                colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", random_state=42, n_jobs=threads)
                .fit(norm(T_["tr"][W_["tr"] == w], pris[w]), C_["tr"][W_["tr"] == w]) for w in WIDTHS}

    def predict(split):
        loc = atlas.locate(T_[split], e_t, band_top_t=3.0)
        w_hat = np.array([7 if abs(r.width - 7) < abs(r.width - 9) else 9 for r in loc])
        c_hat = np.empty(len(w_hat))
        for w in WIDTHS:
            m = w_hat == w
            if m.any():
                c_hat[m] = regs[w].predict(norm(T_[split][m], pris[w]))
        return w_hat, c_hat

    w_cal, c_cal = predict("cal")
    q = fit_relative(c_cal, C_["cal"], alpha=0.1)
    w_te, c_te = predict("te")
    lo, hi = intervals(c_te, q)
    res = {
        "width_accuracy": float(np.mean(w_te == W_["te"]) * 100),
        "end_to_end_mae": float(np.mean(np.abs(c_te - C_["te"]))),
        "mae_by_width": {str(w): float(np.mean(np.abs(c_te - C_["te"])[W_["te"] == w])) for w in WIDTHS},
        "coverage_90": float(coverage(lo, hi, C_["te"]) * 100),
        "interval_relative_halfwidth": float(q),
        "n_test": int(len(c_te)),
    }
    res["criteria_met"] = {
        "width_accuracy>=99.5": bool(res["width_accuracy"] >= 99.5),
        "mae<=1.98": bool(res["end_to_end_mae"] <= 1.98),
        "coverage_90+-2": bool(abs(res["coverage_90"] - 90) <= 2),
    }
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
