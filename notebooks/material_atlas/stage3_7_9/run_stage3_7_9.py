#!/usr/bin/env python
"""Stage 3 concentration models for 7/9-AGNR on the production Atlas v2 front end.

Combines:
  1. Frozen Atlas v2 front end (InputSpec v2, class-conditional novelty thresholds).
  2. Unknown flag filter: spectra flagged unknown (s > tau) receive no estimate and
     are tracked separately.
  3. Width routing: predicted width selects the width-specific XGBoost regressor
     and pristine spectrum.
  4. One XGBoost per width (800 trees, max_depth 8, tree_method="hist") trained
     on seeds 0-2099 across all legacy concentrations.
  5. Split-conformal relative prediction intervals calibrated on seeds 2100-2549.
  6. End-to-end evaluation on held-out test seeds 2550-2999 (37,350 spectra across 83 concs).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

# Ensure atlaslib is on sys.path
def find_repo_root() -> Path:
    cur = Path(__file__).resolve()
    for p in [cur] + list(cur.parents):
        if (p / ".git").exists() or (p / "LOGBOOK.md").exists():
            return p
    fallback = Path("/run/media/shardul/storage/machine_learning/transmission_github/transmissions")
    if fallback.exists():
        return fallback
    raise RuntimeError("Could not resolve repository root")


REPO = find_repo_root()
sys.path.insert(0, str(REPO / "notebooks/material_atlas"))

from atlaslib import Atlas
from atlaslib.conformal import coverage, fit_relative, intervals


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


DATA = find_data_dir(REPO)
DEFAULT_ATLAS = REPO / "notebooks/material_atlas/atlas_v2"
DEFAULT_OUT = REPO / "notebooks/material_atlas/stage3_7_9"
WIDTHS = {7: "size_7.npy", 9: "size_9.npy"}
SITES_PER_CELL = {7: 14, 9: 18} # 100 cells -> 1400 and 1800 total sites


def norm(T, pris):
    """Normalize spectrum by pristine with 3-decimal rounding guard (Data Validity Rule #3)."""
    p = np.round(pris, 3)
    return np.clip(np.round(T, 3) / np.where(p > 0, p, 1.0), 0, 1)


def main():
    ap = argparse.ArgumentParser(description="Stage 3 7/9-AGNR concentration on Atlas v2")
    ap.add_argument("--atlas-path", default=str(DEFAULT_ATLAS), help="Path to production atlas_v2")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory")
    ap.add_argument("--threads", type=int, default=4, help="Thread limit (<= 4)")
    ap.add_argument("--n-estimators", type=int, default=800, help="XGBoost n_estimators")
    ap.add_argument("--quick", action="store_true", help="Quick smoke run")
    a = ap.parse_args()

    # Limit threads strictly to <= 4
    threads = min(a.threads, 4)
    torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)

    n_tr, n_cal, n_te = (120, 150, 180) if a.quick else (2100, 2550, 3000)
    n_est = 50 if a.quick else a.n_estimators
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading Atlas v2 from {a.atlas_path}...")
    atlas = Atlas.load(a.atlas_path)
    print(f"Atlas v2 loaded: {len(atlas.models)} models, novelty={atlas.novelty}, n0={atlas.n0}, z*={atlas.z_star}")

    e_t = np.arange(300) * 0.01
    pris = {w: np.load(REPO / f"{w}_agnr_pris.npy")[:300] for w in WIDTHS}

    print(f"Loading legacy dense data from {DATA}...")
    Xs, Ws, Cs, Seeds_list, Conc_indices = {}, {}, {}, {}, {}
    for w, f in WIDTHS.items():
        mm = np.load(DATA / f, mmap_mode="r")
        n_conc = mm.shape[0]
        concs = 2 * np.arange(1, n_conc + 1)
        for split, sl in (("tr", slice(0, n_tr)), ("cal", slice(n_tr, n_cal)), ("te", slice(n_cal, n_te))):
            T = np.asarray(mm[:, sl, :300]).reshape(-1, 300)
            Xs.setdefault(split, []).append(T)
            Ws.setdefault(split, []).append(np.full(len(T), w))
            Cs.setdefault(split, []).append(np.repeat(concs, sl.stop - sl.start))
            # Track seeds and concentration indices
            seed_grid = np.tile(np.arange(sl.start, sl.stop), n_conc)
            c_idx_grid = np.repeat(np.arange(n_conc), sl.stop - sl.start)
            Seeds_list.setdefault(split, []).append(seed_grid)
            Conc_indices.setdefault(split, []).append(c_idx_grid)

    T_ = {k: np.concatenate(v) for k, v in Xs.items()}
    W_ = {k: np.concatenate(v) for k, v in Ws.items()}
    C_ = {k: np.concatenate(v).astype(float) for k, v in Cs.items()}
    Seeds_ = {k: np.concatenate(v) for k, v in Seeds_list.items()}
    C_idx_ = {k: np.concatenate(v) for k, v in Conc_indices.items()}

    print(f"Dataset splits: train={len(T_['tr'])}, cal={len(T_['cal'])}, test={len(T_['te'])}")

    print(f"Training XGBoost regressors (n_est={n_est}, max_depth=8, threads={threads})...")
    regs = {}
    for w in WIDTHS:
        mask_w = (W_["tr"] == w)
        X_w = norm(T_["tr"][mask_w], pris[w])
        y_w = C_["tr"][mask_w]
        print(f"  Fitting XGBoost for width {w} (N={len(y_w)})...")
        reg = xgb.XGBRegressor(
            n_estimators=n_est,
            max_depth=8,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=42,
            n_jobs=threads,
        )
        reg.fit(X_w, y_w)
        regs[w] = reg

    print("Locating calibration spectra via Atlas v2...")
    loc_cal = atlas.locate(T_["cal"], e_t, band_top_t=3.0)
    unk_cal = np.array([r.unknown for r in loc_cal], dtype=bool)
    w_hat_cal = np.array([7 if abs(r.width - 7) < abs(r.width - 9) else 9 for r in loc_cal])

    # Unflagged calibration spectra get estimates
    c_hat_cal = np.full(len(T_["cal"]), np.nan)
    for w in WIDTHS:
        m = (~unk_cal) & (w_hat_cal == w)
        if np.any(m):
            c_hat_cal[m] = regs[w].predict(norm(T_["cal"][m], pris[w]))

    # Fit relative conformal interval on unflagged calibration predictions
    unflagged_cal_mask = ~unk_cal
    q = fit_relative(c_hat_cal[unflagged_cal_mask], C_["cal"][unflagged_cal_mask], alpha=0.1)
    print(f"Conformal calibration (alpha=0.1): unflagged={np.sum(unflagged_cal_mask)}/{len(unk_cal)} (unk={np.mean(unk_cal)*100:.2f}%), relative halfwidth q={q:.6f}")

    print("Locating test spectra via Atlas v2...")
    loc_te = atlas.locate(T_["te"], e_t, band_top_t=3.0)
    unk_te = np.array([r.unknown for r in loc_te], dtype=bool)
    w_hat_te = np.array([7 if abs(r.width - 7) < abs(r.width - 9) else 9 for r in loc_te])
    w_round_te = np.array([round(r.width) for r in loc_te])
    w_vote_te = np.array([round(r.width_vote) for r in loc_te])
    recon_err_te = np.array([r.recon_error for r in loc_te])
    s_te = np.array([r.novelty_s for r in loc_te])
    pred_dens_te = np.array([r.density for r in loc_te])

    # Test predictions: unflagged get estimates; flagged get NaN (no estimate)
    c_hat_unflagged = np.full(len(T_["te"]), np.nan)
    c_hat_all = np.empty(len(T_["te"]))

    for w in WIDTHS:
        m_unflagged = (~unk_te) & (w_hat_te == w)
        if np.any(m_unflagged):
            c_hat_unflagged[m_unflagged] = regs[w].predict(norm(T_["te"][m_unflagged], pris[w]))
        m_all = (w_hat_te == w)
        if np.any(m_all):
            c_hat_all[m_all] = regs[w].predict(norm(T_["te"][m_all], pris[w]))

    lo_te, hi_te = np.full(len(T_["te"]), np.nan), np.full(len(T_["te"]), np.nan)
    unflagged_te_mask = ~unk_te
    lo_te[unflagged_te_mask], hi_te[unflagged_te_mask] = intervals(c_hat_unflagged[unflagged_te_mask], q)

    cov_90 = float(coverage(lo_te[unflagged_te_mask], hi_te[unflagged_te_mask], C_["te"][unflagged_te_mask]) * 100)
    mae_unflagged = float(np.mean(np.abs(c_hat_unflagged[unflagged_te_mask] - C_["te"][unflagged_te_mask])))
    mae_all = float(np.mean(np.abs(c_hat_all - C_["te"])))

    width_acc = float(np.mean(w_hat_te == W_["te"]) * 100)
    width_acc_round = float(np.mean(w_round_te == W_["te"]) * 100)
    width_acc_vote = float(np.mean(w_vote_te == W_["te"]) * 100)
    width_acc_unflagged = float(np.mean(w_hat_te[unflagged_te_mask] == W_["te"][unflagged_te_mask]) * 100)

    # Per-width breakdown
    mae_by_width_unflagged = {}
    mae_by_width_all = {}
    unk_by_width = {}
    width_acc_by_width = {}
    for w in WIDTHS:
        w_mask = (W_["te"] == w)
        w_unflagged = w_mask & (~unk_te)
        mae_by_width_unflagged[str(w)] = float(np.mean(np.abs(c_hat_unflagged[w_unflagged] - C_["te"][w_unflagged])))
        mae_by_width_all[str(w)] = float(np.mean(np.abs(c_hat_all[w_mask] - C_["te"][w_mask])))
        unk_by_width[str(w)] = float(np.mean(unk_te[w_mask]) * 100)
        width_acc_by_width[str(w)] = float(np.mean(w_hat_te[w_mask] == w) * 100)

    # Per-concentration breakdown
    conc_breakdown = {}
    for w in WIDTHS:
        conc_breakdown[str(w)] = []
        w_mask = (W_["te"] == w)
        concs_w = np.unique(C_["te"][w_mask])
        n_sites = SITES_PER_CELL[w] * 100
        for c in sorted(concs_w):
            c_mask = w_mask & (C_["te"] == c)
            d = c / n_sites
            c_unf_mask = c_mask & (~unk_te)
            n_tot = int(np.sum(c_mask))
            n_unf = int(np.sum(c_unf_mask))
            unk_pct = float(np.mean(unk_te[c_mask]) * 100)
            w_acc_c = float(np.mean(w_hat_te[c_mask] == w) * 100)
            mae_c_unf = float(np.mean(np.abs(c_hat_unflagged[c_unf_mask] - c))) if n_unf > 0 else None
            mae_c_all = float(np.mean(np.abs(c_hat_all[c_mask] - c)))
            cov_c = float(coverage(lo_te[c_unf_mask], hi_te[c_unf_mask], c) * 100) if n_unf > 0 else None
            conc_breakdown[str(w)].append({
                "concentration": int(c),
                "density": float(d),
                "n_total": n_tot,
                "n_unflagged": n_unf,
                "unknown_rate": unk_pct,
                "width_accuracy": w_acc_c,
                "mae_unflagged": mae_c_unf,
                "mae_all": mae_c_all,
                "coverage_90": cov_c,
            })

    res = {
        "width_accuracy": width_acc,
        "width_accuracy_round": width_acc_round,
        "width_accuracy_vote": width_acc_vote,
        "width_accuracy_unflagged": width_acc_unflagged,
        "width_accuracy_by_width": width_acc_by_width,
        "unknown_flag_rate": float(np.mean(unk_te) * 100),
        "unknown_by_width": unk_by_width,
        "end_to_end_mae": mae_unflagged,
        "end_to_end_mae_all": mae_all,
        "mae_by_width": mae_by_width_unflagged,
        "mae_by_width_all": mae_by_width_all,
        "coverage_90": cov_90,
        "interval_relative_halfwidth": float(q),
        "n_test": int(len(T_["te"])),
        "n_unflagged": int(np.sum(unflagged_te_mask)),
        "n_unknown": int(np.sum(unk_te)),
        "criteria_met": {
            "width_accuracy>=99.5": bool(width_acc >= 99.5),
            "mae<=1.98": bool(mae_unflagged <= 1.98),
            "coverage_90+-2": bool(abs(cov_90 - 90.0) <= 2.0),
        },
        "concentration_breakdown": conc_breakdown,
    }

    # Save metrics JSON
    metrics_file = out / "metrics.json"
    metrics_file.write_text(json.dumps(res, indent=2))
    print(f"\nSaved metrics to {metrics_file}")

    # Save test predictions for reviewer recomputation
    preds_file = out / "predictions_test.npz"
    np.savez_compressed(
        preds_file,
        y_true=C_["te"],
        w_true=W_["te"],
        seeds=Seeds_["te"],
        c_idx=C_idx_["te"],
        w_hat=w_hat_te,
        w_round=w_round_te,
        w_vote=w_vote_te,
        unknown=unk_te,
        c_hat_unflagged=c_hat_unflagged,
        c_hat_all=c_hat_all,
        lo=lo_te,
        hi=hi_te,
        q=np.array([q]),
        recon_error=recon_err_te,
        novelty_s=s_te,
        pred_density=pred_dens_te,
    )
    print(f"Saved test predictions to {preds_file}")

    # Summary report
    print("\n" + "=" * 60)
    print("STAGE 3 CONCENTRATION EVALUATION REPORT (STAGE3-1)")
    print("=" * 60)
    print(f"Label-Free Width Accuracy:     {width_acc:.3f}% (round: {width_acc_round:.3f}%, vote: {width_acc_vote:.3f}%) [Gate >=99.5%: {'PASS' if res['criteria_met']['width_accuracy>=99.5'] else 'FAIL'}]")
    print(f"End-to-End MAE (unflagged):    {mae_unflagged:.3f} impurities (7: {mae_by_width_unflagged['7']:.3f}, 9: {mae_by_width_unflagged['9']:.3f}) [Gate <=1.98: {'PASS' if res['criteria_met']['mae<=1.98'] else 'FAIL'}]")
    print(f"End-to-End MAE (all spectra):  {mae_all:.3f} impurities (7: {mae_by_width_all['7']:.3f}, 9: {mae_by_width_all['9']:.3f})")
    print(f"90% Conformal Coverage:        {cov_90:.2f}% (relative q={q:.4f}) [Gate 90+-2%: {'PASS' if res['criteria_met']['coverage_90+-2'] else 'FAIL'}]")
    print(f"Unknown Flag Rate:             {res['unknown_flag_rate']:.2f}% ({res['n_unknown']} / {res['n_test']} flagged)")
    print(f"  7-AGNR Unknown Rate:         {unk_by_width['7']:.2f}%")
    print(f"  9-AGNR Unknown Rate:         {unk_by_width['9']:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
