#!/usr/bin/env python
"""
mw_xgboost.py — Technique 2/4: Gradient-boosted trees (width classifier + conc regressor).

Two stages: an XGBClassifier separates 7- from 9-AGNR, then its predicted width
probability is appended as an extra feature for the XGBRegressor. That explicit
width conditioning is the one structural advantage trees had over the networks in
BUILD-06 — the neural scripts now do the same thing (see mw_mlp.py IMPROVEMENT 3).

Note trees need no target scaling: they fit raw concentrations natively, which is
exactly why they were unaffected by the target-scale problem that hurt the nets.

Usage
-----
    python mw_xgboost.py
    python mw_xgboost.py --n-estimators-reg 800 --max-depth-reg 10
"""

import time
import argparse

import numpy as np
import xgboost as xgb

import mw_common as mw
from mw_common import log, banner

TAG = "xgboost"
DISPLAY = "XGBoost"


def run(args, train_data, test_data, out_dir, results_dir):
    banner("TECHNIQUE 2/4: XGBOOST (WIDTH CLASSIFIER + CONCENTRATION REGRESSOR)")
    t0 = time.time()
    X_tr, yw_tr, yc_tr, _ = train_data
    X_te, yw_te, yc_te, _ = test_data

    # ---- 1. Width classifier -----------------------------------------
    log(f"[1/2] Width classifier: {args.n_estimators_clf} trees, depth {args.max_depth_clf}, "
        f"{len(X_tr):,} samples...")
    t_clf = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=args.n_estimators_clf, max_depth=args.max_depth_clf,
        learning_rate=args.lr_clf, tree_method="hist", random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr, yw_tr)
    log(f"      fit in {time.time() - t_clf:.1f}s")
    pred_w = clf.predict(X_te)
    clf_path = out_dir / "mw_xgb_width.json"
    clf.save_model(str(clf_path))
    log(f"✓ Saved width classifier -> {clf_path.name}")

    # ---- 2. Concentration regressor ----------------------------------
    log(f"[2/2] Concentration regressor: {args.n_estimators_reg} trees, depth {args.max_depth_reg}...")
    t_reg = time.time()
    reg = xgb.XGBRegressor(
        n_estimators=args.n_estimators_reg, max_depth=args.max_depth_reg,
        learning_rate=args.lr_reg, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, tree_method="hist", random_state=42, n_jobs=-1,
    )
    X_tr_aug = np.hstack([X_tr, clf.predict_proba(X_tr)[:, 1:2]])
    X_te_aug = np.hstack([X_te, clf.predict_proba(X_te)[:, 1:2]])
    log(f"      width-probability feature appended -> {X_tr_aug.shape[1]} features")
    reg.fit(X_tr_aug, yc_tr)
    log(f"      fit in {time.time() - t_reg:.1f}s")
    pred_c = reg.predict(X_te_aug)
    reg_path = out_dir / "mw_xgb_conc.json"
    reg.save_model(str(reg_path))
    log(f"✓ Saved concentration regressor -> {reg_path.name}")

    if args.snap_grid:
        pred_c = mw.snap_to_grid(pred_c, pred_w)
        log("  Applied even-integer grid snapping")

    metrics = mw.compute_multi_metrics(pred_w, yw_te, pred_c, yc_te)
    metrics["train_time_sec"] = round(time.time() - t0, 1)
    mw.report_metrics(DISPLAY, metrics, "train_time_sec")

    mw.save_result(results_dir, TAG, DISPLAY, metrics, pred_w, pred_c, yw_te, yc_te,
                   extra={"snap_grid": bool(args.snap_grid)})
    return metrics


def main():
    parser = argparse.ArgumentParser(description="XGBoost width + concentration models.")
    mw.add_common_args(parser)
    parser.add_argument("--n-estimators-clf", type=int, default=300)
    parser.add_argument("--max-depth-clf", type=int, default=6)
    parser.add_argument("--lr-clf", type=float, default=0.05)
    parser.add_argument("--n-estimators-reg", type=int, default=500)
    parser.add_argument("--max-depth-reg", type=int, default=8)
    parser.add_argument("--lr-reg", type=float, default=0.04)
    args = parser.parse_args()

    out_dir, results_dir = mw.setup_run(args, TAG)
    banner("MULTI-WIDTH SUITE — XGBOOST")
    log(f"Data dir: {args.data_dir}")
    log(f"Out dir:  {out_dir}")

    train_data, _, test_data, _, _ = mw.load_data(
        args.data_dir, args.samples_per_conc, args.spectrum_len)

    run(args, train_data, test_data, out_dir, results_dir)
    log(f"[DONE] xgboost finished in {(time.time() - mw._T_START)/60:.1f} min")


if __name__ == "__main__":
    main()
