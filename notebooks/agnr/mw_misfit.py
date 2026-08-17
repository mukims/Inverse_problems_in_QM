#!/usr/bin/env python
"""
mw_misfit.py — Technique 1/4: Physical Misfit Function (analytical baseline).

No learning. Builds a per-concentration reference library by averaging training
spectra, then assigns each test spectrum the (width, concentration) whose
reference minimises the mean-squared misfit over the cropped energy window.

This is the "how far does pure physics get you" control that the three learned
models must beat.

Usage
-----
    python mw_misfit.py
    python mw_misfit.py --crop-start 20 --crop-end 150 --snap-grid
"""

import time
import argparse

import numpy as np

import mw_common as mw
from mw_common import log, pbar, banner

TAG = "misfit"
DISPLAY = "Physical Misfit"


def build_reference_library(X_tr, yw_tr, yc_tr, pris_crop, concs, wlabel, crop):
    """Configuration-averaged reference spectrum per concentration, in raw units."""
    cs, ce = crop
    refs = []
    for c in pbar(concs, f"Refs w={wlabel}", unit="conc"):
        mask = (yw_tr == wlabel) & (yc_tr == c)
        if np.any(mask):
            refs.append(np.mean(X_tr[mask, cs:ce], axis=0) * pris_crop)
        else:
            refs.append(pris_crop)
    return np.asarray(refs, dtype=np.float32)


def run(args, train_data, test_data, p7, p9, results_dir):
    banner("TECHNIQUE 1/4: PHYSICAL MISFIT FUNCTION (ANALYTICAL BASELINE)")
    t0 = time.time()
    crop = (args.crop_start, args.crop_end)
    cs, ce = crop

    X_tr, yw_tr, yc_tr, _ = train_data
    _, yw_te, yc_te, raw_te = test_data

    ref7 = build_reference_library(X_tr, yw_tr, yc_tr, p7[cs:ce], mw.CONCS_7, 0, crop)
    ref9 = build_reference_library(X_tr, yw_tr, yc_tr, p9[cs:ce], mw.CONCS_9, 1, crop)
    log(f"  Reference libraries: 7-AGNR {ref7.shape} | 9-AGNR {ref9.shape} "
        f"| energy window [{cs}:{ce}]")

    raw_crop = raw_te[:, cs:ce]
    L = float(ce - cs)
    N = len(raw_crop)

    # Chunked: the full [N, C, L] difference tensor is ~1 GB at N=37k, C=49, L=130.
    CHUNK = 2048
    min7 = np.empty(N, np.float32); best7 = np.empty(N, np.float32)
    min9 = np.empty(N, np.float32); best9 = np.empty(N, np.float32)

    log(f"  Scanning {N:,} test spectra against {len(ref7)}+{len(ref9)} references...")
    for s in pbar(range(0, N, CHUNK), "Misfit scan", unit="chunk",
                  total=(N + CHUNK - 1) // CHUNK):
        e = min(s + CHUNK, N)
        blk = raw_crop[s:e]
        rows = np.arange(e - s)
        m7 = np.sum((blk[:, None, :] - ref7[None]) ** 2, axis=2) / L
        i7 = m7.argmin(axis=1)
        min7[s:e], best7[s:e] = m7[rows, i7], mw.CONCS_7[i7]
        m9 = np.sum((blk[:, None, :] - ref9[None]) ** 2, axis=2) / L
        i9 = m9.argmin(axis=1)
        min9[s:e], best9[s:e] = m9[rows, i9], mw.CONCS_9[i9]

    # Width decision: whichever library achieves the smaller minimum misfit
    pred_w = np.where(min7 <= min9, 0, 1)
    pred_c = np.where(pred_w == 0, best7, best9)
    if args.snap_grid:
        pred_c = mw.snap_to_grid(pred_c, pred_w)   # already on-grid; no-op safeguard

    metrics = mw.compute_multi_metrics(pred_w, yw_te, pred_c, yc_te)
    metrics["eval_time_sec"] = round(time.time() - t0, 1)
    mw.report_metrics(DISPLAY, metrics, "eval_time_sec")

    mw.save_result(results_dir, TAG, DISPLAY, metrics, pred_w, pred_c, yw_te, yc_te,
                   extra={"crop": [cs, ce], "snap_grid": bool(args.snap_grid)})
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Physical Misfit analytical baseline.")
    mw.add_common_args(parser)
    parser.add_argument("--crop-start", type=int, default=20,
                        help="First energy channel used in the misfit window (default 20)")
    parser.add_argument("--crop-end", type=int, default=150,
                        help="Last energy channel (exclusive, default 150)")
    args = parser.parse_args()

    out_dir, results_dir = mw.setup_run(args, TAG)
    banner("MULTI-WIDTH SUITE — PHYSICAL MISFIT")
    log(f"Data dir: {args.data_dir}")
    log(f"Out dir:  {out_dir}")

    train_data, _, test_data, p7, p9 = mw.load_data(
        args.data_dir, args.samples_per_conc, args.spectrum_len)

    run(args, train_data, test_data, p7, p9, results_dir)
    log(f"[DONE] misfit baseline finished in {(time.time() - mw._T_START)/60:.1f} min")


if __name__ == "__main__":
    main()
