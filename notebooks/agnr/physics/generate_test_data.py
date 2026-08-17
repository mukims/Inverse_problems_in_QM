#!/usr/bin/env python
"""
generate_test_data.py -- generate held-out test datasets of AGNR transmission spectra.

Computes the energy-dependent transmission spectrum T(E) for specified
concentrations and random impurity configurations across 100 unit cells.

Usage:
------
    python generate_test_data.py
    python generate_test_data.py --concs 3 5 7 9 11 13 15 17 19 21 --nconfigs 50
    python generate_test_data.py --size 7 --nconfigs 100 --start-cfg 5000 --out-dir ../../data/test/transmission_results_new
"""

import os
import sys
import argparse
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add local directory for agnr_lib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agnr_lib as A


def compute_single_spectrum(args):
    cfg, conc, size, leads, d, nonlocal_mode = args
    w_vals = A.energy_grid()
    out = np.zeros(len(w_vals), dtype=np.float32)
    for i, w in enumerate(w_vals):
        out[i] = A.device_transmission(
            w=w,
            d=d,
            t=A.T_DEFAULT,
            e=0.0,
            m=size,
            config=cfg,
            concentration=conc,
            leads=leads,
            nonlocal_mode=nonlocal_mode,
        )
    return cfg, out


def generate_test_dataset(concs, size=7, nconfigs=100, start_cfg=0,
                          out_dir=None, leads_file=None, workers=None,
                          d=1e-4, nonlocal_mode="IR"):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]

    if out_dir is None:
        out_dir = project_root / "data" / "test" / "transmission_results"
    else:
        out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve lead Green's function
    if leads_file is None:
        lead_candidates = [
            project_root / "data" / "agnr" / f"size_{size}" / f"leads_{size}.npy",
            project_root / "leads" / f"agnr_{size}.npy",
            Path.home() / "Desktop" / "backup" / "agnr" / f"size_{size}" / f"leads_{size}.npy",
        ]
        lead_path = None
        for cand in lead_candidates:
            if cand.exists():
                lead_path = cand
                break
        if lead_path is None:
            print(f"[INFO] Computing leads for size {size} via Sancho-Rubio decimation...")
            w_vals = A.energy_grid()
            leads, _ = A.leads_sancho_rubio(w_vals, d, A.T_DEFAULT, 0.0, size)
        else:
            print(f"[INFO] Loading precomputed leads from: {lead_path}")
            leads = np.load(str(lead_path))
    else:
        leads = np.load(str(leads_file))

    if workers is None:
        workers = min(20, max(1, cpu_count() - 2))

    print(f"[INFO] Output Directory: {out_dir}")
    print(f"[INFO] System: {size}-AGNR (100 unit cells, 2m={2*size} sites/cell)")
    print(f"[INFO] Concentrations ({len(concs)}): {concs}")
    print(f"[INFO] Configs per concentration: {nconfigs} (start_cfg={start_cfg})")
    print(f"[INFO] Parallel Workers: {workers}")
    print(f"[INFO] Broadening d: {d}, Nonlocal Mode: {nonlocal_mode}")

    total_start = time.time()

    for conc in concs:
        c_start = time.time()
        print(f"\n[INFO] Starting generation for concentration c = {conc}")
        tasks = [
            (cfg, conc, size, leads, d, nonlocal_mode)
            for cfg in range(start_cfg, start_cfg + nconfigs)
        ]

        with Pool(processes=workers) as pool:
            for cfg, spec in tqdm(
                pool.imap_unordered(compute_single_spectrum, tasks),
                total=nconfigs,
                desc=f"conc {conc:2d}",
            ):
                out_file = out_dir / f"{size}_agnr_conc{conc}_cfg{cfg}_test.npy"
                np.save(str(out_file), spec)

        c_elapsed = time.time() - c_start
        print(f"  ✓ Finished conc {conc} in {c_elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n[DONE] Successfully generated {len(concs) * nconfigs} test spectra in {total_elapsed:.1f}s!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate held-out quantum transport test spectra for AGNR."
    )
    parser.add_argument(
        "--concs",
        type=int,
        nargs="+",
        default=list(range(3, 45, 2)),
        help="List of impurity concentrations (default: 3 5 7 ... 43)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=7,
        help="AGNR width index (default: 7)",
    )
    parser.add_argument(
        "--nconfigs",
        type=int,
        default=100,
        help="Number of configs per concentration (default: 100)",
    )
    parser.add_argument(
        "--start-cfg",
        type=int,
        default=0,
        help="Starting configuration index (default: 0)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: data/test/transmission_results)",
    )
    parser.add_argument(
        "--leads-file",
        type=str,
        default=None,
        help="Path to precomputed leads .npy file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--broadening",
        type=float,
        default=1e-4,
        help="Imaginary broadening eta (default: 1e-4)",
    )
    parser.add_argument(
        "--nonlocal-mode",
        type=str,
        default="IR",
        choices=["IR", "IL"],
        help="Nonlocal operator mode (default: IR for test data)",
    )

    args = parser.parse_args()

    generate_test_dataset(
        concs=args.concs,
        size=args.size,
        nconfigs=args.nconfigs,
        start_cfg=args.start_cfg,
        out_dir=args.out_dir,
        leads_file=args.leads_file,
        workers=args.workers,
        d=args.broadening,
        nonlocal_mode=args.nonlocal_mode,
    )


if __name__ == "__main__":
    main()
