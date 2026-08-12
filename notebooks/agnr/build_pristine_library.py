#!/usr/bin/env python
"""
build_pristine_library.py -- pristine (impurity-free) AGNR transmission spectra
for a range of widths, used as the reference library for band-gap / width
identification (step 1 of the inference pipeline).

For each width m we compute T_pristine(w) on the 300-point grid using the
precomputed leads, under BOTH non-local conventions ("IR" = test-data
convention, "IL" = training-data convention) so downstream code can match
whichever the incoming signature was generated with.

Output: <out>/pristine_library.npz containing
    widths      (M,)
    w           (300,)
    T_IR        (M, 300)
    T_IL        (M, 300)
    gap_IR      (M,)   estimated transport gap edge (energy)
    gap_IL      (M,)
"""

import os
import sys
import argparse
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agnr_lib as A  # noqa: E402


def _one(args):
    m, leads_dir, d, t, e = args
    try:
        leads = A.load_leads(m, leads_dir)
    except Exception as ex:
        return m, None, None, f"leads missing: {ex}"
    if leads.shape[0] < len(A.energy_grid()):
        return m, None, None, f"leads too short: {leads.shape}"
    try:
        s_ir = A.spectrum(m, leads, 0, 0, d=d, t=t, e=e, nonlocal_mode="IR")
        s_il = A.spectrum(m, leads, 0, 0, d=d, t=t, e=e, nonlocal_mode="IL")
    except Exception as ex:
        return m, None, None, f"compute failed: {ex}"
    return m, s_ir, s_il, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+",
                    default=list(range(5, 30)))
    ap.add_argument("--leads-dir", type=str, default="~/Desktop/backup/agnr")
    ap.add_argument("--out", type=str, default="~/agnr_infer")
    ap.add_argument("--d", type=float, default=1e-5)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--e", type=float, default=0.0)
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    tasks = [(m, args.leads_dir, args.d, args.t, args.e) for m in args.widths]
    got = {}
    with Pool(args.n_jobs) as p:
        for m, s_ir, s_il, err in p.imap_unordered(_one, tasks):
            if err:
                print(f"[WARN] width {m}: {err}")
                continue
            got[m] = (s_ir, s_il)
            print(f"[OK] width {m}: max T_IR={s_ir.max():.3f}  max T_IL={s_il.max():.3f}")

    widths = sorted(got)
    if not widths:
        raise SystemExit("no widths computed")

    T_IR = np.stack([got[m][0] for m in widths])
    T_IL = np.stack([got[m][1] for m in widths])
    gap_IR = np.array([A.band_gap(T_IR[i])[0] for i in range(len(widths))])
    gap_IL = np.array([A.band_gap(T_IL[i])[0] for i in range(len(widths))])

    path = os.path.join(out, "pristine_library.npz")
    np.savez(path, widths=np.array(widths), w=A.energy_grid(),
             T_IR=T_IR, T_IL=T_IL, gap_IR=gap_IR, gap_IL=gap_IL)
    print(f"\n[DONE] {path}  widths={widths}")
    print("width : gap_IR  gap_IL")
    for i, m in enumerate(widths):
        print(f"{m:>5} : {gap_IR[i]:.2f}    {gap_IL[i]:.2f}")


if __name__ == "__main__":
    main()
