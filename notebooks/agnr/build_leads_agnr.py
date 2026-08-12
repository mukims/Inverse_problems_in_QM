#!/usr/bin/env python
"""
build_leads_agnr.py -- precompute AGNR lead surface Green's functions with the
Sancho-Rubio decimation scheme.

NOT RUN AUTOMATICALLY. This is the "generate more leads" job: it extends the
lead library to whatever widths you want, so width identification can span a
larger candidate set. Run it when you're ready (it's embarrassingly parallel
over widths and cheap per width -- see timings below).

Why Sancho-Rubio
----------------
The original AGNR lead solver (agnr_leads.ipynb / agnr.py::leads_vectorized)
uses the plain Dyson fixed point, which adds ONE unit cell per iteration and
converges linearly -- it needed ~1e5-4e5 iterations and a max_iter guard.
Sancho-Rubio *decimates*: iteration n folds in 2^n cells, so the effective
chain doubles each step and convergence is quadratic -- ~20-40 iterations.

Reference:
    M. P. Lopez Sancho, J. M. Lopez Sancho, J. Rubio,
    "Highly convergent schemes for the calculation of bulk and surface Green
     functions", J. Phys. F: Met. Phys. 15 (1985) 851.

Algorithm (batched over all 300 energies at once; all blocks are 2m x 2m):

    W   = E - h        running BULK on-site block      (beta_matrix)
    Ws  = E - h        running SURFACE on-site block
    a   = T^dagger     renormalised "leftward"  hopping
    b   = T            renormalised "rightward" hopping   (T = T1_matrix)

    repeat:
        g    = W^-1
        a_gb = a g b ;  b_ga = b g a
        Ws  -= a_gb                  # surface: self-energy from one side only
        W   -= a_gb + b_ga           # bulk: both sides
        a    = a g a ;  b = b g b    # hoppings shrink quadratically
    until max(|a|,|b|) < tol

    g_surface = Ws^-1

AGNR-specific note
------------------
Unlike the square-lattice case (where the inter-cell hopping is the identity and
so a == b by symmetry), the AGNR hopping T1_matrix is sparse and NOT symmetric.
The left/right renormalised hoppings genuinely differ, which is why a and b are
tracked separately. `Ws` accumulates only the `a_gb` term because the semi-
infinite lead extends to one side of the surface cell.

Validation
----------
Correctness is checked by the surface Dyson residual

    || (E - h) g_s - T^dagger g_s T g_s - I ||_max  ->  0

which is solver-independent (no reference implementation needed). Use
--validate to print it per width; it should be ~1e-12 or smaller.

Output
------
<out-dir>/size_{m}/leads_{m}.npy   shape (300, 2m, 2m) complex128
matching the layout that agnr_lib.load_leads() / ca_agnr.py expect
(row index = int(round(w*100)), w on [0,3) step 0.01).

Examples
--------
    # extend the library to widths 30-60
    python build_leads_agnr.py --widths $(seq 30 60)

    # rebuild everything with validation, 8 parallel widths
    python build_leads_agnr.py --widths $(seq 5 60) --validate --n-jobs 8 --overwrite

Rough cost: time per width grows like (2m)^3 per energy per iteration; with only
~20 iterations it is seconds for small m and well under a minute for m ~ 60.
"""

import os
import sys
import argparse
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agnr_lib as A  # noqa: E402


def per_energy_residual(g_s, w_vals, d, t, e, m):
    """Surface Dyson residual per energy (max norm over the block), shape (B,).

    g_s must satisfy  (E - h) g_s - T^dag g_s T g_s = I  for the semi-infinite lead.
    """
    unit = A.beta_matrix(w_vals, d, t, e, m)          # (B, 2m, 2m) = E - h
    T = A.T1_matrix(t, m)
    I = np.eye(2 * m, dtype=complex)[None, :, :]
    res = unit @ g_s - T.conj().T @ g_s @ T @ g_s - I
    return np.max(np.abs(res), axis=(1, 2))


def surface_residual(g_s, w_vals, d, t, e, m):
    """Solver-independent correctness check: surface Dyson residual (max norm)."""
    return float(np.max(per_energy_residual(g_s, w_vals, d, t, e, m)))


def robustify(g_s, w_vals, d, t, e, m, tol, max_iter, robust_tol,
              factors=(10, 100, 1000, 10000)):
    """Regularise isolated singular energies (e.g. van Hove points at subband
    edges) without disturbing the rest of the spectrum.

    At the base broadening `d` a few energies can sit on a genuine DOS
    divergence, where the near-singular W^-1 inversions leave a large surface
    Dyson residual. For each such energy we re-solve ONLY that energy with a
    progressively larger broadening (d*10, d*100, ...) until its residual falls
    below `robust_tol`, then splice that regularised block back in. Every other
    energy keeps the base `d`, so the library stays consistent except at the
    unavoidable singular points, which are logged.

    Returns (g_s, info) where info records the bumped energies and their d.
    """
    g_s = g_s.astype(np.complex128)
    res = per_energy_residual(g_s, w_vals, d, t, e, m)
    d_used = np.full(len(w_vals), float(d))
    bad = np.where(res > robust_tol)[0]
    bumped = []

    for f in factors:
        if len(bad) == 0:
            break
        d_try = d * f
        w_bad = w_vals[bad]
        g_bad, _ = A.leads_sancho_rubio(w_bad, d_try, t, e, m, tol=tol, max_iter=max_iter)
        r_bad = per_energy_residual(g_bad.astype(np.complex128), w_bad, d_try, t, e, m)
        fixed = r_bad <= robust_tol
        gi = bad[fixed]
        g_s[gi] = g_bad[fixed].astype(np.complex128)
        d_used[gi] = d_try
        bumped.extend((int(i), d_try) for i in gi)
        bad = bad[~fixed]

    # true post-fix residual: evaluate each energy at the d actually used
    res_final = per_energy_residual(g_s, w_vals, d, t, e, m)
    for i, dt in bumped:
        res_final[i] = per_energy_residual(g_s[i:i + 1], w_vals[i:i + 1], dt, t, e, m)[0]

    info = {
        "n_bumped": len(bumped),
        "bumped": bumped,                       # [(idx, d_used), ...]
        "still_bad": [int(i) for i in bad],     # never converged (should be empty)
        "d_used": d_used,
        "max_res_after": float(res_final.max()),
    }
    return g_s, info


def build_one(args):
    m, out_dir, d, t, e, tol, max_iter, overwrite, validate, robust, robust_tol = args
    m = int(m)
    dir_path = os.path.join(os.path.expanduser(out_dir), f"size_{m}")
    fn = os.path.join(dir_path, f"leads_{m}.npy")
    if os.path.exists(fn) and not overwrite:
        return m, "skip (exists)", None, None, None

    w_vals = A.energy_grid()
    g_s, iters = A.leads_sancho_rubio(w_vals, d, t, e, m, tol=tol, max_iter=max_iter)
    g_s = g_s.astype(np.complex128)

    info = None
    if robust:
        g_s, info = robustify(g_s, w_vals, d, t, e, m, tol, max_iter, robust_tol)

    # residual reported after robustification (per-energy d) if robust, else base d
    if info is not None:
        res = info["max_res_after"]
    elif validate:
        res = surface_residual(g_s, w_vals, d, t, e, m)
    else:
        res = None

    os.makedirs(dir_path, exist_ok=True)
    np.save(fn, g_s)
    np.savetxt(os.path.join(dir_path, f"leads_{m}_meta.csv"),
               np.column_stack((np.arange(len(w_vals)), w_vals)),
               delimiter=",", header="row_idx,w", comments="", fmt=["%d", "%.2f"])
    # log which energies were regularised (broadening differs there)
    if info is not None and info["n_bumped"]:
        rows = np.array([(i, w_vals[i], dt) for i, dt in info["bumped"]])
        np.savetxt(os.path.join(dir_path, f"leads_{m}_robust.csv"), rows,
                   delimiter=",", header="row_idx,w,d_used", comments="",
                   fmt=["%d", "%.2f", "%.1e"])
    return m, f"ok ({iters} iters)", res, g_s.shape, info


def main():
    ap = argparse.ArgumentParser(
        description="Precompute AGNR lead surface Green's functions (Sancho-Rubio).")
    ap.add_argument("--widths", type=int, nargs="+", default=list(range(5, 30)),
                    help="AGNR widths m to compute (unit cell has 2m sites).")
    ap.add_argument("--out-dir", type=str, default="~/Desktop/backup/agnr",
                    help="Base dir; writes <out-dir>/size_{m}/leads_{m}.npy.")
    ap.add_argument("--d", type=float, default=A.D_DEFAULT, help="Broadening eta (default 1e-5).")
    ap.add_argument("--t", type=float, default=A.T_DEFAULT, help="Hopping amplitude.")
    ap.add_argument("--e", type=float, default=0.0, help="On-site offset.")
    ap.add_argument("--tol", type=float, default=1e-12,
                    help="Stop when max|renormalised hopping| < tol.")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=4, help="Widths computed in parallel.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="Print the surface Dyson residual per width (should be ~1e-12).")
    ap.add_argument("--robust", action="store_true",
                    help="Regularise isolated singular energies (van Hove points) by "
                         "locally increasing the broadening only at those energies.")
    ap.add_argument("--robust-tol", type=float, default=1e-3,
                    help="Per-energy residual above which an energy is re-solved (default 1e-3).")
    args = ap.parse_args()

    print(f"[INFO] widths   : {args.widths}")
    print(f"[INFO] out_dir  : {args.out_dir}")
    print(f"[INFO] params   : d={args.d}, t={args.t}, e={args.e}, tol={args.tol}")
    print(f"[INFO] method   : sancho-rubio (quadratic convergence)")

    if args.robust:
        print(f"[INFO] robust    : on (re-solve energies with residual > {args.robust_tol})")

    tasks = [(m, args.out_dir, args.d, args.t, args.e, args.tol, args.max_iter,
              args.overwrite, args.validate, args.robust, args.robust_tol)
             for m in args.widths]

    with Pool(args.n_jobs) as p:
        for m, status, res, shape, info in p.imap_unordered(build_one, tasks):
            extra = f"  shape={shape}" if shape else ""
            extra += f"  residual={res:.2e}" if res is not None else ""
            if info is not None and info["n_bumped"]:
                pts = ", ".join(f"w={info['bumped'][k][0]/100:.2f}@d={info['bumped'][k][1]:.0e}"
                                for k in range(len(info["bumped"])))
                extra += f"  [robust: bumped {info['n_bumped']} energy(s): {pts}]"
                if info["still_bad"]:
                    extra += f"  UNRESOLVED at idx {info['still_bad']}"
            print(f"[width {m:>3}] {status}{extra}")

    print("\n[DONE] lead generation complete.")


if __name__ == "__main__":
    main()
