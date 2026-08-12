#!/usr/bin/env python
"""
compute_leads_sq.py
===================
Precompute semi-infinite lead surface Green's functions for the
square-lattice / GNR-strip system (``notebooks/square_lattice``).

Each unit cell is a 1-D tight-binding chain of ``l`` sites:

    H_ii = w + i*d - e            (on-site)
    H_i,i+1 = H_i+1,i = t         (nearest-neighbour hopping)

Neighbouring unit cells are coupled by the identity hopping matrix
``H_hop = I_l``. For every energy ``w`` on the grid ``[0, 4)`` (step 0.01,
400 points) we want the surface Green's function ``g_L(w)`` of the
semi-infinite chain, i.e. the fixed point of

    g_s = (unit - H_hop g_s H_hop)^-1          with   unit = w + i d - e + (NN hop)

Two solvers are provided (choose with ``--method``):

  * ``iterative``     -- the plain Dyson fixed-point iteration
                         g <- (I - g H_hop g H_hop)^-1 g, vectorised over the
                         whole 400-point energy grid. Faithful to CA.ipynb /
                         the AGNR lead solver, but converges *linearly* and can
                         need >1e5 steps near band edges.

  * ``sancho-rubio``  -- (DEFAULT) the decimation / renormalisation scheme of
                         M. P. Lopez Sancho, J. M. Lopez Sancho & J. Rubio,
                         "Highly convergent schemes for the calculation of bulk
                         and surface Green functions", J. Phys. F 15 (1985) 851.
                         Each step folds in 2^n unit cells, so the effective
                         chain length doubles every iteration and the surface
                         GF converges *quadratically* -- typically ~20-40 steps
                         instead of ~1e5. See the SANCHO-RUBIO NOTES below.

Both solvers return identical surface Green's functions (to tolerance); use
``--method iterative`` if you want to reproduce the original behaviour exactly.

Output
------
``<out_dir>/leads_{l}.npy``   shape ``(400, l, l)``, dtype ``complex128``,
indexed so that ``leads[int(round(w*100))] == g_L(w)`` -- the exact layout
consumed by ``ca_sq.py`` and by ``leads_combined/``.
A companion ``<out_dir>/leads_{l}_meta.csv`` (row_idx, w) is also written.

Examples
--------
    # default sizes, fast (Sancho-Rubio) solver, default output dir
    python compute_leads_sq.py

    # reproduce the original iterative solver for two sizes
    python compute_leads_sq.py --sizes 10 25 --method iterative
"""

import os
import argparse

import numpy as np
from tqdm import tqdm

# ----------------------------------------------------------------------
# Energy grid (shared with ca_sq.py): 0.00, 0.01, ..., 3.99  -> 400 points
# ----------------------------------------------------------------------
W_MIN, W_MAX, W_STEP = 0.0, 4.0, 0.01


def energy_grid() -> np.ndarray:
    return np.arange(W_MIN, W_MAX, W_STEP)


# ----------------------------------------------------------------------
# 1) Batched unit-cell Hamiltonian  (B, l, l)
# ----------------------------------------------------------------------
def unitcell_batch(w_vals, d, t, e, l):
    """1-D chain unit cell for every energy at once -> (B, l, l) complex.

    Returns ``unit = (w + i d - e) I_l + t * (nearest-neighbour hopping)``,
    the matrix whose inverse is the bare single-cell Green's function.
    """
    B = len(w_vals)
    l = int(l)
    base = np.zeros((B, l, l), dtype=complex)

    diag = (w_vals + 1j * d - e).astype(complex)
    ii = np.arange(l)
    base[:, ii, ii] = diag[:, None]

    if l > 1:
        idy = np.arange(l - 1)
        base[:, idy, idy + 1] = t
        base[:, idy + 1, idy] = t

    return base


# ----------------------------------------------------------------------
# 2a) Vectorised Dyson iteration (original method) — LINEAR convergence
# ----------------------------------------------------------------------
def leads_iterative(w_vals, d, t, e, l, tol=1e-6, max_iter=200000):
    """Iterate g = (I - g H_hop g H_hop)^-1 g for all energies simultaneously.

    Faithful to CA.ipynb's ``leads()`` and the AGNR lead solver. Correct but
    slow: convergence is linear, so band-edge energies can take >1e5 steps.
    """
    l = int(l)
    unit = unitcell_batch(w_vals, d, t, e, l)          # (B, l, l)

    g = np.linalg.inv(unit)                            # single-cell surface = H^-1
    G = g.copy()

    hopp = np.eye(l, dtype=complex)                    # inter-cell hopping = I_l
    iden = np.eye(l, dtype=complex)[None, :, :]        # (1, l, l) broadcast

    diff = np.inf
    count = 0
    pbar = tqdm(total=max_iter, desc=f"Dyson l={l}", leave=False)
    while diff > tol and count < max_iter:
        A = iden - g @ hopp @ G @ hopp                 # (B, l, l)
        G_new = np.linalg.solve(A, g)                  # batched solve
        diff = np.max(np.abs(G_new - G))
        G = G_new
        count += 1
        pbar.update(1)
        pbar.set_postfix({"diff": f"{diff:.2e}"})
    pbar.close()

    if count >= max_iter:
        print(f"[WARN] l={l}: iterative hit max_iter={max_iter} without reaching "
              f"tol={tol} (last diff={diff:.2e})")
    return G, count


# ----------------------------------------------------------------------
# 2b) Sancho-Rubio decimation (default) — QUADRATIC convergence
# ----------------------------------------------------------------------
#
# ===================== SANCHO-RUBIO NOTES ============================
#
# Reference
#   M. P. Lopez Sancho, J. M. Lopez Sancho, J. Rubio,
#   "Highly convergent schemes for the calculation of bulk and surface
#    Green functions", J. Phys. F: Met. Phys. 15 (1985) 851.
#
# Idea
#   The plain Dyson iteration adds ONE unit cell per step, so it needs
#   ~N steps to see N cells and converges linearly. Sancho-Rubio instead
#   *decimates*: at iteration n it has effectively folded 2^n cells into
#   renormalised on-site and hopping blocks. The chain length it "sees"
#   doubles every step, giving quadratic convergence -- tens of steps
#   rather than ~1e5, independent of how flat the band edge is.
#
# Working variables (all l x l, per energy; here batched over energies):
#   W   = (w + i d - e)I + intra-cell hop  -- running *bulk* on-site block
#         (this is exactly ``unit``; the code keeps E-h in one matrix).
#   Ws  = same, but for the *surface* cell (one-sided self-energy only).
#   a   = renormalised hopping that reaches "leftwards"  (starts as H_hop^dagger)
#   b   = renormalised hopping that reaches "rightwards" (starts as H_hop)
#
# Recursion (per step), with  g = W^-1 :
#   a_gb = a @ g @ b
#   b_ga = b @ g @ a
#   Ws <- Ws - a_gb                 # surface sees cells on one side only
#   W  <- W  - a_gb - b_ga          # bulk sees both sides
#   a  <- a @ g @ a                 # hoppings shrink super-exponentially
#   b  <- b @ g @ b
# Stop when max(||a||, ||b||) < tol. Then g_s = Ws^-1.
#
# Sign / convention note
#   Because the inter-cell hopping here is the identity (Hermitian and
#   symmetric), the "left" and "right" renormalised hoppings coincide and
#   the surface is symmetric, so which one-sided term (a_gb vs b_ga) is used
#   for Ws does not matter. For a general non-symmetric coupling you would
#   pick the term matching the side the leads attach on. We keep a and b
#   separate anyway so the routine stays correct for arbitrary H_hop.
#
# Caveats
#   * Needs a finite broadening d > 0: it regularises the W^-1 inversions
#     at band edges where the bare cell is singular. (Same requirement as
#     the iterative method.)
#   * Convergence is measured on the *hoppings* a, b (they -> 0), which is
#     the standard, robust criterion for this scheme.
#   * Result agrees with the iterative solver to `tol` (checked in
#     validate_sq.py).
# ====================================================================

def leads_sancho_rubio(w_vals, d, t, e, l, tol=1e-9, max_iter=200):
    """Surface Green's function via Sancho-Rubio decimation, batched over energy.

    Converges in ~log2(chain length) steps (typically 20-40), versus ~1e5 for
    the iterative method. See the SANCHO-RUBIO NOTES above for the algorithm.
    """
    l = int(l)
    B = len(w_vals)

    unit = unitcell_batch(w_vals, d, t, e, l)          # (B, l, l) == E - h
    hopp = np.eye(l, dtype=complex)                    # inter-cell hopping = I_l

    W = unit.copy()                                    # running bulk on-site (E - h)
    Ws = unit.copy()                                   # running surface on-site
    # broadcast the (constant) couplings to a per-energy batch so they can
    # be renormalised independently at each energy from here on.
    a = np.broadcast_to(hopp.conj().T, (B, l, l)).copy()   # "left"  hopping (H_hop^dagger)
    b = np.broadcast_to(hopp, (B, l, l)).copy()            # "right" hopping (H_hop)

    conv = np.inf
    count = 0
    pbar = tqdm(total=max_iter, desc=f"Sancho-Rubio l={l}", leave=False)
    while conv > tol and count < max_iter:
        g = np.linalg.inv(W)                           # (B, l, l)
        a_gb = a @ g @ b
        b_ga = b @ g @ a

        Ws = Ws - a_gb                                 # surface: one-sided self-energy
        W = W - a_gb - b_ga                            # bulk: both sides

        a = a @ g @ a                                  # hoppings shrink quadratically
        b = b @ g @ b

        # convergence measured on the (vanishing) renormalised hoppings
        conv = max(np.max(np.abs(a)), np.max(np.abs(b)))
        count += 1
        pbar.update(1)
        pbar.set_postfix({"|hop|": f"{conv:.2e}"})
    pbar.close()

    if count >= max_iter:
        print(f"[WARN] l={l}: Sancho-Rubio hit max_iter={max_iter} without reaching "
              f"tol={tol} (last |hop|={conv:.2e})")

    g_s = np.linalg.inv(Ws)                            # surface Green's function
    return g_s, count


# ----------------------------------------------------------------------
# 3) Dispatcher
# ----------------------------------------------------------------------
_SOLVERS = {
    "sancho-rubio": leads_sancho_rubio,
    "iterative": leads_iterative,
}


def compute_size(l, d, t, e, method, tol, max_iter):
    w_vals = energy_grid()
    solver = _SOLVERS[method]
    G_all, iters = solver(w_vals, d, t, e, l, tol=tol, max_iter=max_iter)
    return G_all.astype(np.complex128), iters


# ----------------------------------------------------------------------
# 4) Driver
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Precompute square-lattice lead surface Green's functions.")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60],
                        help="Unit-cell widths l to compute (default matches leads_combined).")
    parser.add_argument("--method", choices=list(_SOLVERS), default="sancho-rubio",
                        help="Surface-GF solver (default: sancho-rubio, fast).")
    parser.add_argument("--out-dir", type=str, default="~/transmissions_sq/leads",
                        help="Directory to write leads_{l}.npy into.")
    parser.add_argument("--d", type=float, default=1e-4,
                        help="Imaginary broadening eta for the leads (default 1e-4).")
    parser.add_argument("--t", type=float, default=1.0, help="Hopping amplitude.")
    parser.add_argument("--e", type=float, default=0.0, help="On-site energy offset.")
    parser.add_argument("--tol", type=float, default=None,
                        help="Convergence tolerance (default: 1e-9 for sancho-rubio, "
                             "1e-6 for iterative).")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Max iterations (default: 200 for sancho-rubio, "
                             "200000 for iterative).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if leads_{l}.npy already exists.")
    args = parser.parse_args()

    # method-dependent defaults
    tol = args.tol if args.tol is not None else (1e-9 if args.method == "sancho-rubio" else 1e-6)
    max_iter = args.max_iter if args.max_iter is not None else (200 if args.method == "sancho-rubio" else 200000)

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    w_vals = energy_grid()
    meta = np.column_stack((np.arange(len(w_vals)), w_vals))

    print(f"[INFO] method   : {args.method}  (tol={tol}, max_iter={max_iter})")
    print(f"[INFO] energies : {len(w_vals)} points on [{W_MIN}, {W_MAX}) step {W_STEP}")
    print(f"[INFO] sizes    : {args.sizes}")
    print(f"[INFO] out_dir  : {out_dir}")
    print(f"[INFO] params   : d={args.d}, t={args.t}, e={args.e}")

    for l in args.sizes:
        fn = os.path.join(out_dir, f"leads_{l}.npy")
        if os.path.exists(fn) and not args.overwrite:
            print(f"[SKIP] {fn} exists (use --overwrite to force).")
            continue

        print(f"[INFO] computing l={l} ...")
        G_all, iters = compute_size(l, args.d, args.t, args.e, args.method, tol, max_iter)
        np.save(fn, G_all)
        np.savetxt(os.path.join(out_dir, f"leads_{l}_meta.csv"), meta,
                   delimiter=",", header="row_idx,w", comments="", fmt=["%d", "%.2f"])
        print(f"[OK]   {fn}  shape={G_all.shape}  ({iters} {args.method} iters)")

    print("\n[DONE] all sizes complete.")


if __name__ == "__main__":
    main()
