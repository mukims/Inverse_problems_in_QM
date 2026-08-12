#!/usr/bin/env python
"""
ca_sq.py
========
Generate transmission spectra for large numbers of random impurity
configurations of the square-lattice / GNR-strip system, mirroring the
AGNR pipeline in ``ca_agnr.py``.

For a device of ``NCELLS`` (=100) unit cells, each a 1-D chain of ``l``
sites, impurities are placed at random on the ``100 x l`` site grid. An
impurity shifts the on-site energy of its site to ``w + i*d - 0.5``. The
transmission ``T(w)`` is built by recursively attaching the 100 unit cells
to the precomputed left lead surface Green's function and applying the
Landauer trace formula, for every energy on ``[0, 4)`` step 0.01 (400
points).

Requires the precomputed leads from ``compute_leads_sq.py``:
``leads_{l}.npy`` of shape ``(400, l, l)``.

Output
------
One ``.npy`` (shape ``(400,)``, float64) per configuration:
``<out_dir>/size_{l}_conc_{c}/sq_size{l}_conc{c}_cfg{cfg}.npy``

Raw (unclipped) transmissions are saved -- clip to the pristine spectrum
and configuration-average at analysis time (as in ``CA.ipynb``).

Examples
--------
    # size 10, concentrations 5..50 step 5, 10000 configs each
    python ca_sq.py --size 10 --leads-dir ~/transmissions_sq/leads

    # a targeted run: size 25, two concentrations, 2000 configs
    python ca_sq.py --size 25 --concs 10 20 --nconfigs 2000 \
        --leads ~/transmissions_sq/leads/leads_25.npy
"""

import os
import argparse
from functools import lru_cache
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

# ----------------------------------------------------------------------
# Grid / device geometry (shared with compute_leads_sq.py)
# ----------------------------------------------------------------------
W_MIN, W_MAX, W_STEP = 0.0, 4.0, 0.01
NCELLS = 100          # device length in unit cells


def energy_grid() -> np.ndarray:
    return np.arange(W_MIN, W_MAX, W_STEP)


# ----------------------------------------------------------------------
# 1) Unit-cell Hamiltonian (single energy)
# ----------------------------------------------------------------------
def unitcell(w, d, t, e, l):
    """1-D chain unit cell: diagonal (w + i d - e), NN hopping t."""
    l = int(l)
    base = (w + 1j * d - e) * np.eye(l, dtype=complex)
    if l > 1:
        idy = np.arange(l - 1)
        base[idy, idy + 1] = t
        base[idy + 1, idy] = t
    return base


# ----------------------------------------------------------------------
# 2) Cached random impurity selection  (grid is 100 x l)
# ----------------------------------------------------------------------
DEVICE_COMBS = {}


def _get_device_combs(width: int) -> np.ndarray:
    """All (cell, site) coordinates: cell in [0..NCELLS-1], site in [0..width-1]."""
    width = int(width)
    if width not in DEVICE_COMBS:
        DEVICE_COMBS[width] = np.stack(
            np.meshgrid(np.arange(NCELLS), np.arange(width), indexing="ij"),
            axis=-1,
        ).reshape(-1, 2)
    return DEVICE_COMBS[width]


@lru_cache(maxsize=4096)
def chosen_for_config(n: int, width: int, config: int) -> np.ndarray:
    """Deterministically draw n impurity sites for a given config (seed)."""
    n = int(n)
    width = int(width)
    config = int(config)
    device_combs = _get_device_combs(width)
    rng = np.random.RandomState(config)
    idx = rng.choice(len(device_combs), size=n, replace=False)
    return device_combs[idx]


def possible_combs(n: int, width: int):
    n = int(n)
    width = int(width)

    def combs_for_seed(seed: int):
        return chosen_for_config(n, width, seed)

    return combs_for_seed


# ----------------------------------------------------------------------
# 3) Device unit cell with impurities for a given cell index
# ----------------------------------------------------------------------
def unidevice(w, d, t, e, l, config, n, numberofunitcell, combs_fn=None):
    """Unit-cell Hamiltonian for cell `numberofunitcell`, impurities inserted."""
    l = int(l)
    if combs_fn is None:
        combs_fn = possible_combs(int(n), l)

    imps = combs_fn(int(config))
    x = imps[:, 0]          # cell indices
    y = imps[:, 1]          # site indices
    z = int(numberofunitcell)

    mat = unitcell(w, d, t, e, l)

    mask = (x == z)
    if not np.any(mask):
        return mat          # fast path: no impurity in this cell

    imp_indices = y[mask]
    mat[imp_indices, imp_indices] = (w + 1j * d - 0.5)
    return mat


# ----------------------------------------------------------------------
# 4) Transmission through the full device at one energy
# ----------------------------------------------------------------------
# Leads are loaded once per worker process (see _init_worker).
_LEADS = None
_SIZE = None


def _init_worker(leads_path, size):
    global _LEADS, _SIZE
    _LEADS = np.load(leads_path)
    _SIZE = int(size)


def device(w, d, t, e, l, config, n, leads=None):
    """Landauer transmission |Tr[...]| for one config at energy w."""
    l = int(l)
    if leads is None:
        leads = _LEADS

    ene = int(round(w * 100))
    left = leads[ene]                       # left lead surface Green's function
    hopp = np.eye(l, dtype=complex)         # inter-cell hopping = I_l
    iden = np.eye(l, dtype=complex)

    combs_fn = possible_combs(n, l)

    # Recursively attach the 100 device unit cells to the left lead.
    G = left
    for i in range(NCELLS):
        g_d = np.linalg.inv(unidevice(w, d, t, e, l, config, n, i, combs_fn=combs_fn))
        A = iden - g_d @ hopp @ G @ hopp
        G = np.linalg.solve(A, g_d)

    left_device = G
    right = left                            # symmetric right lead

    # Connect the assembled left device to the right lead.
    c_l = np.linalg.solve(iden - right @ hopp @ left_device @ hopp, left_device)
    c_r = np.linalg.solve(iden - left_device @ hopp @ right @ hopp, right)

    # Broadening / spectral matrices and the Landauer trace.
    G_ll = c_l - c_l.conj().T
    G_rr = c_r - c_r.conj().T
    G_lr = left_device @ hopp @ c_r
    Gnon = G_lr - G_lr.conj().T

    tr1 = G_ll @ hopp @ G_rr @ hopp - hopp @ Gnon @ hopp @ Gnon
    return np.abs(np.trace(tr1))


# ----------------------------------------------------------------------
# 5) Worker: full spectrum for one configuration
# ----------------------------------------------------------------------
def compute_for_one_config(args):
    cfg, conc, size, d, t, e = args
    w_vals = energy_grid()
    out = np.zeros(len(w_vals), dtype=float)
    for i, w in enumerate(w_vals):
        out[i] = device(w, d, t, e, size, cfg, conc)
    return out


# ----------------------------------------------------------------------
# 6) Parallel execution for ONE concentration
# ----------------------------------------------------------------------
def compute_for_concentration(conc, size, nconfigs, out_dir, leads_path,
                              d, t, e, n_jobs, resume):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[INFO] conc = {conc}  ->  {out_dir}")

    args = []
    for cfg in range(nconfigs):
        fn = os.path.join(out_dir, f"sq_size{size}_conc{conc}_cfg{cfg}.npy")
        if resume and os.path.exists(fn):
            continue
        args.append((cfg, conc, size, d, t, e))

    if not args:
        print(f"[SKIP] conc={conc}: all {nconfigs} configs already present.")
        return

    with Pool(processes=n_jobs, initializer=_init_worker,
              initargs=(leads_path, size)) as pool:
        for (cfg, *_), result in zip(
                args,
                tqdm(pool.imap(compute_for_one_config, args),
                     total=len(args), desc=f"conc {conc}")):
            fn = os.path.join(out_dir, f"sq_size{size}_conc{conc}_cfg{cfg}.npy")
            np.save(fn, result)


# ----------------------------------------------------------------------
# 7) Driver
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate square-lattice transmission spectra for many "
                    "impurity configurations.")
    parser.add_argument("--size", type=int, default=10,
                        help="Unit-cell width l (default 10).")
    parser.add_argument("--concs", type=int, nargs="+", default=None,
                        help="Impurity concentrations (default: 5 10 15 ... 50).")
    parser.add_argument("--nconfigs", type=int, default=10000,
                        help="Configurations per concentration (default 10000).")
    parser.add_argument("--leads", type=str, default=None,
                        help="Path to leads_{size}.npy (overrides --leads-dir).")
    parser.add_argument("--leads-dir", type=str, default="~/transmissions_sq/leads",
                        help="Directory holding leads_{size}.npy.")
    parser.add_argument("--out-dir", type=str, default="~/transmissions_sq",
                        help="Base output directory; a size_{l}_conc_{c} subdir "
                             "is created per concentration.")
    parser.add_argument("--d", type=float, default=1e-3,
                        help="Imaginary broadening eta for the device (default 1e-3).")
    parser.add_argument("--t", type=float, default=1.0, help="Hopping amplitude.")
    parser.add_argument("--e", type=float, default=0.0, help="On-site energy offset.")
    parser.add_argument("--n-jobs", type=int, default=max(1, cpu_count() - 1),
                        help="Worker processes (default: cpu_count - 1).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs whose output file already exists.")
    args = parser.parse_args()

    size = args.size
    concs = args.concs if args.concs is not None else list(range(5, 51, 5))

    leads_path = (os.path.expanduser(args.leads) if args.leads
                  else os.path.join(os.path.expanduser(args.leads_dir),
                                    f"leads_{size}.npy"))
    if not os.path.exists(leads_path):
        raise FileNotFoundError(
            f"Leads not found: {leads_path}\n"
            f"Run:  python compute_leads_sq.py --sizes {size}")

    base_out = os.path.expanduser(args.out_dir)

    print(f"[INFO] size      : {size}")
    print(f"[INFO] concs     : {concs}")
    print(f"[INFO] nconfigs  : {args.nconfigs}")
    print(f"[INFO] leads     : {leads_path}")
    print(f"[INFO] out_dir   : {base_out}")
    print(f"[INFO] params    : d={args.d}, t={args.t}, e={args.e}")
    print(f"[INFO] cpu cores : {cpu_count()}  (using {args.n_jobs})")

    for conc in concs:
        out_dir = os.path.join(base_out, f"size_{size}_conc_{conc}")
        compute_for_concentration(conc, size, args.nconfigs, out_dir, leads_path,
                                  args.d, args.t, args.e, args.n_jobs, args.resume)

    print("\n[DONE] all concentrations complete.")


if __name__ == "__main__":
    main()
