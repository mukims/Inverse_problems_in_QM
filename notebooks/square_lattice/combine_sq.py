#!/usr/bin/env python
"""
combine_sq.py
=============
Merge the per-config transmission files written by ``ca_sq.py`` into one
stacked array per concentration, mirroring the 7-AGNR combined format
described in ``README_COMBINED.md``.

Input  (produced by ca_sq.py):
    <in_dir>/size_{l}_conc_{c}/sq_size{l}_conc{c}_cfg{cfg}.npy   # shape (400,)

Output:
    <out_dir>/conc_{c}.npy        # shape (num_configs, 400), float64
    <out_dir>/conc_{c}_meta.csv   # row_idx,config  (maps each row -> its cfg seed)

Rows are ordered by config index so row i corresponds to ``cfg == meta[i]``
(missing configs are simply skipped, and their absence is recorded by the
meta file rather than by leaving gaps).

Options
-------
--clip-to PATH   Optional pristine spectrum (.npy, shape (400,)). If given,
                 each spectrum is clipped to [0, pristine] before stacking,
                 matching the ``np.clip(transmission, 0, pris)`` step used in
                 CA.ipynb. Omit to keep raw transmissions.

Examples
--------
    python combine_sq.py --size 10 --in-dir ~/transmissions_sq \
        --out-dir ~/transmissions_sq/size_10_combined

    python combine_sq.py --size 25 --concs 10 20 30 \
        --in-dir ~/transmissions_sq --clip-to ~/transmissions_sq/pristine_25.npy
"""

import os
import re
import glob
import argparse

import numpy as np
from tqdm import tqdm

N_ENERGIES = 400


def find_concentrations(in_dir, size):
    """Auto-detect concentrations from size_{l}_conc_{c} subdirectories."""
    pat = re.compile(rf"size_{size}_conc_(\d+)$")
    concs = []
    for name in os.listdir(in_dir):
        m = pat.match(name)
        if m and os.path.isdir(os.path.join(in_dir, name)):
            concs.append(int(m.group(1)))
    return sorted(concs)


def config_index(path):
    """Extract the integer cfg index from ...conc{c}_cfg{cfg}.npy."""
    m = re.search(r"_cfg(\d+)\.npy$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def combine_concentration(in_dir, out_dir, size, conc, pristine=None):
    conc_dir = os.path.join(in_dir, f"size_{size}_conc_{conc}")
    files = glob.glob(os.path.join(conc_dir, f"sq_size{size}_conc{conc}_cfg*.npy"))
    if not files:
        print(f"[SKIP] conc={conc}: no files in {conc_dir}")
        return

    files.sort(key=config_index)          # order rows by config seed
    configs = [config_index(f) for f in files]

    rows = np.empty((len(files), N_ENERGIES), dtype=np.float64)
    for i, f in enumerate(tqdm(files, desc=f"conc {conc}", leave=False)):
        spec = np.load(f)
        if spec.shape != (N_ENERGIES,):
            raise ValueError(f"{f} has shape {spec.shape}, expected ({N_ENERGIES},)")
        if pristine is not None:
            spec = np.clip(spec, 0.0, pristine)
        rows[i] = spec

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"conc_{conc}.npy"), rows)
    meta = np.column_stack((np.arange(len(configs)), np.asarray(configs)))
    np.savetxt(os.path.join(out_dir, f"conc_{conc}_meta.csv"), meta,
               delimiter=",", header="row_idx,config", comments="", fmt="%d")
    print(f"[OK]   conc={conc}: {rows.shape} -> conc_{conc}.npy")


def main():
    parser = argparse.ArgumentParser(
        description="Combine per-config square-lattice transmissions into "
                    "per-concentration stacks.")
    parser.add_argument("--size", type=int, required=True, help="Unit-cell width l.")
    parser.add_argument("--in-dir", type=str, default="~/transmissions_sq",
                        help="Base dir holding size_{l}_conc_{c} subdirs.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output dir (default: <in-dir>/size_{l}_combined).")
    parser.add_argument("--concs", type=int, nargs="+", default=None,
                        help="Concentrations to combine (default: auto-detect).")
    parser.add_argument("--clip-to", type=str, default=None,
                        help="Optional pristine spectrum .npy to clip each row to [0, pris].")
    args = parser.parse_args()

    in_dir = os.path.expanduser(args.in_dir)
    out_dir = (os.path.expanduser(args.out_dir) if args.out_dir
               else os.path.join(in_dir, f"size_{args.size}_combined"))

    concs = args.concs if args.concs is not None else find_concentrations(in_dir, args.size)
    if not concs:
        raise SystemExit(f"No concentrations found for size {args.size} under {in_dir}")

    pristine = None
    if args.clip_to:
        pristine = np.load(os.path.expanduser(args.clip_to))
        if pristine.shape != (N_ENERGIES,):
            raise ValueError(f"pristine has shape {pristine.shape}, expected ({N_ENERGIES},)")

    print(f"[INFO] size    : {args.size}")
    print(f"[INFO] in_dir  : {in_dir}")
    print(f"[INFO] out_dir : {out_dir}")
    print(f"[INFO] concs   : {concs}")
    print(f"[INFO] clip    : {'yes (' + args.clip_to + ')' if pristine is not None else 'no (raw)'}")

    for conc in concs:
        combine_concentration(in_dir, out_dir, args.size, conc, pristine=pristine)

    print("\n[DONE] combine complete.")


if __name__ == "__main__":
    main()
