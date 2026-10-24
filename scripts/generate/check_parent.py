#!/usr/bin/env python
"""Validity checks of the parent-lattice training set (spec section 5), adapted from check_store.py.

    PYTHONPATH=src python scripts/generate/check_parent.py [--root DIR]

Checks: all stored values are finite; no two spectra are identical; for every draw the disordered spectrum does not
exceed the clean open-channel count except within the known numerical-spike tolerance (check_seed_excess).
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

import paths
from shazam.inverse.data import shard_paths
from tbribbon.check_store import check_seed_excess

TOL = 0.05                  # same tolerance as check_store.py
DEFAULT_ROOT = paths.data_root() / "_generated" / "hexagonal_parent_v1"


def check(root):
    """{finite, unique, max_seed_excess, ok, n_draws, n_duplicates, n_over_tol} for all shards under root."""
    finite, seen, n_dup, n_draws, n_over, worst = True, set(), 0, 0, 0, -np.inf
    for path in shard_paths(root):
        with np.load(path) as z:
            a = {k: z[k] for k in z.files}                 # each array is decompressed once
        finite &= all(bool(np.isfinite(v).all()) for v in a.values() if v.dtype.kind == "f")
        for row in a["x"]:
            h = hashlib.md5(np.ascontiguousarray(row).tobytes()).digest()
            n_dup += h in seen
            seen.add(h)
        for raw, clean in zip(a["raw"], a["clean"]):
            excess, ok = check_seed_excess(raw[None, :].astype(float), clean.astype(float), tol=TOL)
            worst, n_over = max(worst, excess), n_over + (not ok)
        n_draws += len(a["index"])
    if n_draws == 0:
        raise FileNotFoundError(f"no shard_*.npz under {root}")
    return dict(finite=finite, unique=n_dup == 0, max_seed_excess=float(worst), n_draws=n_draws, n_duplicates=n_dup,
                n_over_tol=n_over, ok=bool(finite and n_dup == 0 and n_over == 0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    a = ap.parse_args(argv)
    res = check(a.root.expanduser())
    for k, v in res.items():
        print(f"{k:>16}: {v}")
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
