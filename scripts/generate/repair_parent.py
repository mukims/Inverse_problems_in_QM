#!/usr/bin/env python
"""Re-simulate draws of a generated parent-lattice folder whose stored spectrum is not finite, and patch them in place.

Shards written before the lead fallback (tbribbon.leads.RETRY_SHIFTS) can hold a NaN channel where the lead decimation
overflowed. Each such draw is regenerated from its index with the current code (same prior draw, same disorder seed);
every stored field except the spectrum (`x`, `raw`) and `seconds` must come out identical, or the repair stops. The
shard is rewritten atomically and its manifest entry records what was patched.

    PYTHONPATH=src python scripts/generate/repair_parent.py --root "$TRANSMISSION_DATA/_generated/hexagonal_parent_v1"
"""
import argparse
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from shazam.inverse.runio import atomic_write_bytes, code_version

_spec = importlib.util.spec_from_file_location("generate_parent", Path(__file__).resolve().parent / "generate_parent.py")
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)

SPECTRUM = ("x", "raw")
ALLOWED_TO_CHANGE = SPECTRUM + ("seconds",)


def nonfinite_rows(arrays):
    return np.flatnonzero(~np.all(np.isfinite(arrays["raw"]), axis=1) | ~np.all(np.isfinite(arrays["x"]), axis=1))


def repair_shard(path, manifest_entry):
    """Patch the non-finite rows of one shard; returns the patched draw indices."""
    with np.load(path) as z:
        arrays = {k: z[k].copy() for k in z.files}
    rows = nonfinite_rows(arrays)
    if not len(rows):
        return []
    fixed = []
    for r in rows:
        index = int(arrays["index"][r])
        with gp._limit_blas() or nullcontext():         # one BLAS thread, as in the generator's workers
            new = gp._record(index)                     # the generator's own per-draw record, current code
        for k, v in new.items():
            if k in ALLOWED_TO_CHANGE or k not in arrays:
                continue
            if not np.array_equal(np.asarray(v, dtype=arrays[k].dtype), arrays[k][r]):
                raise SystemExit(f"draw {index}: field {k!r} changed on regeneration; refusing to patch")
        for k in SPECTRUM:
            arrays[k][r] = np.asarray(new[k], dtype=arrays[k].dtype)
        if not np.all(np.isfinite(arrays["raw"][r])):
            raise SystemExit(f"draw {index}: still not finite after regeneration")
        fixed.append(index)
    atomic_write_bytes(path, lambda f: np.savez(f, **arrays))
    note = dict(indices=fixed, fields=list(SPECTRUM), reason="non-finite lead Green's function (pre-fallback code)",
                code_version=code_version())
    manifest_entry.setdefault("patched", []).append(note)
    return fixed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True)
    a = ap.parse_args(argv)
    manifest = json.loads((a.root / "manifest.json").read_text())
    by_name = {f"shard_{e['shard']:05d}.npz": e for e in manifest["shards"]}
    total = []
    for path in sorted(a.root.glob("shard_*.npz")):
        fixed = repair_shard(path, by_name[path.name])
        if fixed:
            print(f"{path.name}: patched draws {fixed}")
            total += fixed
    if total:
        atomic_write_bytes(a.root / "manifest.json", lambda f: f.write(json.dumps(manifest, indent=2).encode()))
    print(f"patched {len(total)} draw(s)")


if __name__ == "__main__":
    main()
