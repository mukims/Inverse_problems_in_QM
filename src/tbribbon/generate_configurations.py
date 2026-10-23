#!/usr/bin/env python
"""Generate and save cloud_<key>_config.npz alongside every cloud in CloudStores.

Multi-processed, parallel, resumable.
"""
import argparse
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np

import paths
from atlaslib import CloudStore


def _process_one(args):
    store_path, model_id, density, force, unit = args
    store = CloudStore(store_path)
    if not force and store.has_configuration_cloud(model_id, density):
        p = store._dir(model_id) / f"cloud_{store._meta(model_id)['clouds'][f'd{density:.4f}']['formula'] if False else ''}" # just checking exists
        return model_id, density, 0, 0.0
    t0 = time.time()
    out_p = store.write_configuration_cloud(model_id, density, force=force, unit=unit)
    dt = time.time() - t0
    return model_id, density, out_p.stat().st_size, dt


def generate_configurations_for_store(store_path, n_jobs=8, force=False, unit="lattice"):
    store = CloudStore(Path(store_path).expanduser())
    tasks = []
    for mid in sorted(store.models()):
        for d in store.densities(mid):
            tasks.append((str(store.root), mid, d, force, unit))

    print(f"Store {store.root.name}: {len(tasks)} clouds to process with {n_jobs} workers...", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    done = 0
    total_bytes = 0
    with ctx.Pool(n_jobs) as p:
        for mid, d, sz, dt in p.imap_unordered(_process_one, tasks, chunksize=8):
            done += 1
            total_bytes += sz
            if done % 100 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / (elapsed + 1e-6)
                print(f"  [{done}/{len(tasks)}] clouds processed ({rate:.1f} clouds/s, written {total_bytes/1024/1024:.1f} MB)", flush=True)

    print(f"Finished {store.root.name} in {time.time() - t0:.1f}s.\n", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate cloud_<key>_config.npz alongside clouds")
    ap.add_argument("--stores", default=str(paths.data_root()),
                    help="Comma-separated store paths")
    ap.add_argument("--n-jobs", type=int, default=8, help="Number of worker processes")
    ap.add_argument("--force", action="store_true", help="Overwrite existing config.npz files")
    ap.add_argument("--unit", choices=["lattice", "angstrom"], default="lattice")
    args = ap.parse_args()

    for sp in [s.strip() for s in args.stores.split(",") if s.strip()]:
        generate_configurations_for_store(sp, n_jobs=args.n_jobs, force=args.force, unit=args.unit)
