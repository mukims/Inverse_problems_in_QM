"""Build Shazam's ensemble lookup (LOOKUP-1): catalogue from the graphene and new-material stores, calibrate, save."""
import argparse
import os
import time
from pathlib import Path

import paths
from atlaslib import CloudStore, InputSpec
from atlaslib.lookup import Catalogue, calibrate, calibrate_per_material
from tbribbon.materials import make_model

HERE = paths.results_dir("material_atlas", create=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(paths.data_root()), help="the ribbon catalogue (graphene and all other materials)")
    ap.add_argument("--out", default=str(HERE / "lookup_v4"))
    a = ap.parse_args()
    t0 = time.time()
    store = CloudStore(a.store)
    models = []
    for mid in store.models(with_clouds=True):
        mat, edge, n = mid.split("/")
        if mat != "square":          # the tree also holds a square strip: not a catalogued device
            models.append(make_model(mat, edge, int(n[1:])))
    cat = Catalogue.build(store, models, InputSpec(version="v4"))
    print(f"[lookup] {len(cat.models)} devices, grid {cat.grid[0]}-{cat.grid[-1]} ({cat.grid.size} points), "
          f"{len(cat.val_x)} validation spectra, {time.time() - t0:.0f}s", flush=True)
    cal = calibrate(cat)
    print(f"[lookup] kappa {cal['kappa']}: validation coverage {cal['coverage']:.3f} on {cal['n']} spectra", flush=True)
    per = calibrate_per_material(cat)
    for mat, v in per.items():
        print(f"[lookup]   {mat}: interval kappa {v['kappa']}, validation coverage {v['coverage']:.3f} on {v['n']} spectra",
              flush=True)
    cat.settings = dict(cat.settings, calibration=cal, calibration_per_material=per)
    cat.save(a.out)
    print(f"[lookup] saved {a.out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
