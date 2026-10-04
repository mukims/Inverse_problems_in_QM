"""Build Shazam's ensemble lookup (LOOKUP-1): catalogue from the graphene and new-material stores, calibrate, save."""
import argparse
import os
import time
from pathlib import Path

from atlaslib import InputSpec, MultiStore
from atlaslib.lookup import Catalogue, calibrate
from tbribbon.materials import make_model

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphene-store", default="~/atlas_store/engine_v1")
    ap.add_argument("--materials-store", default="~/atlas_store/materials_ev_full")
    ap.add_argument("--out", default=str(HERE / "lookup_v1"))
    a = ap.parse_args()
    t0 = time.time()
    store = MultiStore(os.path.expanduser(a.graphene_store), os.path.expanduser(a.materials_store))
    models = []
    for mid in store.models():
        mat, edge, n = mid.split("/")
        if mat != "square":          # engine_v1 also holds a clean-only square strip: not a catalogued device
            models.append(make_model(mat, edge, int(n[1:])))
    cat = Catalogue.build(store, models, InputSpec(version="v4"))
    print(f"[lookup] {len(cat.models)} devices, grid {cat.grid[0]}-{cat.grid[-1]} ({cat.grid.size} points), "
          f"{len(cat.val_x)} validation spectra, {time.time() - t0:.0f}s", flush=True)
    cal = calibrate(cat)
    print(f"[lookup] kappa {cal['kappa']}: validation coverage {cal['coverage']:.3f} on {cal['n']} spectra", flush=True)
    cat.settings = dict(cat.settings, calibration=cal)
    cat.save(a.out)
    print(f"[lookup] saved {a.out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
