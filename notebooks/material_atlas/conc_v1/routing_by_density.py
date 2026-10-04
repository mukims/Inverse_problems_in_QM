"""Evaluate Shazam routing breakdown by density level on CONC-1 pilot data.

Reruns atlas_v4m routing on held-out test seeds (850-999) across all 24 density levels
for the 5 pilot ribbons, recording routed %, unknown %, silently misread %, and median s/tau.
"""
import json
import os
from pathlib import Path
import numpy as np

from atlaslib import Atlas, CloudStore, Registry
from atlaslib.energy import on_axis
from stage3_generic import _load, split_masks


def main():
    atlas_path = Path("notebooks/material_atlas/atlas_v4m")
    store_path = Path(os.path.expanduser("~/atlas_store/conc_v1"))
    out_dir = Path("notebooks/material_atlas/conc_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "routing_by_density.json"

    atlas = Atlas.load(str(atlas_path))
    reg = Registry(atlas.models)
    store = CloudStore(store_path)

    results = {}
    for mid in store.models():
        m = reg.get(mid)
        e_t, pris, T, y, s = _load(store, mid)
        te = split_masks(s, train_max=699, val_max=849)[2]
        T_te, y_te = T[te], y[te]

        loc = atlas.locate(T_te, *on_axis(atlas.spec, m, e_t, m.band_top_t))
        unknown = np.array([r.unknown for r in loc])
        got = np.array([r.nearest_model for r in loc])
        ratio = np.array([r.novelty_ratio for r in loc])

        routed = (~unknown) & (got == mid)
        misread = (~unknown) & (got != mid)

        ribbon_report = {}
        for d in np.unique(y_te):
            mask = (y_te == d)
            ribbon_report[f"{d:.4f}"] = {
                "n": int(mask.sum()),
                "routed_pct": round(100.0 * float(routed[mask].mean()), 3),
                "unknown_pct": round(100.0 * float(unknown[mask].mean()), 3),
                "silently_misread_pct": round(100.0 * float(misread[mask].mean()), 3),
                "median_s_tau": round(float(np.median(ratio[mask])), 4),
            }

        results[mid] = ribbon_report
        print(f"Finished {mid}: {len(ribbon_report)} densities evaluated.")

    out_file.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
