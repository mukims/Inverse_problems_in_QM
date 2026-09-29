#!/usr/bin/env python
"""Atlas v2 from engine clouds; generalisation test on held-out widths."""
import argparse
import json
from pathlib import Path

import numpy as np

from atlaslib import Atlas, CloudStore, InputSpec, Registry
from tbribbon.materials import make_model

HELD_OUT = {("graphene-ideal", "armchair"): [8, 12, 13], ("graphene-ideal", "zigzag"): [8]}
TRAIN = {("graphene-ideal", "armchair"): range(5, 17), ("graphene-ideal", "zigzag"): range(4, 13)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="~/atlas_store/engine_v1")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "atlas_v2"))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    store, spec, out = CloudStore(a.store), InputSpec(), Path(a.out)
    reg = Registry()
    for (mat, edge), widths in TRAIN.items():
        for n in widths:
            reg.add(make_model(mat, edge, n))
    train_ids = [m.model_id for m in reg if m.width not in HELD_OUT[(m.material, m.edge)]]
    atlas = Atlas.build(store, reg, train_ids, spec, threads=a.threads)
    atlas.save(out)
    res = {}
    for (mat, edge), widths in HELD_OUT.items():
        for n in widths:
            mid = f"{mat}/{edge}/N{n}"
            T = np.concatenate([store.read_cloud(mid, d)[0] for d in store.densities(mid)])
            e_t, _ = store.read_pristine(mid)
            loc = atlas.locate(T, e_t, reg.get(mid).band_top_t)
            w = np.array([r.width for r in loc])
            res[mid] = {"material_accuracy": float(np.mean([r.material == mat for r in loc]) * 100),
                        "edge_accuracy": float(np.mean([r.edge == edge for r in loc]) * 100),
                        "median_width": float(np.median(w)),
                        "between_neighbours_pct": float(np.mean((w > n - 1.5) & (w < n + 1.5)) * 100)}
    (out / "generalisation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
