#!/usr/bin/env python
"""BUILD-23 / BUILD-24: Shazam on graphene + new materials; leave-one-material-out closest lattice and closeness.

--mode frozen (BUILD-23): atlas_v4's graphene-trained encoder; new materials appended as references.
--mode joint  (BUILD-24): encoder retrained on all materials; every hidden-material map is retrained without it.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from atlaslib import Atlas, CloudStore, MultiStore, Registry
from build_atlas_v2 import GRIDS
from shazam_eval import clean_ground_truth, identify, query_group
from tbribbon.materials import make_model

TRAIN_MAX, VAL_MIN, VAL_MAX, TEST_MIN = 699, 700, 849, 850


def registry(mat_store):
    reg = Registry()
    for (mat, edge), widths in GRIDS["sparse31"].items():
        for n in widths:
            reg.add(make_model(mat, edge, n))
    new_ids = []
    for mid in mat_store.models():
        mat, edge, n = mid.split("/")
        reg.add(make_model(mat, edge, int(n[1:])))
        new_ids.append(mid)
    return reg, new_ids


def check_seeds(store, ids):
    bad = [(m, d) for m in ids for d in store.densities(m)
           if not np.array_equal(np.sort(store.read_cloud(m, d)[1]), np.arange(1000))]
    if bad:
        raise SystemExit(f"clouds without seeds 0-999 (seed split impossible): {bad[:5]} ...")


def calibrated(atlas, store, reg, ids):
    atlas.calibrate_novelty(store, reg, ids, val_seed_min=VAL_MIN, max_seed=VAL_MAX, group_by="material_edge")
    return atlas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["frozen", "joint"], required=True)
    ap.add_argument("--atlas", default="notebooks/material_atlas/atlas_v4")
    ap.add_argument("--graphene-store", default="~/atlas_store/engine_v1")
    ap.add_argument("--materials-store", default="~/atlas_store/materials_ev_full")
    ap.add_argument("--novelty-store", default="~/atlas_store/novelty_v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    t0, out = time.time(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    mat_store = CloudStore(os.path.expanduser(a.materials_store))
    store = MultiStore(os.path.expanduser(a.graphene_store), mat_store)
    reg, new_ids = registry(mat_store)
    graphene_ids = [i for i in reg.ids() if i not in new_ids]
    all_ids = graphene_ids + new_ids
    check_seeds(store, new_ids)
    materials = sorted({reg.get(i).material for i in new_ids})
    base = Atlas.load(a.atlas)
    spec = base.spec

    if a.mode == "frozen":
        atlas = base
        atlas.add_models(store, reg, new_ids, max_seed=TRAIN_MAX, refs_per_model=2000)
    else:
        atlas = Atlas.build(store, reg, all_ids, spec, threads=a.threads, max_seed=VAL_MAX, val_seed_min=VAL_MIN)
    calibrated(atlas, store, reg, all_ids)
    atlas.save(out)
    print(f"[shazam] {a.mode} map: {len(atlas.models)} models, n0={atlas.n0}, z*={atlas.z_star}", flush=True)

    ident = identify(atlas, store, reg, all_ids, test_seed_min=TEST_MIN)
    print(json.dumps(ident["per_material"], indent=1), flush=True)

    nov = CloudStore(os.path.expanduser(a.novelty_store))
    sq = make_model("square", "strip", 10)
    reg_sq = Registry([sq])
    square = query_group(atlas, nov, reg_sq, [sq.model_id], test_seed_min=0)

    lomo = {}
    for mat in materials:
        hidden_ids = [i for i in new_ids if reg.get(i).material == mat]
        if a.mode == "frozen":
            hidden = atlas.without_models(hidden_ids)
        else:
            keep = [i for i in all_ids if i not in hidden_ids]
            hidden = Atlas.build(store, reg, keep, spec, threads=a.threads, max_seed=VAL_MAX, val_seed_min=VAL_MIN)
            calibrated(hidden, store, reg, keep)
            hidden.save(out / f"loo_{mat}")
        lomo[mat] = query_group(hidden, store, reg, hidden_ids, test_seed_min=TEST_MIN)
        print(f"[shazam] hidden {mat}: unknown {lomo[mat]['unknown_pct']}%, nearest {lomo[mat]['nearest_material']}, "
              f"z50 {lomo[mat]['z_quantiles'][1]}", flush=True)

    gt = clean_ground_truth(store, reg, spec, all_ids)
    agree = {}
    for mat in materials:
        rib = lomo[mat]["per_model_nearest_material"]
        agree[mat] = {"eV": round(100.0 * np.mean([rib[i] == gt[i]["nearest_eV"]["material"] for i in rib]), 1),
                      "shape": round(100.0 * np.mean([rib[i] == gt[i]["nearest_shape"]["material"] for i in rib]), 1)}

    res = dict(identification=ident, square=square, lomo=lomo, clean_ground_truth=gt, agreement=agree,
               settings=dict(mode=a.mode, base_atlas=a.atlas, spec=spec.as_dict(), materials_store=a.materials_store,
                             seed_split=[TRAIN_MAX, VAL_MIN, VAL_MAX, TEST_MIN], group_by="material_edge",
                             n0=atlas.n0, z_star=atlas.z_star, runtime_s=round(time.time() - t0)))
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(f"[shazam] done in {time.time() - t0:.0f}s -> {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
