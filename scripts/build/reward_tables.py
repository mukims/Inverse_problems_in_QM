"""Band weights, physics scales and truth-cost tables for reward fine-tuning (spec
docs/superpowers/specs/2026-11-04-reward-finetuning-design.md, sections 4 and 2).

    TRANSMISSION_DATA=... PYTHONPATH=src nice -n 10 python scripts/build/reward_tables.py --jobs 22

For the train and val splits: every draw's band weights (parent_data.band_weights, one BLAS thread per worker), the
physics scales fitted on train only, the (n, 156) label and physics family-cost tables, and the (156, 156) state-state
physics table used to decode by minimum expected cost. Also the CHK check of section 2: for val CHK truths that the
reference model reads as a two-type lattice, the band weight of the dropped type, against the weakest type's weight of
the CHK truths it reads right. Writes into results/inverse_design/reward_tables/ (the .npz tables are not tracked).
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch

import paths
from shazam.inverse import data, reward as rw
from shazam.inverse.encoding import STATES, encode
from shazam.inverse.model import InverseModel
from shazam.inverse.runio import atomic_write_json, code_version, sha256_of
from tbribbon.parent_data import to_input_bandtop

DEFAULT_DATA = paths.data_root() / "_generated" / "hexagonal_parent_v1"
DEFAULT_OUT = paths.results_dir("inverse_design", "reward_tables")
DEFAULT_MODEL = paths.results_dir("inverse_design", "model_v2_depth_bandtop_600k") / "model.pt"
NEEDED = ("index", "on", "lead", "params", "edge", "width", "density")


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def band_weights_of(draws, n_jobs=1):
    """(n, 3) band weights (C, H, K; 0 for an absent type) of the draws' hosts, in order."""
    params = [d.params for d in draws]
    if n_jobs <= 1:
        rows = [rw.band_weight_row(p) for p in params]
    else:
        with mp.get_context("fork").Pool(n_jobs, initializer=_init_worker) as pool:
            rows = pool.map(rw.band_weight_row, params, chunksize=256)        # importable: picklable by name
    return np.asarray(rows, dtype=np.float64).reshape(len(params), 3)


def build_tables(draws, weights, scales):
    """{"label": (n, 156), "physics": (n, 156)} float32 family-cost tables of the draws."""
    states = np.array([encode(d)[0] for d in draws])
    return dict(label=rw.label_family_cost_table(states), physics=rw.physics_family_cost_table(draws, weights, scales))


def chk_check(model, raw, draws, weights):
    """Band weight of the type dropped when a CHK truth is read as a two-type lattice, against the weakest type's weight
    when it is read right (quartiles)."""
    with torch.no_grad():
        x = torch.as_tensor(to_input_bandtop(raw))
        logits = [model.discrete(model.encode(x[i:i + 2048])).masked_fill(~model.mask_known, -torch.inf)
                  for i in range(0, len(x), 2048)]
        top = torch.cat(logits).argmax(1).numpy()
    dropped, weakest_right = [], []
    for d, w, k in zip(draws, weights, top):
        if d.params.pattern != (True, True, True):
            continue
        pred = STATES[k].pattern
        if sum(pred) == 2:
            dropped.append(float(w[[not on for on in pred].index(True)]))
        elif pred == (True, True, True):
            weakest_right.append(float(w.min()))
    q = lambda v: [float(x) for x in np.quantile(v, [0.25, 0.5, 0.75])] if v else None
    return dict(n_read_two_type=len(dropped), dropped_type_weight_quartiles=q(dropped),
                n_read_right=len(weakest_right), weakest_type_weight_quartiles_when_right=q(weakest_right))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="the reference model, for the CHK check")
    ap.add_argument("--jobs", type=int, default=22)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    loaded = {}
    for split in ("train", "val"):
        arrays = data.load_split(a.data, split, NEEDED + (("raw",) if split == "val" else ()))
        draws = data.draws(arrays, slice(None))
        weights = band_weights_of(draws, a.jobs)
        np.savez(out / f"{split}_band_weights.npz", index=arrays["index"], weights=weights)
        loaded[split] = (arrays, draws, weights)
        print(f"[tables] {split}: {len(draws)} band weights ({time.time() - t0:.0f} s)", flush=True)
    scales = rw.fit_physics_scales(loaded["train"][1], loaded["train"][2])
    atomic_write_json(out / "physics_scales.json", dict(scales.to_json(), code_version=code_version()))
    np.save(out / "state_state_physics.npy", rw.state_state_physics_table(scales))
    for split, (arrays, draws, weights) in loaded.items():
        tables = build_tables(draws, weights, scales)
        for arm, cost in tables.items():
            np.savez(out / f"{split}_cost_{arm}.npz", index=arrays["index"], cost=cost.astype(np.float16))
        print(f"[tables] {split}: cost tables ({time.time() - t0:.0f} s)", flush=True)
    arrays, draws, weights = loaded["val"]
    check = chk_check(InverseModel.load(a.model), arrays["raw"], draws, weights)
    atomic_write_json(out / "chk_check.json", dict(check, model=str(a.model), model_sha256=sha256_of(a.model),
                                                   code_version=code_version()))
    print("[tables] CHK check:", json.dumps(check), flush=True)
    print(f"[tables] wrote {out} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
