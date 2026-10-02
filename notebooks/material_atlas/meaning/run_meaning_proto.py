# notebooks/material_atlas/meaning/run_meaning_proto.py
"""BUILD-18 (MEANING-1): does a "meaning" objective give Shazam physically meaningful similarity?

Three encoders on the 29 graphene ribbons of atlas_v2_loo (armchair N13 and zigzag N8 held out):
  AE (atlas_v2_loo), paraphrase (supervised contrastive), physics (soft contrastive on clean-spectrum distance).
Read-only on the stores and on atlas_v2_loo.

Run from the repo root:
  PYTHONPATH=notebooks/material_atlas:notebooks ~/miniconda3/envs/ml/bin/python -u -m meaning.run_meaning_proto --smoke
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from atlaslib import Atlas, CloudStore, InputSpec
from atlaslib.encoder import embed
from meaning.contrastive import embed_structure, train_structure_encoder
from meaning.distances import clean_distance_matrix, distances_to, kernel_sigma
from meaning.metrics import evaluate_embedding
from meaning.split import check_same_ribbons, seed_split
from tbribbon.materials import make_model

HERE = Path(__file__).resolve().parent
GRID = {("graphene-ideal", "armchair"): [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 27, 31, 40, 50],
        ("graphene-ideal", "zigzag"): [4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 27, 40, 50]}
HELD = ["graphene-ideal/armchair/N13", "graphene-ideal/zigzag/N8"]
DENS = [0.005, 0.01, 0.02, 0.04]
TAU, K_PER_CLASS = 0.1, 16


def load(spec, eng, nov):
    data = {}
    for (mat, edge), widths in GRID.items():
        for n in widths:
            m = make_model(mat, edge, n)
            e_t, pris = eng.read_pristine(m.model_id)
            X, S = [], []
            for d in eng.densities(m.model_id):
                c, s = eng.read_cloud(m.model_id, d)
                X.append(spec.to_input(c, e_t, m.band_top_t))
                S.append(s)
            data[m.model_id] = dict(X=np.concatenate(X).astype(np.float32), seed=np.concatenate(S),
                                    clean=spec.to_input(pris[None], e_t, m.band_top_t)[0], name=f"{edge} N{n}")
    sq = make_model("square", "strip", 10)
    e_sq, pris_sq = nov.read_pristine(sq.model_id)
    X_sq = np.concatenate([spec.to_input(nov.read_cloud(sq.model_id, d)[0], e_sq, None) for d in DENS])
    return data, X_sq.astype(np.float32), spec.to_input(pris_sq[None], e_sq, None)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="200 steps and 300 references per ribbon")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    steps, n_refs = (200, 300) if a.smoke else (2500, 2000)
    out = HERE / "results" / ("smoke" if a.smoke else "full")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)

    spec = InputSpec(version="v2")
    data, X_sq, clean_sq = load(spec, CloudStore(os.path.expanduser("~/atlas_store/engine_v1")),
                                CloudStore(os.path.expanduser("~/atlas_store/novelty_v1")))
    train_ids = [k for k in data if k not in HELD]
    loo = Atlas.load(HERE.parent / "atlas_v2_loo")
    check_same_ribbons([m.model_id for m in loo.models], train_ids)

    names = [data[k]["name"] for k in train_ids]
    split = {k: seed_split(data[k]["seed"]) for k in data}
    X_train = [data[k]["X"][split[k][0]] for k in train_ids]
    X_val = [data[k]["X"][split[k][1]] for k in train_ids]
    rng = np.random.default_rng(7)
    refs = [X[rng.permutation(len(X))[:n_refs]] for X in X_train]
    known = [data[k]["X"][split[k][2]] for k in train_ids]
    untrained = {data[h]["name"]: data[h]["X"][split[h][2]] for h in HELD}
    unseen = {"square N10": X_sq}

    C = np.stack([data[k]["clean"] for k in train_ids])
    D = clean_distance_matrix(C)
    sigma = kernel_sigma(D)
    clean_dist = {data[h]["name"]: distances_to(data[h]["clean"], C) for h in HELD}
    clean_dist["square N10"] = distances_to(clean_sq, C)

    def ae(X):
        return (embed(loo.encoder, X)[0] - loo.mu) / loo.sd

    results = [dict(model="AE (atlas_v2_loo)",
                    **evaluate_embedding(ae, refs, known, untrained, unseen, clean_dist, names))]
    print(json.dumps(results[-1]), flush=True)
    for mode in ("paraphrase", "physics"):
        model, history = train_structure_encoder(
            X_train, X_val, mode, D=D if mode == "physics" else None, sigma=sigma if mode == "physics" else None,
            steps=steps, k=K_PER_CLASS, tau=TAU, threads=a.threads)
        torch.save(model.state_dict(), out / f"encoder_{mode}.pt")
        results.append(dict(model=mode, history=history,
                            **evaluate_embedding(lambda X, m=model: embed_structure(m, X),
                                                 refs, known, untrained, unseen, clean_dist, names)))
        print(json.dumps({k: v for k, v in results[-1].items() if k != "history"}), flush=True)

    physics_nearest = {n: [(names[i], round(float(d[i]), 3)) for i in np.argsort(d)[:4]]
                       for n, d in clean_dist.items()}
    json.dump(dict(results=results, physics_nearest_by_clean_spectrum=physics_nearest, sigma=sigma,
                   steps=steps, refs_per_ribbon=n_refs, tau=TAU, batch=K_PER_CLASS * len(train_ids),
                   runtime_s=round(time.time() - t0)), open(out / "results.json", "w"), indent=2)
    print(f"[meaning] done in {time.time() - t0:.0f}s -> {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
