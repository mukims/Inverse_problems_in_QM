"""MEANING-2 (BUILD-25): graded similarity across materials, one hidden material at a time.

Encoders per hidden material M: AE (atlas_v5m/loo_M), physics (BUILD-18 single sigma), multiscale, stress.
Run from the repo root:
  PYTHONPATH=notebooks/material_atlas:notebooks ~/miniconda3/envs/ml/bin/python -u -m meaning.run_meaning_v2 [--smoke]
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from atlaslib import Atlas, CloudStore, MultiStore
from atlaslib.encoder import embed
from atlaslib.energy import on_axis
from meaning.contrastive import embed_structure, train_structure_encoder
from meaning.distances import clean_distance_matrix, distances_to, kernel_sigma
from meaning.metrics import evaluate_held_out
from meaning.split import check_same_ribbons, seed_split
from tbribbon.materials import make_model

HERE = Path(__file__).resolve().parent
V5M = HERE.parent / "atlas_v5m"
TAU, K_PER_CLASS, SCALES, LAM = 0.1, 16, (1.0, 4.0, 16.0), 1.0


def load_all(spec, store, ids):
    data = {}
    for mid in ids:
        mat, edge, n = mid.split("/")
        m = make_model(mat, edge, int(n[1:]))
        e_t, pris = store.read_pristine(mid)
        e_ax, top = on_axis(spec, m, e_t, m.band_top_t)
        X, S = [], []
        for d in store.densities(mid):
            c, s = store.read_cloud(mid, d)
            X.append(spec.to_input(c, e_ax, top))
            S.append(s)
        data[mid] = dict(X=np.concatenate(X).astype(np.float32), seed=np.concatenate(S),
                         clean=spec.to_input(pris[None], e_ax, top)[0], material=mat)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="200 steps, 300 references per ribbon, first hidden material only")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--graphene-store", default="~/atlas_store/engine_v1")
    ap.add_argument("--materials-store", default="~/atlas_store/materials_ev_full")
    ap.add_argument("--novelty-store", default="~/atlas_store/novelty_v1")
    a = ap.parse_args()
    t0 = time.time()
    steps, n_refs = (200, 300) if a.smoke else (2500, 2000)
    out = HERE / "results" / ("v2_smoke" if a.smoke else "v2")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)

    full = Atlas.load(V5M)
    spec = full.spec
    store = MultiStore(os.path.expanduser(a.graphene_store), os.path.expanduser(a.materials_store))
    all_ids = [m.model_id for m in full.models]
    data = load_all(spec, store, all_ids)
    nov = CloudStore(os.path.expanduser(a.novelty_store))
    sq = make_model("square", "strip", 10)
    e_sq, p_sq = nov.read_pristine(sq.model_id)
    e_ax, top = on_axis(spec, sq, e_sq, sq.band_top_t)
    X_sq = np.concatenate([spec.to_input(nov.read_cloud(sq.model_id, d)[0], e_ax, top)
                           for d in nov.densities(sq.model_id)]).astype(np.float32)
    clean_sq = spec.to_input(p_sq[None], e_ax, top)[0]

    new_materials = sorted({d["material"] for d in data.values()} - {"graphene-ideal"})
    summary = {}
    for M in (new_materials[:1] if a.smoke else new_materials):
        loo = Atlas.load(V5M / f"loo_{M}")
        train_ids = [m.model_id for m in loo.models]
        check_same_ribbons(train_ids, [i for i in all_ids if data[i]["material"] != M])
        hidden_ids = [i for i in all_ids if data[i]["material"] == M]
        split = {i: seed_split(data[i]["seed"]) for i in all_ids}
        X_train = [data[i]["X"][split[i][0]] for i in train_ids]
        X_val = [data[i]["X"][split[i][1]] for i in train_ids]
        rng = np.random.default_rng(7)
        refs = [X[rng.permutation(len(X))[:n_refs]] for X in X_train]
        known = [data[i]["X"][split[i][2]] for i in train_ids]
        held = {i: data[i]["X"][split[i][2]] for i in hidden_ids}
        held["square/strip/N10"] = X_sq
        C = np.stack([data[i]["clean"] for i in train_ids])
        D = clean_distance_matrix(C)
        sigma0, alpha = kernel_sigma(D), 1.5 / float(D.max())
        clean_dist = {i: distances_to(data[i]["clean"], C) for i in hidden_ids}
        clean_dist["square/strip/N10"] = distances_to(clean_sq, C)
        materials = [data[i]["material"] for i in train_ids]

        def ae(X, m=loo):
            return (embed(m.encoder, X)[0] - m.mu) / m.sd

        rows = [dict(model="AE (atlas_v5m loo)",
                     **evaluate_held_out(ae, refs, known, held, clean_dist, train_ids, materials))]
        for mode, kw in (("physics", dict(sigma=sigma0)),
                         ("multiscale", dict(sigma=[sigma0 * s for s in SCALES])),
                         ("stress", dict(alpha=alpha, lam=LAM))):
            model, history = train_structure_encoder(X_train, X_val, mode, D=D, steps=steps, k=K_PER_CLASS,
                                                     tau=TAU, threads=a.threads, **kw)
            torch.save(model.state_dict(), out / f"encoder_{M}_{mode}.pt")
            rows.append(dict(model=mode, history=history,
                             **evaluate_held_out(lambda X, m=model: embed_structure(m, X),
                                                 refs, known, held, clean_dist, train_ids, materials)))

        def summarise(r):
            g = r["groups"]
            rib = [g[i] for i in hidden_ids]
            return {"known_identification_pct": r["known_identification_pct"],
                    "median_spearman_hidden_ribbons": round(float(np.median([x["spearman_vs_clean"] for x in rib])), 3),
                    "material_agreement_pct": round(100.0 * float(np.mean([x["material_agrees"] for x in rib])), 1),
                    "min_auroc_vs_known": min(x["auroc_vs_known"] for x in rib),
                    "square_spearman": g["square/strip/N10"]["spearman_vs_clean"],
                    "square_auroc_vs_known": g["square/strip/N10"]["auroc_vs_known"]}

        summary[M] = {r["model"]: summarise(r) for r in rows}
        json.dump(dict(hidden=M, results=rows, sigma0=sigma0, alpha=alpha, scales=SCALES, lam=LAM, steps=steps,
                       refs_per_ribbon=n_refs, train_ribbons=train_ids),
                  open(out / f"{M}.json", "w"), indent=2)
        print(f"[meaning-v2] hidden {M}: {json.dumps(summary[M])}", flush=True)

    json.dump(dict(summary=summary, runtime_s=round(time.time() - t0)), open(out / "summary.json", "w"), indent=2)
    print(f"[meaning-v2] done in {time.time() - t0:.0f}s -> {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
