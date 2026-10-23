"""Evaluate the all-ribbons autoencoder (AE-1) against PCA and the raw spectrum, on seeds 850-999 only.

    python notebooks/material_atlas/eval_ae.py --runs notebooks/material_atlas/ae_v1/L32 .../L64 .../L128
Tests: (1) reconstruction against PCA at the same latent size; (2) nearest-reference identification, the Shazam vote,
in the latent space, the PCA space and the raw 416 channels; (3) novelty of the unseen square strip.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from shazam import ae_data
from atlaslib import CloudStore, Conv1dAE, embed
from atlaslib.energy import on_axis
from atlaslib.lookup import window_inputs
from tbribbon.materials import MATERIALS

HERE = Path(__file__).resolve().parent
K = 15


def knn(refs, queries, k=K, chunk=512):
    """Indices and Euclidean distances of the k nearest references of every query."""
    R = torch.from_numpy(np.ascontiguousarray(refs, dtype=np.float32))
    rn = (R * R).sum(1)
    idx, dist = [], []
    for i in range(0, len(queries), chunk):
        Q = torch.from_numpy(np.ascontiguousarray(queries[i:i + chunk], dtype=np.float32))
        d2 = (Q * Q).sum(1, keepdim=True) - 2 * Q @ R.T + rn
        v, j = torch.topk(d2.clamp_min(0), k, largest=False)
        idx.append(j.numpy())
        dist.append(v.sqrt().numpy())
    return np.concatenate(idx), np.concatenate(dist)


def vote(labels):
    """Most common label among each row's neighbours (lowest label wins a tie)."""
    return np.array([np.bincount(r).argmax() for r in labels])


def standardise(refs, *others):
    mu, sd = refs.mean(0), refs.std(0) + 1e-9
    return [(x - mu) / sd for x in (refs, *others)]


def load_model(run):
    ck = torch.load(Path(run) / "encoder.pt", map_location="cpu", weights_only=True)
    m = Conv1dAE(ck["latent"], ck["seq_len"])
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def per_group(values, groups):
    return {g: float(np.mean(values[groups == g])) for g in np.unique(groups)}


def identify(spaces, refs_meta, q_meta):
    """Device, material+edge and density answers of the 15-nearest vote, per space."""
    out = {}
    for name, (R, Q) in spaces.items():
        idx, dist = knn(R, Q)
        dev = vote(refs_meta["dev"][idx])
        mat = lambda d: np.array([refs_meta["devices"][i].split("/")[0] for i in d])
        me = lambda d: np.array(["/".join(refs_meta["devices"][i].split("/")[:2]) for i in d])
        right = dev == q_meta["dev"]
        dens = np.median(refs_meta["dens"][idx], axis=1)
        rel = np.abs(dens - q_meta["dens"]) / q_meta["dens"]
        out[name] = {"device_pct": 100 * float(right.mean()),
                     "material_pct": 100 * float((mat(dev) == mat(q_meta["dev"])).mean()),
                     "material_edge_pct": 100 * float((me(dev) == me(q_meta["dev"])).mean()),
                     "median_rel_density_err_pct": 100 * float(np.median(rel[right])) if right.any() else None,
                     "by_material_device_pct": {m_: 100 * float(right[mat(q_meta["dev"]) == m_].mean()) for m_ in sorted(set(mat(q_meta["dev"])))},
                     "mean_neighbour_distance": dist.mean(1)}
    return out


def novelty(R, known_val, known_test, square, k=K):
    """Mean distance to the k nearest references: threshold at the 99th percentile of validation scores."""
    score = lambda Q: knn(R, Q, k)[1].mean(1)
    sv, st, ss = score(known_val), score(known_test), score(square)
    thr = np.percentile(sv, 99)
    return {"auroc": float(roc_auc_score(np.r_[np.zeros(len(st)), np.ones(len(ss))], np.r_[st, ss])),
            "square_flagged_pct": 100 * float((ss > thr).mean()), "known_flagged_pct": 100 * float((st > thr).mean())}


def square_strip(root="~/atlas_store/novelty_v1"):
    store = CloudStore(Path(root).expanduser())
    (mid,) = [m for m in store.models() if m.startswith("square/")]
    e_t, _ = store.read_pristine(mid)
    e_ev, _ = on_axis(ae_data.SPEC, type("M", (), {"model_id": mid, "t_ev": MATERIALS["square"]["t_ev"]})(), e_t)
    X = []
    for d in store.densities(mid):
        X.append(window_inputs(ae_data.SPEC, e_ev, store.read_cloud(mid, d)[0])[0])      # unseen lattice: all 600 are test
    return np.concatenate(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--refs-per-cloud", type=int, default=200)
    ap.add_argument("--queries-per-cloud", type=int, default=25)
    ap.add_argument("--pca-fit", type=int, default=200000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda s: print(f"[eval-ae] {s} ({time.time() - t0:.0f}s)", flush=True)
    rp, qp = (5, 3) if a.smoke else (a.refs_per_cloud, a.queries_per_cloud)
    test = ae_data.collect("test", per_cloud=3 if a.smoke else None)
    refs = ae_data.collect("train", per_cloud=rp, include=("catalogue",))
    keep = refs["seed"] >= 0                                  # a reference is a disordered spectrum
    refs = {k: (v[keep] if k != "devices" else v) for k, v in refs.items()}
    cat_val = ae_data.collect("val", per_cloud=qp, include=("catalogue",))
    cat_test = ae_data.collect("test", per_cloud=qp, include=("catalogue",))
    assert list(refs["devices"]) == list(cat_test["devices"]), "reference and query device lists differ"
    square = square_strip()
    fit = refs["x"][np.random.default_rng(0).choice(len(refs["x"]), min(a.pca_fit, len(refs["x"])), replace=False)]
    log(f"test {len(test['x']):,}; references {len(refs['x']):,}; queries {len(cat_test['x']):,}; square {len(square)}")
    results = {}
    for run in a.runs:
        model = load_model(run)
        L = model.to_latent.out_features
        pca = PCA(L, svd_solver="randomized", random_state=0).fit(fit)
        Z_te, err_ae = embed(model, test["x"])
        err_pca = ((pca.inverse_transform(pca.transform(test["x"])) - test["x"]) ** 2).mean(1)
        src = np.array(ae_data.SOURCES)[test["src"]]
        mat = np.array([test["devices"][i].split("/")[0] for i in test["dev"]])
        spaces = {"autoencoder": [embed(model, x)[0] for x in (refs["x"], cat_test["x"], cat_val["x"], square)],
                  "pca": [pca.transform(x) for x in (refs["x"], cat_test["x"], cat_val["x"], square)],
                  "raw": [refs["x"], cat_test["x"], cat_val["x"], square]}
        spaces = {n: standardise(*v) if n != "raw" else v for n, v in spaces.items()}
        ident = identify({n: (v[0], v[1]) for n, v in spaces.items()}, refs, cat_test)
        nov = {n: novelty(v[0], v[2], v[1], v[3]) for n, v in spaces.items()}
        for n in ident:
            ident[n].pop("mean_neighbour_distance")
        results[run] = {"latent": L, "n_test": int(len(test["x"])),
                        "reconstruction_mse": {"autoencoder": float(err_ae.mean()), "pca": float(err_pca.mean()),
                                               "autoencoder_by_source": per_group(err_ae, src), "pca_by_source": per_group(err_pca, src),
                                               "autoencoder_by_material": per_group(err_ae, mat), "pca_by_material": per_group(err_pca, mat)},
                        "identification": ident, "novelty_square": nov}
        log(f"{run}: L{L} mse ae {err_ae.mean():.6f} pca {err_pca.mean():.6f}; device "
            + ", ".join(f"{n} {v['device_pct']:.2f}%" for n, v in ident.items()) + "; square AUROC "
            + ", ".join(f"{n} {v['auroc']:.4f}" for n, v in nov.items()))
        (Path(run) / ("results_smoke.json" if a.smoke else "results.json")).write_text(json.dumps(results[run], indent=2))
    return results


if __name__ == "__main__":
    main()
