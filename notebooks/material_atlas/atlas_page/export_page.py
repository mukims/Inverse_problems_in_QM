"""Export a page-sized copy of a Shazam map for the in-browser atlas page (PAGE-1)."""
import json
from pathlib import Path

import numpy as np

from atlaslib.energy import on_axis

ENCODER_ORDER = ["encoder.0.weight", "encoder.0.bias", "encoder.1.weight", "encoder.1.bias",
                 "encoder.3.weight", "encoder.3.bias", "encoder.4.weight", "encoder.4.bias",
                 "encoder.6.weight", "encoder.6.bias", "encoder.7.weight", "encoder.7.bias",
                 "to_latent.weight", "to_latent.bias"]
CONVS = [{"Cout": 32, "K": 7, "stride": 2, "pad": 3}, {"Cout": 64, "K": 5, "stride": 2, "pad": 2},
         {"Cout": 128, "K": 3, "stride": 2, "pad": 1}]
FIELDS = ("material", "edge", "width_vote", "density", "unknown", "novelty_ratio", "nearest_model")
COMPARED = ("material", "edge", "width_vote", "unknown", "nearest_model")


def _answer(r):
    return {f: (bool(getattr(r, f)) if f == "unknown" else getattr(r, f)) for f in FIELDS}


def export_page(atlas, store, reg, ids, out_dir, refs_per_model=500, n_test=300, val=(700, 849),
                test_seed_min=850, extra=None, seed=0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    page = atlas.with_reference_cap(refs_per_model, seed=seed)
    page.calibrate_novelty(store, reg, ids, val_seed_min=val[0], max_seed=val[1], group_by="material_edge")
    spec = page.spec

    sd = page.encoder.state_dict()
    np.concatenate([sd[k].numpy().ravel() for k in ENCODER_ORDER]).astype("<f4").tofile(out / "encoder.bin")
    page.refs.astype("<f4").tofile(out / "refs.bin")
    page.ref_model.astype("<u2").tofile(out / "ref_model.bin")
    page.ref_density.astype("<f4").tofile(out / "ref_density.bin")
    mean = page.refs.mean(0)
    _, _, vt = np.linalg.svd(page.refs - mean, full_matrices=False)

    clean, median, tests = {}, {}, []
    rng = np.random.default_rng(seed)
    agree, total = {f: 0 for f in COMPARED}, 0
    for mid in ids:
        m = reg.get(mid)
        e_t, pris = store.read_pristine(mid)
        e_ax, top = on_axis(spec, m, e_t, m.band_top_t)
        clean[mid] = np.round(spec.to_input(pris[None], e_ax, top)[0], 5).tolist()
        median[mid] = {}
        for d in store.densities(mid):
            c, s = store.read_cloud(mid, d)
            median[mid][f"{d:.4f}"] = np.round(np.median(spec.to_input(c, e_ax, top), 0), 5).tolist()
            ct = c[s >= test_seed_min]
            if len(ct):
                pick = ct[rng.permutation(len(ct))[:5]]
                a_full, a_page = atlas.locate(pick, e_ax, top), page.locate(pick, e_ax, top)
                for rf, rp in zip(a_full, a_page):
                    total += 1
                    for f in COMPARED:
                        agree[f] += getattr(rf, f) == getattr(rp, f)
                tests += [(e_ax, row, top) for row in pick]
    tests = [tests[i] for i in rng.permutation(len(tests))[:n_test]] + list(extra or [])
    vectors = []
    for e_ax, row, top in tests:
        x = spec.to_input(row[None], e_ax, top)
        from atlaslib.encoder import embed
        z = (embed(page.encoder, x)[0] - page.mu) / page.sd
        vectors.append({"e_ev": np.round(e_ax, 6).tolist(), "T": np.round(row, 6).tolist(),
                        "band_top_ev": None if top is None else float(top), "x": np.round(x[0], 6).tolist(),
                        "zs": np.round(z[0], 6).tolist(), "answer": _answer(page.locate(row[None], e_ax, top)[0])})

    meta = {"spec": spec.as_dict(), "k": page.k, "n_refs": int(len(page.refs)),
            "models": [{"id": m.model_id, "material": m.material, "edge": m.edge, "width": m.width,
                        "t_ev": m.t_ev, "band_top_ev": round(m.band_top_t * m.t_ev, 4)} for m in page.models],
            "mu": page.mu.astype(float).tolist(), "sd": page.sd.astype(float).tolist(),
            "pca": {"mean": mean.astype(float).tolist(), "components": vt[:2].astype(float).tolist()},
            "threshold_table": page.threshold_table, "threshold_params": page.threshold_params,
            "z_star": page.z_star, "encoder": {"latent": int(page.mu.size), "convs": CONVS},
            "clean": clean, "median": median,
            "agreement_with_full_map": {f: round(100.0 * v / max(total, 1), 2) for f, v in agree.items()} | {"n": total}}
    (out / "model.json").write_text(json.dumps(meta))
    (out / "test_vectors.json").write_text(json.dumps(vectors))
    return meta
