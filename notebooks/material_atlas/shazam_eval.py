"""Evaluation for Shazam on several materials: identification, a hidden material, and the clean-spectrum ground truth."""
from collections import Counter

import numpy as np

from atlaslib.energy import on_axis
from meaning.distances import rms_distance, shape_profile


def _locate_test(atlas, store, reg, mid, test_seed_min):
    m = reg.get(mid)
    e_t, _ = store.read_pristine(mid)
    out = []
    for d in store.densities(mid):
        c, s = store.read_cloud(mid, d)
        c = c[s >= test_seed_min]
        if len(c):
            out += atlas.locate(c, *on_axis(atlas.spec, m, e_t, m.band_top_t))
    return out


def _pct(flags):
    return round(100.0 * float(np.mean(flags)), 3) if len(flags) else float("nan")


def _median_z(loc):
    z = [r.novelty_z for r in loc if r.novelty_z is not None]
    return round(float(np.median(z)), 3) if z else None


def identify(atlas, store, reg, ids, test_seed_min=850):
    per_model, by_mat, by_group = {}, {}, {}
    for mid in ids:
        m = reg.get(mid)
        loc = _locate_test(atlas, store, reg, mid, test_seed_min)
        row = dict(material_accuracy=_pct([r.material == m.material for r in loc]),
                   edge_accuracy=_pct([r.edge == m.edge for r in loc]),
                   width_accuracy=_pct([round(r.width_vote) == m.width for r in loc]),
                   unknown_pct=_pct([r.unknown for r in loc]),
                   median_z=_median_z(loc), n=len(loc))
        per_model[mid] = row
        by_mat.setdefault(m.material, []).append(row)
        by_group.setdefault(f"{m.material}/{m.edge}", []).append(row)

    def pool(rows):
        n = sum(r["n"] for r in rows)
        keys = ("material_accuracy", "edge_accuracy", "width_accuracy", "unknown_pct")
        return {k: round(sum(r[k] * r["n"] for r in rows) / n, 3) for k in keys} | {"n": n}

    return {"per_model": per_model,
            "per_material": {k: pool(v) for k, v in by_mat.items()},
            "unknown_pct_by_group": {k: pool(v)["unknown_pct"] for k, v in by_group.items()}}


def query_group(atlas, store, reg, ids, test_seed_min=850):
    per = {mid: _locate_test(atlas, store, reg, mid, test_seed_min) for mid in ids}
    loc = [r for mid in ids for r in per[mid]]
    n = len(loc)

    def share(keys):
        return {k: round(100.0 * v / n, 3) for k, v in Counter(keys).most_common()}

    z = np.array([r.novelty_z for r in loc if r.novelty_z is not None], float)
    ratio = np.array([r.novelty_ratio for r in loc])
    return {"n": n, "unknown_pct": _pct([r.unknown for r in loc]),
            "nearest_material": share(r.material for r in loc),
            "nearest_material_edge": share(f"{r.material}/{r.edge}" for r in loc),
            "nearest_model_top3": [(k, round(100.0 * v / n, 3)) for k, v in Counter(r.nearest_model for r in loc).most_common(3)],
            "z_quantiles": [round(float(q), 3) for q in np.percentile(z, [10, 50, 90])] if len(z) else [None] * 3,
            "ratio_quantiles": [round(float(q), 3) for q in np.percentile(ratio, [10, 50, 90])],
            "per_model_nearest_material": {mid: Counter(r.material for r in rs).most_common(1)[0][0]
                                           for mid, rs in per.items() if rs}}


def clean_ground_truth(store, reg, spec, ids):
    vec_ev, vec_shape = {}, {}
    for mid in ids:
        m = reg.get(mid)
        e_t, pris = store.read_pristine(mid)
        e_ax, top = on_axis(spec, m, e_t, m.band_top_t)
        vec_ev[mid] = spec.to_input(pris[None], e_ax, top)[0]
        vec_shape[mid] = shape_profile(np.round(pris, 3), e_ax)
    out = {}
    for mid in ids:
        mat = reg.get(mid).material
        others = [o for o in ids if reg.get(o).material != mat]
        row = {}
        for name, vec in (("nearest_eV", vec_ev), ("nearest_shape", vec_shape)):
            d = {o: rms_distance(vec[mid], vec[o]) for o in others}
            best = min(d, key=d.get)
            row[name] = {"model": best, "material": reg.get(best).material, "distance": round(d[best], 4)}
        out[mid] = row
    return out
