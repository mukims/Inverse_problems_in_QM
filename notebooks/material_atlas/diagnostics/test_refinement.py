"""Accurate and fast evaluation of n0 scan matching atlas.locate logic."""

import numpy as np
from atlaslib.store import CloudStore
from atlaslib import Atlas

store = CloudStore("/home/shardul/atlas_store/engine_v1")
atlas = Atlas.load("notebooks/material_atlas/atlas_v2")

# Cache locate info for all classes:
# for each spectrum: target_mid, pred_dens, s (novelty_s)
print("Extracting forward pass info for all 124 classes...")

val1_cache = {}  # 700..774 (75 seeds)
val2_cache = {}  # 775..849 (75 seeds)
val_all_cache = {}  # 700..849 (150 seeds)
test_cache = {}  # 850..999 (150 seeds)

for m in atlas.models:
    mid = m.model_id
    e_t, _ = store.read_pristine(mid)
    for d in store.densities(mid):
        c, s = store.read_cloud(mid, d)
        v1 = (s >= 700) & (s <= 774)
        v2 = (s >= 775) & (s <= 849)
        vall = (s >= 700) & (s <= 849)
        tst = (s >= 850) & (s <= 999)

        loc1 = atlas.locate(c[v1], e_t, m.band_top_t)
        loc2 = atlas.locate(c[v2], e_t, m.band_top_t)
        loc_all = atlas.locate(c[vall], e_t, m.band_top_t)
        loc_tst = atlas.locate(c[tst], e_t, m.band_top_t)

        key = (mid, f"{d:.4f}", m.edge)
        val1_cache[key] = [(r.material, r.edge, r.width_vote, r.density, r.novelty_s) for r in loc1]
        val2_cache[key] = [(r.material, r.edge, r.width_vote, r.density, r.novelty_s) for r in loc2]
        val_all_cache[key] = [(r.material, r.edge, r.width_vote, r.density, r.novelty_s) for r in loc_all]
        test_cache[key] = [(r.material, r.edge, r.width_vote, r.density, r.novelty_s) for r in loc_tst]

print("Extraction complete. Running n0 calibration...")

def calibrate(data_cache, n_weight, n0):
    # Step 1: Per class c and w
    class_stats = {}
    edge_w = {}
    for (mid, d_str, edge), items in data_cache.items():
        s_vals = np.array([x[4] for x in items])
        log_s = np.log(np.maximum(s_vals, 1e-12))
        c = float(np.median(log_s))
        mad = float(np.median(np.abs(log_s - c)))
        w = 1.4826 * mad
        if w < 1e-8:
            w = 1.0
        class_stats[(mid, d_str, edge)] = (c, w)
        edge_w.setdefault(edge, []).append(w)

    # Step 2: Edge scale
    w_edge = {e: float(np.median(ws)) for e, ws in edge_w.items()}

    # Step 3: Shrunk w'
    w_prime = {}
    for (mid, d_str, edge), (c, w) in class_stats.items():
        wp = (float(n_weight) * w + n0 * w_edge[edge]) / (float(n_weight) + n0)
        w_prime[(mid, d_str, edge)] = wp

    # Step 4: Per-edge tail z*
    edge_z = {}
    for (mid, d_str, edge), items in data_cache.items():
        s_vals = np.array([x[4] for x in items])
        log_s = np.log(np.maximum(s_vals, 1e-12))
        c, _ = class_stats[(mid, d_str, edge)]
        wp = w_prime[(mid, d_str, edge)]
        z = (log_s - c) / wp
        edge_z.setdefault(edge, []).extend(z)

    z_star_edge = {e: float(np.percentile(zs, 99)) for e, zs in edge_z.items()}

    # Step 5: Threshold table tau(model, density)
    table = {}
    params = {}
    for (mid, d_str, edge), (c, _) in class_stats.items():
        wp = w_prime[(mid, d_str, edge)]
        z_star = z_star_edge[edge]
        tau = float(np.exp(c + z_star * wp))
        table.setdefault(mid, {})[d_str] = tau
        params.setdefault(mid, {})[d_str] = {"c": c, "w": wp}

    return table, params, w_edge, z_star_edge

def evaluate_fa(eval_cache, table):
    """Evaluate false alarms using exact locate() snapping logic."""
    counts = []
    counts_by_edge = {}
    line_details = []
    for (mid, d_str, edge), items in eval_cache.items():
        d_map = table[mid]
        avail_d = [float(k_d) for k_d in d_map.keys()]
        line_fa = 0
        for mat, r_edge, width_vote, pred_dens, s in items:
            snapped_d = min(avail_d, key=lambda x: abs(x - pred_dens))
            snapped_key = f"{snapped_d:.4f}"
            tau = d_map[snapped_key]
            if s > tau:
                line_fa += 1
        counts.append(line_fa)
        counts_by_edge.setdefault(edge, []).append(line_fa)
        line_details.append((mid, d_str, edge, line_fa, len(items)))

    counts = np.array(counts)
    mean_c = np.mean(counts)
    var_c = np.var(counts, ddof=1)
    disp = var_c / (mean_c + 1e-12)
    max_c = np.max(counts)
    total_fa = np.sum(counts)
    total_n = sum(len(items) for items in eval_cache.values())
    rate = total_fa / total_n

    edge_rates = {}
    for e, e_counts in counts_by_edge.items():
        edge_rates[e] = np.sum(e_counts) / (len(e_counts) * len(items))

    return {
        "mean": mean_c,
        "var": var_c,
        "disp": disp,
        "max": max_c,
        "total_fa": total_fa,
        "total_n": total_n,
        "rate": rate,
        "edge_rates": edge_rates,
        "line_details": line_details,
        "counts": counts
    }

print("\n=== Validation Split Calibration (700..774) -> Evaluated on (775..849) ===")
print("n0  |   mean   |   var    | disp (var/mean) | max/75 | total FA (%) | armchair FA | zigzag FA")
print("-" * 85)
for n0 in [0, 25, 50, 100, 150, 300]:
    table, params, w_edge, z_star = calibrate(val1_cache, 75, n0)
    res = evaluate_fa(val2_cache, table)
    print(f"{n0:3d} | {res['mean']:8.4f} | {res['var']:8.4f} | {res['disp']:15.4f} | {res['max']:6d} | {res['total_fa']:4d}/{res['total_n']} ({res['rate']*100:.2f}%) | {res['edge_rates']['armchair']*100:.3f}% | {res['edge_rates']['zigzag']*100:.3f}%")

print("\n=== Recalibrate on Full Val (700..849, 150 seeds) -> Evaluate on Test (850..999) ===")
print("n0  | disp (var/mean) | max/150 | total FA (%) | armchair FA | zigzag FA | z*_arm | z*_zig")
print("-" * 90)
for n0 in [0, 25, 50, 100, 150, 300]:
    table, params, w_edge, z_star = calibrate(val_all_cache, 150, n0)
    res = evaluate_fa(test_cache, table)
    print(f"{n0:3d} | {res['disp']:15.4f} | {res['max']:7d} | {res['total_fa']:4d}/{res['total_n']} ({res['rate']*100:.2f}%) | {res['edge_rates']['armchair']*100:.3f}% | {res['edge_rates']['zigzag']*100:.3f}% | {z_star['armchair']:.3f}  | {z_star['zigzag']:.3f}")
    # Check lines with highest FA
    top_lines = sorted(res['line_details'], key=lambda x: x[3], reverse=True)[:3]
    top_str = ", ".join(f"{x[0].split('/')[-1]}-d{x[1]}: {x[3]}/150" for x in top_lines)
    print(f"     Top lines: {top_str}")
