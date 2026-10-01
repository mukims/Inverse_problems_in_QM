"""Evaluate n0 scan using exact atlas.locate calls on test set."""

import numpy as np
from atlaslib.store import CloudStore
from atlaslib import Atlas

store = CloudStore("/home/shardul/atlas_store/engine_v1")
atlas = Atlas.load("notebooks/material_atlas/atlas_v2")

# Cache locate info for test set so we don't repeat the CNN+kNN forward pass 6 times
print("Extracting test embeddings...")
test_results = {}
for m in atlas.models:
    mid = m.model_id
    e_t, _ = store.read_pristine(mid)
    for d in store.densities(mid):
        c, s = store.read_cloud(mid, d)
        t = (s >= 850)
        loc = atlas.locate(c[t], e_t, m.band_top_t)
        test_results[(mid, f"{d:.4f}", m.edge)] = [(r.novelty_s, r.density, r.unknown_recon, r.model_id if hasattr(r, 'model_id') else mid) for r in loc]

print("Collecting validation scores...")
val_data = {}
for m in atlas.models:
    mid = m.model_id
    e_t, _ = store.read_pristine(mid)
    for d in store.densities(mid):
        c, s = store.read_cloud(mid, d)
        v = (s >= 700) & (s <= 849)
        loc = atlas.locate(c[v], e_t, m.band_top_t)
        s_vals = np.array([r.novelty_s for r in loc])
        val_data[(mid, f"{d:.4f}", m.edge)] = s_vals

print("Running exact evaluations across n0...")
for n0 in [0, 25, 50, 100, 150, 300]:
    # c, w
    class_stats = {}
    edge_w = {}
    for (mid, d_str, edge), s_vals in val_data.items():
        log_s = np.log(np.maximum(s_vals, 1e-12))
        c_val = float(np.median(log_s))
        w_val = 1.4826 * float(np.median(np.abs(log_s - c_val)))
        if w_val < 1e-8:
            w_val = 1.0
        class_stats[(mid, d_str, edge)] = (c_val, w_val)
        edge_w.setdefault(edge, []).append(w_val)

    w_edge = {e: float(np.median(ws)) for e, ws in edge_w.items()}
    w_prime = {}
    for (mid, d_str, edge), (c_val, w_val) in class_stats.items():
        wp = (150.0 * w_val + n0 * w_edge[edge]) / (150.0 + n0)
        w_prime[(mid, d_str, edge)] = wp

    edge_z = {}
    for (mid, d_str, edge), s_vals in val_data.items():
        log_s = np.log(np.maximum(s_vals, 1e-12))
        c_val, _ = class_stats[(mid, d_str, edge)]
        wp = w_prime[(mid, d_str, edge)]
        z = (log_s - c_val) / wp
        edge_z.setdefault(edge, []).extend(z)

    z_star_edge = {e: float(np.percentile(zs, 99)) for e, zs in edge_z.items()}

    # Update atlas threshold_table
    table = {}
    for (mid, d_str, edge), (c_val, _) in class_stats.items():
        wp = w_prime[(mid, d_str, edge)]
        tau = float(np.exp(c_val + z_star_edge[edge] * wp))
        table.setdefault(mid, {})[d_str] = tau

    test_counts = []
    lines_ge_8 = []
    edge_counts = {"armchair": [], "zigzag": []}

    for (mid, d_str, edge), items in test_results.items():
        d_map = table[mid]
        avail_d = [float(k) for k in d_map.keys()]
        fa = 0
        for s, pred_d, unk_recon, _ in items:
            snapped_d = min(avail_d, key=lambda x: abs(x - pred_d))
            snapped_key = f"{snapped_d:.4f}"
            tau = d_map[snapped_key]
            if s > tau:
                fa += 1
        test_counts.append(fa)
        edge_counts[edge].append(fa)
        if fa >= 8:
            short_mid = mid.split("/")[-1]
            lines_ge_8.append(f"{short_mid}-d{d_str}:{fa}")

    test_counts = np.array(test_counts)
    disp = np.var(test_counts, ddof=1) / np.mean(test_counts)
    arm_fa = sum(edge_counts["armchair"]) / (len(edge_counts["armchair"]) * 150)
    zig_fa = sum(edge_counts["zigzag"]) / (len(edge_counts["zigzag"]) * 150)
    total_fa = sum(test_counts)
    print(f"n0={n0:3d}: disp={disp:.3f}, max={max(test_counts):2d}/150, arm_fa={arm_fa*100:.2f}%, zig_fa={zig_fa*100:.2f}%, total={total_fa}/18600 ({total_fa/186:.2f}%), lines>=8: {lines_ge_8}")
