"""Scan z_arm values with z_zig=3.350 to see test line behavior."""

import numpy as np
from atlaslib.store import CloudStore
from atlaslib import Atlas

store = CloudStore("/home/shardul/atlas_store/engine_v1")
atlas = Atlas.load("notebooks/material_atlas/atlas_v2")

val_data = {}
test_data = {}
for m in atlas.models:
    mid = m.model_id
    e_t, _ = store.read_pristine(mid)
    for d in store.densities(mid):
        c, s = store.read_cloud(mid, d)
        v = (s >= 700) & (s <= 849)
        t = (s >= 850)
        loc_v = atlas.locate(c[v], e_t, m.band_top_t)
        loc_t = atlas.locate(c[t], e_t, m.band_top_t)
        val_data[(mid, f"{d:.4f}", m.edge)] = [r.novelty_s for r in loc_v]
        test_data[(mid, f"{d:.4f}", m.edge)] = [(r.novelty_s, r.density) for r in loc_t]

class_stats = {}
for (mid, d_str, edge), s_vals in val_data.items():
    log_s = np.log(np.maximum(s_vals, 1e-12))
    c_val = float(np.median(log_s))
    w_val = 1.4826 * float(np.median(np.abs(log_s - c_val)))
    if w_val < 1e-8:
        w_val = 1.0
    class_stats[(mid, d_str, edge)] = (c_val, w_val)

for z_arm in [2.985, 3.05, 3.10, 3.144, 3.20, 3.25, 3.30, 3.35]:
    table = {}
    for (mid, d_str, edge), (c_val, w_val) in class_stats.items():
        z_use = z_arm if edge == "armchair" else 3.350
        tau = float(np.exp(c_val + z_use * w_val))
        table.setdefault(mid, {})[d_str] = tau

    counts = []
    lines = []
    for (mid, d_str, edge), items in test_data.items():
        d_map = table[mid]
        avail_d = [float(k) for k in d_map.keys()]
        fa = 0
        for s, pred_d in items:
            snapped_d = min(avail_d, key=lambda x: abs(x - pred_d))
            tau = d_map[f"{snapped_d:.4f}"]
            if s > tau:
                fa += 1
        counts.append(fa)
        lines.append((mid, d_str, edge, fa))

    counts = np.array(counts)
    arm_counts = [x[3] for x in lines if x[2] == "armchair"]
    zig_counts = [x[3] for x in lines if x[2] == "zigzag"]
    high_arm = [f"{x[0].split('/')[-1]}-d{x[1]}:{x[3]}" for x in lines if x[2] == "armchair" and x[3] >= 8]
    high_zig = [f"{x[0].split('/')[-1]}-d{x[1]}:{x[3]}" for x in lines if x[2] == "zigzag" and x[3] >= 8]
    arm_pct = np.sum(arm_counts) / (len(arm_counts) * 150) * 100
    zig_pct = np.sum(zig_counts) / (len(zig_counts) * 150) * 100
    tot_pct = np.sum(counts) / (len(counts) * 150) * 100
    print(f"z_arm={z_arm:.3f}: tot={tot_pct:.2f}%, arm={arm_pct:.2f}%, zig={zig_pct:.2f}%, max_arm={max(arm_counts)}, max_zig={max(zig_counts)}, high_arm={high_arm}")
