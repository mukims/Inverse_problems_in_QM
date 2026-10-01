"""Scan n0 values for novelty scale shrinkage on validation seeds (700-774 vs 775-849)."""

import numpy as np
from atlaslib.store import CloudStore
from atlaslib import Atlas

store = CloudStore("/home/shardul/atlas_store/engine_v1")
atlas = Atlas.load("notebooks/material_atlas/atlas_v2")

val1_data = {}
val2_data = {}

for m in atlas.models:
    mid = m.model_id
    e_t, _ = store.read_pristine(mid)
    for d in store.densities(mid):
        c, s = store.read_cloud(mid, d)
        v1 = (s >= 700) & (s <= 774)
        v2 = (s >= 775) & (s <= 849)
        loc1 = atlas.locate(c[v1], e_t, m.band_top_t)
        loc2 = atlas.locate(c[v2], e_t, m.band_top_t)
        s1 = np.array([r.novelty_s for r in loc1])
        s2 = np.array([r.novelty_s for r in loc2])
        d_str = f"{d:.4f}"
        val1_data[(mid, d_str, m.edge)] = s1
        val2_data[(mid, d_str, m.edge)] = s2

print(f"Collected {len(val1_data)} classes.")

# 1. Per class on v1 (75 seeds): c and w
c_map = {}
w_map = {}
edge_w = {}
for (mid, d_str, edge), s1 in val1_data.items():
    log_s = np.log(np.maximum(s1, 1e-12))
    c = float(np.median(log_s))
    mad = float(np.median(np.abs(log_s - c)))
    w = 1.4826 * mad
    if w < 1e-8:
        w = 1.0
    c_map[(mid, d_str, edge)] = c
    w_map[(mid, d_str, edge)] = w
    edge_w.setdefault(edge, []).append(w)

w_edge = {e: float(np.median(ws)) for e, ws in edge_w.items()}
print("w_edge (median per edge on v1):", w_edge)

# Try n0 in {0, 25, 50, 100, 150, 300}
# Calibration set has n_v1 = 75 samples per class.
# Formula: w' = (n_cal * w + n0 * w_edge) / (n_cal + n0)
# Notice in step 3 of refinement plan:
# w' = (150 * w + n0 * w_edge) / (150 + n0) when n_cal = 150.
# So for calibration set with 75 seeds, n_cal = 75.
results = []
for n0 in [0, 25, 50, 100, 150, 300]:
    w_prime = {}
    for (mid, d_str, edge), w in w_map.items():
        wp = (75.0 * w + n0 * w_edge[edge]) / (75.0 + n0)
        w_prime[(mid, d_str, edge)] = wp

    # Per-edge standardized tail z on v1
    edge_z = {}
    for (mid, d_str, edge), s1 in val1_data.items():
        log_s = np.log(np.maximum(s1, 1e-12))
        z = (log_s - c_map[(mid, d_str, edge)]) / w_prime[(mid, d_str, edge)]
        edge_z.setdefault(edge, []).extend(z)

    z_star_edge = {e: float(np.percentile(zs, 99)) for e, zs in edge_z.items()}

    # Threshold per class
    tau_map = {}
    for (mid, d_str, edge) in val1_data.keys():
        tau = np.exp(c_map[(mid, d_str, edge)] + z_star_edge[edge] * w_prime[(mid, d_str, edge)])
        tau_map[(mid, d_str)] = tau

    # False alarms on v2 (75 seeds per class)
    counts = []
    for (mid, d_str, edge), s2 in val2_data.items():
        fa = int(np.sum(s2 > tau_map[(mid, d_str)]))
        counts.append(fa)

    counts = np.array(counts)
    mean_c = np.mean(counts)
    var_c = np.var(counts, ddof=1)
    disp = var_c / (mean_c + 1e-12)
    max_c = np.max(counts)
    total_fa = np.sum(counts)
    total_samples = len(counts) * 75
    rate = total_fa / total_samples
    print(f"n0={n0:3d}: z*={z_star_edge}, mean={mean_c:.4f}, var={var_c:.4f}, disp={disp:.4f}, max={max_c}/75, total={total_fa}/{total_samples} ({rate*100:.3f}%)")
    results.append((n0, mean_c, var_c, disp, max_c, total_fa, rate, z_star_edge))
