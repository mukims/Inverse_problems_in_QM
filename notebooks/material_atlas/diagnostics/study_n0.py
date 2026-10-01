"""Comprehensive study of n0 scan on validation and test sets."""

import numpy as np
from atlaslib.store import CloudStore
from atlaslib import Atlas

store = CloudStore("/home/shardul/atlas_store/engine_v1")
atlas = Atlas.load("notebooks/material_atlas/atlas_v2")

val1_data = {}  # 700..774 (75 seeds)
val2_data = {}  # 775..849 (75 seeds)
val_all_data = {}  # 700..849 (150 seeds)
test_data = {}  # 850..999 (150 seeds)

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

        d_str = f"{d:.4f}"
        key = (mid, d_str, m.edge)
        val1_data[key] = np.array([r.novelty_s for r in loc1])
        val2_data[key] = np.array([r.novelty_s for r in loc2])
        val_all_data[key] = np.array([r.novelty_s for r in loc_all])
        test_data[key] = np.array([r.novelty_s for r in loc_tst])

print(f"Collected {len(val1_data)} classes.")

def run_calibration(cal_data, n_weight, n0_vals):
    # Calculate c, w on cal_data
    c_map = {}
    w_map = {}
    edge_w = {}
    for (mid, d_str, edge), scores in cal_data.items():
        log_s = np.log(np.maximum(scores, 1e-12))
        c = float(np.median(log_s))
        mad = float(np.median(np.abs(log_s - c)))
        w = 1.4826 * mad
        if w < 1e-8:
            w = 1.0
        c_map[(mid, d_str, edge)] = c
        w_map[(mid, d_str, edge)] = w
        edge_w.setdefault(edge, []).append(w)

    w_edge = {e: float(np.median(ws)) for e, ws in edge_w.items()}

    out = {}
    for n0 in n0_vals:
        w_prime = {}
        for (mid, d_str, edge), w in w_map.items():
            wp = (float(n_weight) * w + n0 * w_edge[edge]) / (float(n_weight) + n0)
            w_prime[(mid, d_str, edge)] = wp

        # Tail per edge
        edge_z = {}
        for (mid, d_str, edge), scores in cal_data.items():
            log_s = np.log(np.maximum(scores, 1e-12))
            z = (log_s - c_map[(mid, d_str, edge)]) / w_prime[(mid, d_str, edge)]
            edge_z.setdefault(edge, []).extend(z)

        z_star = {e: float(np.percentile(zs, 99)) for e, zs in edge_z.items()}

        tau_map = {}
        for (mid, d_str, edge) in cal_data.keys():
            tau = float(np.exp(c_map[(mid, d_str, edge)] + z_star[edge] * w_prime[(mid, d_str, edge)]))
            tau_map[(mid, d_str)] = tau

        out[n0] = {
            "w_edge": w_edge,
            "z_star": z_star,
            "tau_map": tau_map,
            "c_map": c_map,
            "w_prime": w_prime
        }
    return out

# 1. Validation tuning: Calibrate on val1 (75 seeds), evaluate on val2 (75 seeds)
print("=== Validation Split 1 (700..774) -> Split 2 (775..849) ===")
# Try with n_weight = 75 (actual cal sample size)
print("--- Using n_weight = 75 ---")
val_models_75 = run_calibration(val1_data, 75, [0, 25, 50, 100, 150, 300])
for n0, m in val_models_75.items():
    counts = []
    for (mid, d_str, edge), s2 in val2_data.items():
        fa = int(np.sum(s2 > m["tau_map"][(mid, d_str)]))
        counts.append(fa)
    counts = np.array(counts)
    mean_c = np.mean(counts)
    var_c = np.var(counts, ddof=1)
    disp = var_c / (mean_c + 1e-12)
    max_c = np.max(counts)
    total_fa = np.sum(counts)
    print(f"n0={n0:3d}: mean={mean_c:.4f}, var={var_c:.4f}, disp={disp:.4f}, max={max_c}/75, total={total_fa}/9300 ({total_fa/93:.2f}%)")

# Try with n_weight = 150 (literal 150 from formula)
print("--- Using n_weight = 150 ---")
val_models_150 = run_calibration(val1_data, 150, [0, 25, 50, 100, 150, 300])
for n0, m in val_models_150.items():
    counts = []
    for (mid, d_str, edge), s2 in val2_data.items():
        fa = int(np.sum(s2 > m["tau_map"][(mid, d_str)]))
        counts.append(fa)
    counts = np.array(counts)
    mean_c = np.mean(counts)
    var_c = np.var(counts, ddof=1)
    disp = var_c / (mean_c + 1e-12)
    max_c = np.max(counts)
    total_fa = np.sum(counts)
    print(f"n0={n0:3d}: mean={mean_c:.4f}, var={var_c:.4f}, disp={disp:.4f}, max={max_c}/75, total={total_fa}/9300 ({total_fa/93:.2f}%)")

# 2. Recalibrate on full validation (700..849, 150 seeds) and evaluate on test (850..999, 150 seeds)
print("\n=== Recalibrate on Full Val (700..849) -> Evaluate on Test (850..999) ===")
full_val_models = run_calibration(val_all_data, 150, [0, 25, 50, 100, 150, 300])
for n0, m in full_val_models.items():
    counts = []
    counts_by_edge = {}
    for (mid, d_str, edge), stest in test_data.items():
        fa = int(np.sum(stest > m["tau_map"][(mid, d_str)]))
        counts.append(fa)
        counts_by_edge.setdefault(edge, []).append(fa)
    counts = np.array(counts)
    mean_c = np.mean(counts)
    var_c = np.var(counts, ddof=1)
    disp = var_c / (mean_c + 1e-12)
    max_c = np.max(counts)
    total_fa = np.sum(counts)
    arm_counts = np.array(counts_by_edge["armchair"])
    zig_counts = np.array(counts_by_edge["zigzag"])
    arm_fa = np.sum(arm_counts) / (len(arm_counts) * 150)
    zig_fa = np.sum(zig_counts) / (len(zig_counts) * 150)
    print(f"n0={n0:3d}: z*={m['z_star']}, disp={disp:.4f}, max={max_c}/150, total={total_fa}/18600 ({total_fa/186:.2f}%), arm_fa={arm_fa*100:.3f}%, zig_fa={zig_fa*100:.3f}%")
