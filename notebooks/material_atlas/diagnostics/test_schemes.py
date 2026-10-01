"""Test different formulations of w_edge and shrinkage."""

import numpy as np
from atlaslib.store import CloudStore
from atlaslib import Atlas

store = CloudStore("/home/shardul/atlas_store/engine_v1")
atlas = Atlas.load("notebooks/material_atlas/atlas_v2")

val_all = {}
test_all = {}

for m in atlas.models:
    mid = m.model_id
    e_t, _ = store.read_pristine(mid)
    for d in store.densities(mid):
        c, s = store.read_cloud(mid, d)
        v = (s >= 700) & (s <= 849)
        t = (s >= 850)
        loc_v = atlas.locate(c[v], e_t, m.band_top_t)
        loc_t = atlas.locate(c[t], e_t, m.band_top_t)
        key = (mid, f"{d:.4f}", m.edge, d)
        val_all[key] = [(r.novelty_s, r.density) for r in loc_v]
        test_all[key] = [(r.novelty_s, r.density) for r in loc_t]

print("Loaded all classes.")

def test_scheme(desc, get_w_target, n0_vals):
    print(f"\n--- {desc} ---")
    for n0 in n0_vals:
        # 1. c and w per class
        class_stats = {}
        for (mid, d_str, edge, d), items in val_all.items():
            s_vals = np.array([x[0] for x in items])
            log_s = np.log(np.maximum(s_vals, 1e-12))
            c = float(np.median(log_s))
            w = 1.4826 * float(np.median(np.abs(log_s - c)))
            if w < 1e-8: w = 1.0
            class_stats[(mid, d_str, edge, d)] = (c, w)

        # 2. Target w for shrinkage
        w_targets = get_w_target(class_stats)

        # 3. Shrink w
        w_prime = {}
        for key, (c, w) in class_stats.items():
            wt = w_targets[key]
            wp = (150.0 * w + n0 * wt) / (150.0 + n0)
            w_prime[key] = wp

        # 4. Tail per edge
        edge_z = {}
        for key, items in val_all.items():
            mid, d_str, edge, d = key
            s_vals = np.array([x[0] for x in items])
            log_s = np.log(np.maximum(s_vals, 1e-12))
            c, _ = class_stats[key]
            wp = w_prime[key]
            z = (log_s - c) / wp
            edge_z.setdefault(edge, []).extend(z)

        z_star = {e: float(np.percentile(zs, 99)) for e, zs in edge_z.items()}

        # 5. Thresholds
        table = {}
        for key, (c, _) in class_stats.items():
            mid, d_str, edge, d = key
            wp = w_prime[key]
            tau = float(np.exp(c + z_star[edge] * wp))
            table.setdefault(mid, {})[d_str] = tau

        # 6. Evaluate on test set
        counts = []
        counts_by_edge = {}
        lines = []
        for key, items in test_all.items():
            mid, d_str, edge, d = key
            d_map = table[mid]
            avail_d = [float(k) for k in d_map.keys()]
            fa = 0
            for s, pred_d in items:
                snapped_d = min(avail_d, key=lambda x: abs(x - pred_d))
                tau = d_map[f"{snapped_d:.4f}"]
                if s > tau:
                    fa += 1
            counts.append(fa)
            counts_by_edge.setdefault(edge, []).append(fa)
            lines.append((mid, d_str, edge, fa))

        counts = np.array(counts)
        disp = np.var(counts, ddof=1) / np.mean(counts)
        arm_r = np.sum(counts_by_edge["armchair"]) / (len(counts_by_edge["armchair"]) * 150)
        zig_r = np.sum(counts_by_edge["zigzag"]) / (len(counts_by_edge["zigzag"]) * 150)
        max_fa = np.max(counts)
        high = [f"{x[0].split('/')[-1]}-d{x[1]}:{x[3]}" for x in lines if x[3] >= 8]
        print(f"n0={n0:3d}: disp={disp:.3f}, max={max_fa}/150, arm={arm_r*100:.2f}%, zig={zig_r*100:.2f}%, z*={z_star}, lines>=8: {high}")

# Scheme A: w_target = median(w) over all classes of that edge (as written in plan)
def target_edge(class_stats):
    edge_w = {}
    for (mid, d_str, edge, d), (c, w) in class_stats.items():
        edge_w.setdefault(edge, []).append(w)
    meds = {e: float(np.median(ws)) for e, ws in edge_w.items()}
    return {key: meds[key[2]] for key in class_stats}

# Scheme B: w_target = median(w) over classes of that edge AND density
def target_edge_dens(class_stats):
    ed_w = {}
    for (mid, d_str, edge, d), (c, w) in class_stats.items():
        ed_w.setdefault((edge, d), []).append(w)
    meds = {ed: float(np.median(ws)) for ed, ws in ed_w.items()}
    return {key: meds[(key[2], key[3])] for key in class_stats}

# Scheme C: what if pooled z* (Option B) with shrinkage?
def run_pooled_z(n0_vals):
    print("\n--- Scheme C: Pooled z* (across edges) with edge shrinkage ---")
    for n0 in n0_vals:
        class_stats = {}
        for (mid, d_str, edge, d), items in val_all.items():
            s_vals = np.array([x[0] for x in items])
            log_s = np.log(np.maximum(s_vals, 1e-12))
            c = float(np.median(log_s))
            w = 1.4826 * float(np.median(np.abs(log_s - c)))
            if w < 1e-8: w = 1.0
            class_stats[(mid, d_str, edge, d)] = (c, w)
        w_targets = target_edge(class_stats)
        w_prime = {k: (150.0*w + n0*w_targets[k])/(150.0+n0) for k, (c, w) in class_stats.items()}
        all_z = []
        for key, items in val_all.items():
            s_vals = np.array([x[0] for x in items])
            log_s = np.log(np.maximum(s_vals, 1e-12))
            c, _ = class_stats[key]
            all_z.extend((log_s - c)/w_prime[key])
        z_star = float(np.percentile(all_z, 99))
        table = {}
        for key, (c, _) in class_stats.items():
            mid, d_str, edge, d = key
            tau = float(np.exp(c + z_star * w_prime[key]))
            table.setdefault(mid, {})[d_str] = tau
        counts = []
        counts_by_edge = {}
        lines = []
        for key, items in test_all.items():
            mid, d_str, edge, d = key
            d_map = table[mid]
            avail_d = [float(k) for k in d_map.keys()]
            fa = 0
            for s, pred_d in items:
                snapped_d = min(avail_d, key=lambda x: abs(x - pred_d))
                tau = d_map[f"{snapped_d:.4f}"]
                if s > tau: fa += 1
            counts.append(fa)
            counts_by_edge.setdefault(edge, []).append(fa)
            lines.append((mid, d_str, edge, fa))
        counts = np.array(counts)
        disp = np.var(counts, ddof=1) / np.mean(counts)
        arm_r = np.sum(counts_by_edge["armchair"]) / (len(counts_by_edge["armchair"]) * 150)
        zig_r = np.sum(counts_by_edge["zigzag"]) / (len(counts_by_edge["zigzag"]) * 150)
        high = [f"{x[0].split('/')[-1]}-d{x[1]}:{x[3]}" for x in lines if x[3] >= 8]
        print(f"n0={n0:3d}: disp={disp:.3f}, max={np.max(counts)}/150, arm={arm_r*100:.2f}%, zig={zig_r*100:.2f}%, z*={z_star:.3f}, lines>=8: {high}")

test_scheme("Scheme A: Edge median (Plan specification)", target_edge, [0, 25, 50, 100, 150, 300])
test_scheme("Scheme B: Edge-Density median", target_edge_dens, [0, 25, 50, 100, 150, 300])
run_pooled_z([0, 25, 50, 100, 150, 300])
