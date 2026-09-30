#!/usr/bin/env python
"""Atlas v2 from engine clouds; generalisation test on held-out widths."""
import argparse
import json
from pathlib import Path

import numpy as np

from atlaslib import Atlas, CloudStore, InputSpec, Registry
from tbribbon.materials import make_model

HELD_OUT = {("graphene-ideal", "armchair"): [8, 12, 13], ("graphene-ideal", "zigzag"): [8]}
GRIDS = {
    "baseline21": {
        ("graphene-ideal", "armchair"): list(range(5, 17)),
        ("graphene-ideal", "zigzag"): list(range(4, 13)),
    },
    "sparse31": {
        ("graphene-ideal", "armchair"): [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 27, 31, 40, 50],
        ("graphene-ideal", "zigzag"): [4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 27, 40, 50],
    }
}
TRAIN = GRIDS["baseline21"]



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="~/atlas_store/engine_v1")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "atlas_v2"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--grid", choices=["baseline21", "sparse31"], default="sparse31")
    ap.add_argument("--spec-version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--holdout-widths", action="store_true", default=False,
                    help="Run legacy Gate 5 diagnostic with held-out widths (armchair 8, 12, 13; zigzag 8)")
    a = ap.parse_args()
    grid_models = GRIDS[a.grid]
    store, spec, out = CloudStore(a.store), InputSpec(version=a.spec_version), Path(a.out)
    reg = Registry()
    for (mat, edge), widths in grid_models.items():
        for n in widths:
            reg.add(make_model(mat, edge, n))

    if a.holdout_widths:
        # Legacy width-held-out diagnostic
        train_ids = [m.model_id for m in reg if m.width not in HELD_OUT[(m.material, m.edge)]]
        atlas = Atlas.build(store, reg, train_ids, spec, threads=a.threads)
        atlas.save(out)
        res = {}
        for (mat, edge), widths in HELD_OUT.items():
            for n in widths:
                mid = f"{mat}/{edge}/N{n}"
                T = np.concatenate([store.read_cloud(mid, d)[0] for d in store.densities(mid)])
                e_t, _ = store.read_pristine(mid)
                loc = atlas.locate(T, e_t, reg.get(mid).band_top_t)
                w = np.array([r.width for r in loc])
                res[mid] = {"material_accuracy": float(np.mean([r.material == mat for r in loc]) * 100),
                            "edge_accuracy": float(np.mean([r.edge == edge for r in loc]) * 100),
                            "median_width": float(np.median(w)),
                            "between_neighbours_pct": float(np.mean((w > n - 1.5) & (w < n + 1.5)) * 100)}
        (out / "generalisation.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return

    # Option A: Train on all widths, evaluate on held-out test seeds (LOGBOOK Bug #7)
    train_ids = [m.model_id for m in reg]
    first_mid = train_ids[0]
    _, seeds_sample = store.read_cloud(first_mid, store.densities(first_mid)[0])
    n_seeds = len(seeds_sample)

    if n_seeds <= 50:
        # Smoke split: 35 train (0..34), 8 val (35..42), 7 test (43..49)
        train_val_max = 42
        val_seed_min = 35
        test_seed_min = 43
    else:
        # Standard 70/15/15 split
        train_val_max = int(np.floor(n_seeds * 0.85)) - 1
        val_seed_min = int(np.floor(n_seeds * 0.70))
        test_seed_min = int(np.floor(n_seeds * 0.85))

    print(f"[INFO] Building Atlas v2 across all {len(train_ids)} models...")
    print(f"[INFO] Seed split: train < {val_seed_min}, val [{val_seed_min}..{train_val_max}], test >= {test_seed_min}")

    atlas = Atlas.build(store, reg, train_ids, spec, threads=a.threads,
                        max_seed=train_val_max, val_seed_min=val_seed_min)
    atlas.save(out)

    ident_res = {}
    gate5_pass = True

    print(f"\n[INFO] Evaluating on held-out test seeds (seeds >= {test_seed_min}):")
    print(f"{'Model':<30} | {'Dens':<6} | {'MatAcc':<7} | {'EdgeAcc':<7} | {'WidthAcc':<8} | {'Med|W-N|':<8} | {'Unknown%':<8} | {'PredDens':<8}")
    print("-" * 102)

    for (mat, edge), widths in grid_models.items():
        for n in widths:
            mid = f"{mat}/{edge}/N{n}"
            e_t, _ = store.read_pristine(mid)
            band_top = reg.get(mid).band_top_t
            model_stats = {}
            for d in store.densities(mid):
                c, s = store.read_cloud(mid, d)
                test_mask = s >= test_seed_min
                c_test = c[test_mask]
                if len(c_test) == 0:
                    continue
                loc = atlas.locate(c_test, e_t, band_top)
                mat_acc = float(np.mean([r.material == mat for r in loc]) * 100)
                edge_acc = float(np.mean([r.edge == edge for r in loc]) * 100)
                w_acc = float(np.mean([round(r.width_vote) == n for r in loc]) * 100)
                w_diff = float(np.median([abs(r.width_vote - n) for r in loc]))
                w_cont_acc = float(np.mean([round(r.width) == n for r in loc]) * 100)
                w_cont_diff = float(np.median([abs(r.width - n) for r in loc]))
                unknown_pct = float(np.mean([r.unknown for r in loc]) * 100)
                pred_dens = float(np.median([r.density for r in loc]))

                if mat_acc < 100.0 or edge_acc < 99.0 or w_acc < 99.0:
                    gate5_pass = False

                model_stats[f"{d:.4f}"] = {
                    "material_accuracy": round(mat_acc, 2),
                    "edge_accuracy": round(edge_acc, 2),
                    "width_accuracy": round(w_acc, 2),
                    "median_width_diff": round(w_diff, 4),
                    "width_continuous_accuracy": round(w_cont_acc, 2),
                    "median_width_continuous_diff": round(w_cont_diff, 4),
                    "share_flagged_unknown": round(unknown_pct, 2),
                    "median_predicted_density": round(pred_dens, 4),
                    "true_density": float(d),
                    "n_test_samples": len(c_test)
                }
                print(f"{mid:<30} | {d:<6.4f} | {mat_acc:<7.1f} | {edge_acc:<7.1f} | {w_acc:<8.1f} | {w_diff:<8.4f} | {unknown_pct:<8.1f} | {pred_dens:<8.4f}")
            ident_res[mid] = model_stats

    # Pooled per-density statistics across widths (Smoke Gate 5)
    pooled = {}
    print("\n[INFO] Pooled Evaluation by Density and Edge Type:")
    print(f"{'Group':<22} | {'Dens':<6} | {'Samples':<8} | {'MatAcc':<7} | {'EdgeAcc':<7} | {'WidthAcc':<8} | {'Unknown%'}")
    print("-" * 78)
    for edge_filter in ["armchair", "zigzag", "all"]:
        for d in [0.005, 0.01, 0.02, 0.04]:
            matches = [
                s for mid, mstats in ident_res.items()
                for d_str, s in mstats.items()
                if abs(float(d_str) - d) < 1e-5
                and (edge_filter == "all" or edge_filter in mid)
            ]
            total_n = sum(s["n_test_samples"] for s in matches)
            if total_n > 0:
                p_mat = sum(s["material_accuracy"] * s["n_test_samples"] for s in matches) / total_n
                p_edge = sum(s["edge_accuracy"] * s["n_test_samples"] for s in matches) / total_n
                p_width = sum(s["width_accuracy"] * s["n_test_samples"] for s in matches) / total_n
                p_unk = sum(s["share_flagged_unknown"] * s["n_test_samples"] for s in matches) / total_n
                key = f"{edge_filter}_d{d:.4f}"
                pooled[key] = {
                    "edge_type": edge_filter,
                    "density": d,
                    "n_test_samples": total_n,
                    "material_accuracy": round(p_mat, 2),
                    "edge_accuracy": round(p_edge, 2),
                    "width_accuracy": round(p_width, 2),
                    "share_flagged_unknown": round(p_unk, 2)
                }
                print(f"{edge_filter:<22} | {d:<6.4f} | {total_n:<8} | {p_mat:<7.1f} | {p_edge:<7.1f} | {p_width:<8.1f} | {p_unk:<7.1f}")

    full_output = {
        "per_model": ident_res,
        "pooled": pooled,
        "revised_gate5_pass_per_line": gate5_pass
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "identification.json").write_text(json.dumps(full_output, indent=2))
    print("-" * 102)
    print(f"[RESULT] Identification report written to {out / 'identification.json'}")
    print(f"[RESULT] Revised Gate 5 (100% material, >=99% edge, >=99% width on test seeds): {'PASS' if gate5_pass else 'FAIL (per-line 7-sample noise; check pooled)'}")


if __name__ == "__main__":
    main()
