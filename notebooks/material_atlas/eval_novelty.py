#!/usr/bin/env python
"""Novelty evaluation for Atlas v2:
1. Unseen material: Square strip N10 in ~/atlas_store/novelty_v1
2. Leave-one-out untrained widths: Armchair N13, Zigzag N8 in notebooks/material_atlas/atlas_v2_loo
3. Write atlas_v2/novelty.json
"""
import json
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from atlaslib import Atlas, CloudStore, InputSpec, Registry
from tbribbon.materials import make_model
from tbribbon.generate_clouds import generate

GRID_MODELS_31 = {
    ("graphene-ideal", "armchair"): [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 27, 31, 40, 50],
    ("graphene-ideal", "zigzag"): [4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 27, 40, 50],
}


def eval_square_strip(atlas_path, novelty_store_path, engine_store_path, out_dir):
    print("\n" + "=" * 80)
    print("STEP 3: Unseen Material Evaluation (Square Strip N10)")
    print("=" * 80)

    store = CloudStore(novelty_store_path)
    engine_store = CloudStore(engine_store_path)
    spec = InputSpec(version="v2")
    atlas = Atlas.load(atlas_path)

    # 1. Generate square strip N10 (150 seeds x 4 densities) if not already done
    m_sq = make_model("square", "strip", 10)
    densities = [0.005, 0.010, 0.020, 0.040]
    seeds = range(150)

    needs_gen = any(not store.has_cloud(m_sq.model_id, m_sq.impurities_for_density(d) / m_sq.n_sites)
                    for d in densities)
    if needs_gen:
        print(f"[INFO] Generating square strip N10 into {novelty_store_path}...")
        wrote = generate(store, [m_sq], densities, spec, n_jobs=4, formula="legacy_trace", seeds=seeds)
        print(f"[INFO] Generated {len(wrote)} clouds.")
    else:
        print(f"[INFO] Square strip N10 already generated in {novelty_store_path}.")

    # 2. Query square strip label-free (band_top_t=None)
    e_t, _ = store.read_pristine(m_sq.model_id)
    sq_locs = []
    sq_dens_results = {}
    for d in densities:
        c, s = store.read_cloud(m_sq.model_id, d)
        loc = atlas.locate(c, e_t, band_top_t=None)
        sq_locs.extend(loc)
        unk_b = np.mean([r.unknown for r in loc]) * 100
        unk_recon = np.mean([r.unknown_recon for r in loc]) * 100
        med_ratio = np.median([r.novelty_ratio for r in loc])
        sq_dens_results[f"{d:.4f}"] = {
            "n_samples": len(loc),
            "flagged_unknown_option_b_pct": round(unk_b, 2),
            "flagged_unknown_recon_pct": round(unk_recon, 2),
            "median_novelty_ratio": round(float(med_ratio), 4)
        }
        print(f"Square N10 d={d:.4f}: Option B unknown = {unk_b:.1f}%, Recon unknown = {unk_recon:.1f}%, Med ratio = {med_ratio:.2f}")

    total_sq_unk_b = float(np.mean([r.unknown for r in sq_locs]) * 100)
    total_sq_unk_recon = float(np.mean([r.unknown_recon for r in sq_locs]) * 100)
    print(f"Total Square N10: Option B unknown = {total_sq_unk_b:.2f}%, Recon unknown = {total_sq_unk_recon:.2f}%")

    # 3. AUROC vs known test spectra
    reg = Registry()
    for (mat, edge), widths in GRID_MODELS_31.items():
        for n in widths:
            reg.add(make_model(mat, edge, n))

    known_scores_b = []
    known_scores_recon = []
    for mid in reg.ids():
        e_known, _ = engine_store.read_pristine(mid)
        bt = reg.get(mid).band_top_t
        for d in engine_store.densities(mid):
            c, s = engine_store.read_cloud(mid, d)
            c_test = c[s >= 850]
            loc_k = atlas.locate(c_test, e_known, bt)
            known_scores_b.extend([r.novelty_ratio for r in loc_k])
            known_scores_recon.extend([r.recon_error for r in loc_k])

    sq_scores_b = [r.novelty_ratio for r in sq_locs]
    sq_scores_recon = [r.recon_error for r in sq_locs]

    y_true = np.concatenate([np.zeros(len(known_scores_b)), np.ones(len(sq_scores_b))])
    y_score_b = np.concatenate([known_scores_b, sq_scores_b])
    y_score_recon = np.concatenate([known_scores_recon, sq_scores_recon])

    auroc_b = float(roc_auc_score(y_true, y_score_b))
    auroc_recon = float(roc_auc_score(y_true, y_score_recon))

    print(f"AUROC (Square vs Known): Option B = {auroc_b:.4f}, Recon Error = {auroc_recon:.4f}")

    # Gate verification: B flags >= 95% of square spectra
    gate_pass = total_sq_unk_b >= 95.0
    print(f"Gate 3 (Option B flags >= 95% square spectra): {'PASS' if gate_pass else 'FAIL'}")

    return {
        "total_flagged_unknown_option_b_pct": round(total_sq_unk_b, 2),
        "total_flagged_unknown_recon_pct": round(total_sq_unk_recon, 2),
        "auroc_option_b": round(auroc_b, 4),
        "auroc_recon_error": round(auroc_recon, 4),
        "gate_pass": gate_pass,
        "per_density": sq_dens_results
    }


def eval_leave_one_out(engine_store_path, loo_atlas_path):
    print("\n" + "=" * 80)
    print("STEP 4: Leave-One-Out Untrained Width Evaluation")
    print("=" * 80)

    store = CloudStore(engine_store_path)
    spec = InputSpec(version="v2")
    loo_dir = Path(loo_atlas_path)

    # Exclude Armchair N13 and Zigzag N8
    loo_models = {
        ("graphene-ideal", "armchair"): [w for w in GRID_MODELS_31[("graphene-ideal", "armchair")] if w != 13],
        ("graphene-ideal", "zigzag"): [w for w in GRID_MODELS_31[("graphene-ideal", "zigzag")] if w != 8],
    }
    reg_loo = Registry()
    for (mat, edge), widths in loo_models.items():
        for n in widths:
            reg_loo.add(make_model(mat, edge, n))

    train_ids = [m.model_id for m in reg_loo]
    val_seed_min = 700
    train_val_max = 849
    test_seed_min = 850

    if not (loo_dir / "encoder.pt").exists():
        print(f"[INFO] Building Leave-One-Out Atlas (omitting armchair N13, zigzag N8)...")
        atlas_loo = Atlas.build(store, reg_loo, train_ids, spec, threads=4,
                                max_seed=train_val_max, val_seed_min=val_seed_min)
    else:
        print(f"[INFO] Loading existing LOO Atlas from {loo_dir}...")
        atlas_loo = Atlas.load(loo_dir)

    print("[INFO] Calibrating LOO Option B on validation seeds...")
    atlas_loo.calibrate_novelty(store, reg_loo, train_ids, val_seed_min=val_seed_min, max_seed=train_val_max)
    atlas_loo.save(loo_dir)

    # Evaluate held-out widths: armchair N13 and zigzag N8 on test seeds 850..999
    reg_full = Registry()
    for (mat, edge), widths in GRID_MODELS_31.items():
        for n in widths:
            reg_full.add(make_model(mat, edge, n))

    loo_results = {}
    for mid in ["graphene-ideal/armchair/N13", "graphene-ideal/zigzag/N8"]:
        e_t, _ = store.read_pristine(mid)
        bt = reg_full.get(mid).band_top_t
        model_res = {}
        all_locs = []
        for d in store.densities(mid):
            c, s = store.read_cloud(mid, d)
            c_test = c[s >= test_seed_min]
            loc = atlas_loo.locate(c_test, e_t, bt)
            all_locs.extend(loc)
            unk_b = np.mean([r.unknown for r in loc]) * 100
            unk_recon = np.mean([r.unknown_recon for r in loc]) * 100
            med_ratio = np.median([r.novelty_ratio for r in loc])
            model_res[f"{d:.4f}"] = {
                "flagged_unknown_option_b_pct": round(float(unk_b), 2),
                "flagged_unknown_recon_pct": round(float(unk_recon), 2),
                "median_novelty_ratio": round(float(med_ratio), 4)
            }
        tot_unk_b = float(np.mean([r.unknown for r in all_locs]) * 100)
        tot_unk_recon = float(np.mean([r.unknown_recon for r in all_locs]) * 100)
        loo_results[mid] = {
            "total_flagged_unknown_option_b_pct": round(tot_unk_b, 2),
            "total_flagged_unknown_recon_pct": round(tot_unk_recon, 2),
            "per_density": model_res
        }
        print(f"LOO {mid}: Option B unknown = {tot_unk_b:.1f}%, Recon unknown = {tot_unk_recon:.1f}%")

    return loo_results


def main():
    atlas_path = Path("notebooks/material_atlas/atlas_v2")
    engine_store_path = os.path.expanduser("~/atlas_store/engine_v1")
    novelty_store_path = os.path.expanduser("~/atlas_store/novelty_v1")
    loo_atlas_path = Path("notebooks/material_atlas/atlas_v2_loo")

    sq_results = eval_square_strip(atlas_path, novelty_store_path, engine_store_path, atlas_path)
    loo_results = eval_leave_one_out(engine_store_path, loo_atlas_path)

    # Load identification results to extract per-line and pooled false alarms
    ident_file = atlas_path / "identification.json"
    ident_data = json.loads(ident_file.read_text())

    # Build novelty.json
    novelty_report = {
        "square_strip_unseen_material": sq_results,
        "leave_one_out_untrained_widths": loo_results,
        "known_test_pooled_false_alarms": {
            k: {
                "density": v["density"],
                "edge_type": v["edge_type"],
                "n_test_samples": v["n_test_samples"],
                "option_b_false_alarm_pct": v["share_flagged_unknown"],
                "recon_false_alarm_pct": v.get("share_flagged_unknown_recon", 0.0)
            }
            for k, v in ident_data.get("pooled", {}).items()
        }
    }

    out_file = atlas_path / "novelty.json"
    out_file.write_text(json.dumps(novelty_report, indent=2))
    print(f"\n[RESULT] Saved comprehensive novelty report to {out_file}")


if __name__ == "__main__":
    main()
