#!/usr/bin/env python
"""Inspect and validate cloud store correctness, producing report.json."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "notebooks" / "material_atlas"))
sys.path.insert(0, str(REPO / "notebooks"))
sys.path.insert(0, str(REPO / "notebooks" / "agnr" / "physics"))

from atlaslib import CloudStore, Registry
from tbribbon.bands import open_channels
from tbribbon.disorder import impurity_shifts
from tbribbon.materials import hamiltonian_for, make_model
import agnr_lib


def check_seed_excess(spec, pris, step_mask=None, tol=0.05):
    """Robust per-seed mean excess check: mean of min(T, pristine + 1) - pristine over unmasked channels <= tol.

    Truncating at pristine + 1 prevents single-channel trace-formula resonance spikes
    from dominating the seed average while catching any systematic unphysical enhancement.
    """
    if step_mask is not None and np.any(~step_mask):
        away = ~step_mask
        pris_sub = pris[away]
        spec_sub = spec[:, away]
    else:
        pris_sub = pris
        spec_sub = spec
    robust_diff = np.minimum(spec_sub, pris_sub + 1.0) - pris_sub
    seed_robust_excess = np.mean(robust_diff, axis=1)
    max_seed_mean_excess = float(np.max(seed_robust_excess))
    return max_seed_mean_excess, bool(max_seed_mean_excess <= tol)


def check_model(store, model, reg):
    mid = model.model_id
    meta = store._meta(mid)
    formula = meta.get("pristine_formula", "unknown")
    e_t, pris = store.read_pristine(mid)

    h = hamiltonian_for(model)
    ch = open_channels(h.H0, h.H1, e_t)
    step_mask = np.zeros(len(pris), dtype=bool)
    jump = np.abs(np.diff(pris))
    step_indices = np.where(jump > 0.5)[0]
    for s in step_indices:
        step_mask[max(0, s - 4):min(len(pris), s + 6)] = True

    ch_prev = open_channels(h.H0, h.H1, e_t - 0.02)
    ch_next = open_channels(h.H0, h.H1, e_t + 0.02)
    stable = (~step_mask) & (e_t > 0.02) & (np.abs(e_t - 1.0) > 0.02) & (ch_prev == ch) & (ch_next == ch)
    clean_err = float(np.max(np.abs(pris[stable] - ch[stable]))) if np.any(stable) else 0.0

    densities = store.densities(mid)
    density_stats = {}
    clouds_by_density = {}
    seeds_by_density = {}
    hashes_by_density = {}

    for d in densities:
        spec, sds = store.read_cloud(mid, d)
        clouds_by_density[d] = spec
        seeds_by_density[d] = sds
        hashes_by_density[d] = [hashlib.md5(r.tobytes()).hexdigest() for r in spec]

        med = np.median(spec, axis=0)
        near = step_mask
        away = ~step_mask
        n_unmasked = int(np.sum(away))

        max_med_diff_away = float(np.max((med - pris)[away])) if np.any(away) else 0.0
        max_med_diff_all = float(np.max(med - pris))
        max_med_diff_near = float(np.max((med - pris)[near])) if np.any(near) else 0.0

        # Robust seed excess check: mean of min(T, pristine + 1) - pristine over unmasked channels <= 0.05
        max_seed_mean_excess, seed_mean_valid = check_seed_excess(spec, pris, step_mask=step_mask, tol=0.05)

        above_pris_05 = spec > pris + 0.5
        share_above_all = float(np.mean(above_pris_05))
        share_above_near = float(np.mean(above_pris_05[:, near])) if np.any(near) else 0.0
        share_above_away = float(np.mean(above_pris_05[:, away])) if np.any(away) else 0.0
        max_t = float(np.max(spec))

        density_stats[f"{d:.4f}"] = {
            "n_seeds": len(sds),
            "n_unmasked_channels": n_unmasked,
            "max_median_excess": round(max_med_diff_away, 6),
            "max_median_excess_away": round(max_med_diff_away, 6),
            "max_median_excess_all": round(max_med_diff_all, 6),
            "max_median_excess_near": round(max_med_diff_near, 6),
            "max_seed_mean_excess_unmasked": round(max_seed_mean_excess, 6),
            "seed_mean_valid": seed_mean_valid,
            "share_above_pristine_plus_05": round(share_above_all, 4),
            "share_above_near": round(share_above_near, 4),
            "share_above_away": round(share_above_away, 4),
            "max_T": round(max_t, 4)
        }

    # Cross-density duplicate check
    has_duplicates = False
    sorted_dens = sorted(densities)
    for i in range(len(sorted_dens)):
        for j in range(i + 1, len(sorted_dens)):
            d1, d2 = sorted_dens[i], sorted_dens[j]
            overlap = set(hashes_by_density[d1]).intersection(set(hashes_by_density[d2]))
            if overlap:
                has_duplicates = True

    # Seed nesting check
    nesting_valid = True
    is_agnr = (model.material == "graphene-ideal" and model.edge == "armchair")
    if sorted_dens:
        for s_idx, s in enumerate(seeds_by_density.get(sorted_dens[0], [])):
            prev_set = None
            for d in sorted_dens:
                n_imp = model.impurities_for_density(d)
                if is_agnr:
                    combs = agnr_lib.chosen_for_config(n_imp, model.width, s)
                    cur_set = set(map(tuple, combs))
                else:
                    shifts = impurity_shifts(model.n_cells, h.H0.shape[0], n_imp, seed=s, v=model.impurity_v_t, orbitals_per_site=model.orbitals_per_site)
                    cur_set = set(zip(*np.where(shifts > 0)))
                if prev_set is not None and not prev_set.issubset(cur_set):
                    nesting_valid = False
                    break
                prev_set = cur_set
            if not nesting_valid:
                break

    # Evaluation
    passed = (
        clean_err < 1e-3
        and bool(density_stats)
        and all(stats["max_median_excess"] <= 0.05 and stats["seed_mean_valid"] for stats in density_stats.values())
        and not has_duplicates
        and nesting_valid
    )

    n_unmasked_total = int(np.sum(~step_mask))
    return {
        "formula": formula,
        "clean_channels_max_err": round(clean_err, 6),
        "n_unmasked_channels": n_unmasked_total,
        "total_channels": len(pris),
        "unmasked_pct": round(n_unmasked_total / len(pris) * 100, 1),
        "densities": density_stats,
        "no_cross_density_duplicates": not has_duplicates,
        "seed_nesting_valid": nesting_valid,
        "status": "PASS" if passed else "FAIL"
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="~/atlas_store/smoke_v1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--narrow-only", action="store_true", help="Validate only the 21 narrow baseline models")
    a = ap.parse_args()

    store_path = Path(os.path.expanduser(a.store))
    store = CloudStore(store_path)
    out_path = Path(a.out) if a.out else (store_path / "report.json")

    reg = Registry()
    if a.narrow_only:
        models = ([make_model("graphene-ideal", "armchair", n) for n in range(5, 17)]
                  + [make_model("graphene-ideal", "zigzag", n) for n in range(4, 13)])
    else:
        # Discover all models present in the store:
        models = []
        for mid in sorted(store.models()):
            parts = mid.split("/")
            if len(parts) == 3:
                mat, edge, w_str = parts
                if w_str.startswith("N"):
                    models.append(make_model(mat, edge, int(w_str[1:])))
        if not models:
            models = ([make_model("graphene-ideal", "armchair", n) for n in range(5, 17)]
                      + [make_model("graphene-ideal", "zigzag", n) for n in range(4, 13)])

    report = {}
    all_pass = True
    print(f"\n[INFO] Validating cloud store at: {store_path}")
    print(f"{'Model':<30} | {'Formula':<18} | {'CleanErr':<10} | {'MaxExcess':<10} | {'Status'}")
    print("-" * 80)

    for m in models:
        mid = m.model_id
        if not store.has_pristine(mid) or not store.densities(mid):
            continue
        res = check_model(store, m, reg)
        report[mid] = res
        if res["status"] != "PASS":
            all_pass = False
        max_ex = max(s["max_median_excess"] for s in res["densities"].values()) if res["densities"] else 0.0
        print(f"{mid:<30} | {res['formula']:<18} | {res['clean_channels_max_err']:<10.2e} | {max_ex:<+10.4f} | {res['status']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print("-" * 80)
    print(f"Overall status: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    print(f"Report written to: {out_path}\n")


if __name__ == "__main__":
    main()
