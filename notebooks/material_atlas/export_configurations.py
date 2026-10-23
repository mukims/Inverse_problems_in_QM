#!/usr/bin/env python
"""Export paired quantum transport transmission spectra and microscopic configuration data.

Output .npz file contains:
- 'spectra': (N, n_channels) transmission spectra T(E)
- 'energies': (n_channels,) energy grid (eV or t)
- 'model_id': (N,) string array of ribbon model identifiers
- 'density': (N,) float array of impurity densities
- 'seed': (N,) int array of configuration seeds
- 'n_impurities': (N,) int array of impurity counts
- 'coords': list/ragged array of (n_imp, 2) exact impurity coordinates
- 'cell_idx': list/ragged array of (n_imp,) unit cell indices
- 'site_idx': list/ragged array of (n_imp,) site indices within unit cell
- 'host_onsite_t': list/ragged array of (n_imp,) host on-site potentials (t)
- 'host_onsite_ev': list/ragged array of (n_imp,) host on-site potentials (eV)
- 'impurity_onsite_t': list/ragged array of (n_imp,) perturbed impurity on-site potentials (t)
- 'impurity_onsite_ev': list/ragged array of (n_imp,) perturbed impurity on-site potentials (eV)
- 'defect_v_t': (N,) float array of perturbation shifts v (t)
- 'defect_v_ev': (N,) float array of perturbation shifts v (eV)
"""
import argparse
from pathlib import Path
import numpy as np

from atlaslib import CloudStore, InputSpec
from tbribbon.configuration import (
    reconstruct_configuration,
    device_atom_coordinates,
    device_onsite_profile,
)


def export_pairs(
    store_path,
    model_ids,
    output_path,
    densities=None,
    seeds=None,
    unit="lattice",
    include_full_profiles=False,
):
    store = CloudStore(Path(store_path).expanduser())
    all_models = store.models() if not model_ids else model_ids

    spec_list = []
    mid_list = []
    dens_list = []
    seed_list = []
    n_imp_list = []
    coords_list = []
    cell_idx_list = []
    site_idx_list = []
    host_t_list = []
    host_ev_list = []
    imp_t_list = []
    imp_ev_list = []
    v_t_list = []
    v_ev_list = []
    profiles_list = []
    energy_grid = None

    for mid in sorted(all_models):
        if not store.has_pristine(mid):
            continue
        e_grid, _ = store.read_pristine(mid)
        if energy_grid is None:
            energy_grid = e_grid

        avail_dens = store.densities(mid)
        target_dens = avail_dens if densities is None else [d for d in avail_dens if any(np.isclose(d, td) for td in densities)]

        for d in target_dens:
            cloud, cloud_seeds = store.read_cloud(mid, d)
            if seeds is not None:
                mask = np.isin(cloud_seeds, seeds)
                cloud = cloud[mask]
                cloud_seeds = cloud_seeds[mask]

            for s_idx, s in enumerate(cloud_seeds):
                cfg = store.read_configuration(mid, d, s, unit=unit)
                spec_list.append(cloud[s_idx])
                mid_list.append(mid)
                dens_list.append(d)
                seed_list.append(s)
                n_imp_list.append(cfg.n_impurities)
                coords_list.append(cfg.coordinates)
                cell_idx_list.append(cfg.cell_indices)
                site_idx_list.append(cfg.site_indices)
                host_t_list.append(cfg.host_onsite_t)
                host_ev_list.append(cfg.host_onsite_ev)
                imp_t_list.append(cfg.impurity_onsite_t)
                imp_ev_list.append(cfg.impurity_onsite_ev)
                v_t_list.append(cfg.defect_v_t)
                v_ev_list.append(cfg.defect_v_ev)

                if include_full_profiles:
                    prof = device_onsite_profile(mid, s, density=d, n_cells=cfg.n_cells, unit="t")
                    profiles_list.append(prof)

    out_dict = {
        "spectra": np.array(spec_list, dtype=np.float32),
        "energies": np.array(energy_grid, dtype=np.float32) if energy_grid is not None else np.empty(0),
        "model_id": np.array(mid_list),
        "density": np.array(dens_list, dtype=np.float32),
        "seed": np.array(seed_list, dtype=np.int32),
        "n_impurities": np.array(n_imp_list, dtype=np.int32),
        "coords": np.array(coords_list, dtype=object),
        "cell_idx": np.array(cell_idx_list, dtype=object),
        "site_idx": np.array(site_idx_list, dtype=object),
        "host_onsite_t": np.array(host_t_list, dtype=object),
        "host_onsite_ev": np.array(host_ev_list, dtype=object),
        "impurity_onsite_t": np.array(imp_t_list, dtype=object),
        "impurity_onsite_ev": np.array(imp_ev_list, dtype=object),
        "defect_v_t": np.array(v_t_list, dtype=np.float32),
        "defect_v_ev": np.array(v_ev_list, dtype=np.float32),
    }
    if include_full_profiles:
        out_dict["device_profiles"] = np.array(profiles_list, dtype=object)

    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **out_dict)
    print(f"Exported {len(spec_list)} transmission + configuration pairs to {output_path}")
    return out_dict


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export paired transmission and configuration data")
    ap.add_argument("--store", default="~/atlas_store/materials3_ev_full", help="CloudStore root directory")
    ap.add_argument("--models", default="", help="Comma-separated model IDs (default: all)")
    ap.add_argument("--densities", default="", help="Comma-separated densities (default: all)")
    ap.add_argument("--seeds", default="", help="Seed range e.g. '0-9' or '0,1,2'")
    ap.add_argument("--output", default="paired_transmission_config.npz", help="Output .npz filepath")
    ap.add_argument("--unit", choices=["lattice", "angstrom"], default="lattice")
    ap.add_argument("--include-full-profiles", action="store_true")
    args = ap.parse_args()

    m_ids = [m.strip() for m in args.models.split(",") if m.strip()] or None
    d_vals = [float(d.strip()) for d in args.densities.split(",") if d.strip()] or None
    s_vals = None
    if args.seeds:
        if "-" in args.seeds:
            lo, hi = map(int, args.seeds.split("-"))
            s_vals = list(range(lo, hi + 1))
        else:
            s_vals = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    export_pairs(args.store, m_ids, args.output, densities=d_vals, seeds=s_vals,
                 unit=args.unit, include_full_profiles=args.include_full_profiles)
