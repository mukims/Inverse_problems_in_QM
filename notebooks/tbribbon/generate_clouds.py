#!/usr/bin/env python
"""Parallel, resumable generation of disorder clouds into a CloudStore.
Set OMP_NUM_THREADS=1 before launching to avoid BLAS oversubscription."""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

from atlaslib import CloudStore, InputSpec
from tbribbon.disorder import impurity_shifts
from tbribbon.leads import LeadCache
from tbribbon.materials import hamiltonian_for, make_model
from tbribbon.transport import spectrum

REPO = Path(__file__).resolve().parents[2]
AGNR_PHYSICS = REPO / "notebooks" / "agnr" / "physics"
if str(AGNR_PHYSICS) not in sys.path:
    sys.path.insert(0, str(AGNR_PHYSICS))
import agnr_lib

_W = {}


def seeds_for_width(width):
    return 1000 if width <= 14 else (300 if width <= 27 else 100)


def _init(H0, H1, energies, n_cells, spc, n_imp, v, formula, leads=None):
    if leads is None:
        leads = LeadCache(H0, H1, energies)
    _W.update(H0=H0, H1=H1, E=energies, n_cells=n_cells, spc=spc, n_imp=n_imp, v=v, formula=formula,
              leads=leads)


def _one(seed):
    s = impurity_shifts(_W["n_cells"], _W["spc"], _W["n_imp"], seed, _W["v"])
    return spectrum(_W["H0"], _W["H1"], _W["E"], s, _W["leads"], formula=_W["formula"])


def _init_agnr(m_width, n_imp, leads):
    _W.update(m_width=m_width, n_imp=n_imp, leads=leads)


def _one_agnr(seed):
    m_width, n_imp, leads = _W["m_width"], _W["n_imp"], _W["leads"]
    return np.array([
        agnr_lib.device_transmission(w, 1e-5, 1.0, 0.0, m_width, seed, n_imp, leads, nonlocal_mode="IL")
        for w in agnr_lib.energy_grid()
    ])


def generate(store, models, densities, spec, n_jobs=20, formula="legacy_trace", seeds=None):
    ctx = mp.get_context("spawn")
    e_t, wrote = spec.energies_t(), []
    for m in models:
        is_agnr = (m.material == "graphene-ideal" and m.edge == "armchair")
        is_zgnr = (m.material == "graphene-ideal" and m.edge == "zigzag")
        if is_agnr:
            model_formula = "agnr_lib_IL_1e-5"
            leads = agnr_lib.load_leads(m.width)
            grid_e = agnr_lib.energy_grid()
            pristine_path = store._dir(m.model_id) / "pristine.npy"
            if not pristine_path.exists():
                pris = agnr_lib.spectrum(m.width, leads, config=0, concentration=0, nonlocal_mode="IL", d=1e-5)
                store.write_pristine(m.model_id, grid_e, pris, formula=model_formula)
            for d in densities:
                n_imp = m.impurities_for_density(d)
                actual = n_imp / m.n_sites
                if store.has_cloud(m.model_id, actual):
                    continue
                sd = np.arange(seeds_for_width(m.width)) if seeds is None else np.asarray(list(seeds))
                t0 = time.time()
                with ctx.Pool(n_jobs, _init_agnr, (m.width, n_imp, leads)) as p:
                    spectra = np.array(p.map(_one_agnr, sd, chunksize=4))
                dt = time.time() - t0
                store.write_cloud(m.model_id, actual, n_imp, spectra, sd, grid_e, formula=model_formula)
                wrote.append((m.model_id, actual))
                sec_per = dt / len(sd)
                print(f"{m.model_id} density {actual:.4f}: {len(sd)} spectra ({model_formula}) in {dt:.2f}s ({sec_per:.4f} s/spec)", flush=True)
        else:
            h = hamiltonian_for(m)
            leads = LeadCache(h.H0, h.H1, e_t)
            model_formula = "caroli" if is_zgnr else formula
            pristine_path = store._dir(m.model_id) / "pristine.npy"
            if not pristine_path.exists():
                with ctx.Pool(1, _init, (h.H0, h.H1, e_t, 1, h.H0.shape[0], 0, 0.0, model_formula, leads)) as p:
                    store.write_pristine(m.model_id, e_t, p.map(_one, [0])[0], formula=model_formula)
            for d in densities:
                n_imp = m.impurities_for_density(d)
                actual = n_imp / m.n_sites
                if store.has_cloud(m.model_id, actual):
                    continue
                sd = np.arange(seeds_for_width(m.width)) if seeds is None else np.asarray(list(seeds))
                t0 = time.time()
                with ctx.Pool(n_jobs, _init, (h.H0, h.H1, e_t, m.n_cells, h.H0.shape[0], n_imp, m.impurity_v_t, model_formula, leads)) as p:
                    spectra = np.array(p.map(_one, sd, chunksize=4))
                dt = time.time() - t0
                store.write_cloud(m.model_id, actual, n_imp, spectra, sd, e_t, formula=model_formula)
                wrote.append((m.model_id, actual))
                sec_per = dt / len(sd)
                print(f"{m.model_id} density {actual:.4f}: {len(sd)} spectra ({model_formula}) in {dt:.2f}s ({sec_per:.4f} s/spec)", flush=True)
    return wrote


def _parse_widths(val):
    if not val:
        return []
    res = []
    for part in val.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-"))
            res.extend(range(lo, hi + 1))
        else:
            res.append(int(part))
    return sorted(set(res))


if __name__ == "__main__":
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="~/atlas_store/engine_v1")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--formula", default="legacy_trace")
    ap.add_argument("--n-seeds", type=int, default=None, help="Fixed number of seeds per model (e.g. 50 for smoke build)")
    ap.add_argument("--armchair-widths", default="5-16", help="Armchair widths (e.g. '5-16' or '20,27,31,40,50')")
    ap.add_argument("--zigzag-widths", default="4-12", help="Zigzag widths (e.g. '4-12' or '16,20,27,40,50')")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    arm_widths = _parse_widths(a.armchair_widths)
    zig_widths = _parse_widths(a.zigzag_widths)
    ms = ([make_model("graphene-ideal", "armchair", n) for n in arm_widths]
          + [make_model("graphene-ideal", "zigzag", n) for n in zig_widths])
    seeds = range(a.n_seeds) if a.n_seeds is not None else None
    generate(CloudStore(a.store), ms, [0.005, 0.01, 0.02, 0.04], InputSpec(), n_jobs=a.n_jobs, formula=a.formula, seeds=seeds)
