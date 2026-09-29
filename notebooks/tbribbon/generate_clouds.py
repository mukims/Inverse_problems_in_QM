#!/usr/bin/env python
"""Parallel, resumable generation of disorder clouds into a CloudStore.
Set OMP_NUM_THREADS=1 before launching to avoid BLAS oversubscription."""
import argparse
import os
from multiprocessing import Pool

import numpy as np

from atlaslib import CloudStore, InputSpec
from tbribbon.disorder import impurity_shifts
from tbribbon.leads import LeadCache
from tbribbon.materials import hamiltonian_for, make_model
from tbribbon.transport import spectrum

DEFAULT_FORMULA = "legacy_trace"          # D1: settled as legacy_trace
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


def generate(store, models, densities, spec, n_jobs=20, formula=DEFAULT_FORMULA, seeds=None):
    e_t, wrote = spec.energies_t(), []
    for m in models:
        h = hamiltonian_for(m)
        leads = LeadCache(h.H0, h.H1, e_t)
        pristine_path = store._dir(m.model_id) / "pristine.npy"
        if not pristine_path.exists():
            with Pool(1, _init, (h.H0, h.H1, e_t, 1, h.H0.shape[0], 0, 0.0, formula, leads)) as p:
                store.write_pristine(m.model_id, e_t, p.map(_one, [0])[0])
        for d in densities:
            n_imp = m.impurities_for_density(d)
            actual = n_imp / m.n_sites
            if store.has_cloud(m.model_id, actual):
                continue
            sd = np.arange(seeds_for_width(m.width)) if seeds is None else np.asarray(list(seeds))
            with Pool(n_jobs, _init, (h.H0, h.H1, e_t, m.n_cells, h.H0.shape[0], n_imp, m.impurity_v_t, formula, leads)) as p:
                spectra = np.array(p.map(_one, sd, chunksize=4))
            store.write_cloud(m.model_id, actual, n_imp, spectra, sd, e_t)
            wrote.append((m.model_id, actual))
            print(f"{m.model_id} density {actual:.4f}: {len(sd)} spectra", flush=True)
    return wrote


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="~/atlas_store/engine_v1")
    ap.add_argument("--n-jobs", type=int, default=20)
    ap.add_argument("--formula", default=DEFAULT_FORMULA)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ms = ([make_model("graphene-ideal", "armchair", n) for n in range(5, 17)]
          + [make_model("graphene-ideal", "zigzag", n) for n in range(4, 13)])
    generate(CloudStore(a.store), ms, [0.005, 0.01, 0.02, 0.04], InputSpec(), n_jobs=a.n_jobs, formula=a.formula)
