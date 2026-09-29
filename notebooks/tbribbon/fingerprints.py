#!/usr/bin/env python
"""Clean (pristine) fingerprints of registered models with consistent backends."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from atlaslib import CloudStore, InputSpec
from tbribbon.leads import LeadCache
from tbribbon.materials import hamiltonian_for, make_model
from tbribbon.transport import spectrum

REPO = Path(__file__).resolve().parents[2]
AGNR_PHYSICS = REPO / "notebooks" / "agnr" / "physics"
if str(AGNR_PHYSICS) not in sys.path:
    sys.path.insert(0, str(AGNR_PHYSICS))
import agnr_lib


def run(store, models, spec, out_png=None):
    e_t_spec = spec.energies_t()
    for m in models:
        if m.material == "graphene-ideal" and m.edge == "armchair":
            formula = "agnr_lib_IL_1e-5"
            leads = agnr_lib.load_leads(m.width)
            grid_e = agnr_lib.energy_grid()
            T = agnr_lib.spectrum(m.width, leads, config=0, concentration=0, nonlocal_mode="IL", d=1e-5)
            store.write_pristine(m.model_id, grid_e, T, formula=formula)
            print(f"{m.model_id}: max T {T.max():.2f}, band top {m.band_top_t} t ({formula})", flush=True)
        elif m.material == "square":
            formula = "legacy_trace"
            h = hamiltonian_for(m)
            leads = LeadCache(h.H0, h.H1, e_t_spec)
            T = spectrum(h.H0, h.H1, e_t_spec, np.zeros((1, h.H0.shape[0])), leads, formula=formula)
            store.write_pristine(m.model_id, e_t_spec, T, formula=formula)
            print(f"{m.model_id}: max T {T.max():.2f}, band top {m.band_top_t} t ({formula})", flush=True)
        else:
            formula = "caroli"
            h = hamiltonian_for(m)
            leads = LeadCache(h.H0, h.H1, e_t_spec)
            T = spectrum(h.H0, h.H1, e_t_spec, np.zeros((1, h.H0.shape[0])), leads, formula=formula)
            store.write_pristine(m.model_id, e_t_spec, T, formula=formula)
            print(f"{m.model_id}: max T {T.max():.2f}, band top {m.band_top_t} t ({formula})", flush=True)

    if out_png:
        fig, ax = plt.subplots(figsize=(10, 5))
        for m in models:
            e, T = store.read_pristine(m.model_id)
            ax.plot(e, T, lw=1, label=m.model_id)
        ax.set_xlabel("E (t)"); ax.set_ylabel("T (G0)"); ax.legend(fontsize=6, ncol=3)
        fig.tight_layout(); fig.savefig(out_png, dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="~/atlas_store/engine_v1")
    a = ap.parse_args()
    ms = ([make_model("graphene-ideal", "armchair", n) for n in list(range(5, 17)) + [27]]
          + [make_model("graphene-ideal", "zigzag", n) for n in list(range(4, 13)) + [27]]
          + [make_model("square", "strip", 10)])
    run(CloudStore(a.store), ms, InputSpec(), out_png="notebooks/tbribbon/fingerprints_ideal.png")
