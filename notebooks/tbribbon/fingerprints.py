#!/usr/bin/env python
"""Clean (pristine) fingerprints of registered models on the shared 0-4 t grid."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from atlaslib import CloudStore, InputSpec
from tbribbon.leads import LeadCache
from tbribbon.materials import hamiltonian_for, make_model
from tbribbon.transport import spectrum


def run(store, models, spec, formula="caroli", out_png=None):
    e_t = spec.energies_t()
    for m in models:
        h = hamiltonian_for(m)
        T = spectrum(h.H0, h.H1, e_t, np.zeros((1, h.H0.shape[0])), LeadCache(h.H0, h.H1, e_t), formula=formula)
        store.write_pristine(m.model_id, e_t, T)
        print(f"{m.model_id}: max T {T.max():.2f}, band top {m.band_top_t} t", flush=True)
    if out_png:
        fig, ax = plt.subplots(figsize=(10, 5))
        for m in models:
            ax.plot(e_t, store.read_pristine(m.model_id)[1], lw=1, label=m.model_id)
        ax.set_xlabel("E (t)"); ax.set_ylabel("T (G0)"); ax.legend(fontsize=6, ncol=3)
        fig.tight_layout(); fig.savefig(out_png, dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="~/atlas_store/engine_v1")
    ap.add_argument("--formula", default="caroli")
    a = ap.parse_args()
    ms = ([make_model("graphene-ideal", "armchair", n) for n in list(range(5, 17)) + [27, 50]]
          + [make_model("graphene-ideal", "zigzag", n) for n in list(range(4, 13)) + [27, 50]]
          + [make_model("square", "strip", 10)])
    run(CloudStore(a.store), ms, InputSpec(), a.formula, out_png="notebooks/tbribbon/fingerprints_ideal.png")
