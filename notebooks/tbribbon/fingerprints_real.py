#!/usr/bin/env python
"""Clean (pristine) fingerprints for realistic materials (hBN, Phosphorene, MoS2).

Computes Caroli transmission on the standard 0-4.0 t grid (InputSpec) for both armchair
and zigzag orientations across designated widths, saving spectra into CloudStore and
generating fingerprints_real.png.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure material_atlas and tbribbon are on PYTHONPATH
REPO = Path(__file__).resolve().parents[2]
if str(REPO / "notebooks" / "material_atlas") not in sys.path:
    sys.path.insert(0, str(REPO / "notebooks" / "material_atlas"))
if str(REPO / "notebooks") not in sys.path:
    sys.path.insert(0, str(REPO / "notebooks"))

from atlaslib import CloudStore, InputSpec
from tbribbon.leads import LeadCache
from tbribbon.materials import hamiltonian_for, make_model
from tbribbon.transport import spectrum


def run(store, models, spec, out_png=None):
    e_t_spec = spec.energies_t()
    for m in models:
        formula = "caroli"
        h = hamiltonian_for(m)
        leads = LeadCache(h.H0, h.H1, e_t_spec)
        T = spectrum(h.H0, h.H1, e_t_spec, np.zeros((1, h.H0.shape[0])), leads, formula=formula)
        store.write_pristine(m.model_id, e_t_spec, T, formula=formula)
        print(f"{m.model_id}: max T {T.max():.2f}, band top {m.band_top_t} t ({formula})", flush=True)

    if out_png:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
        mat_titles = {
            "hbn": "hBN (Galvani GW, 2Δ=7.25 eV)",
            "phosphorene": "Phosphorene (Rudenko 5-hop, GW)",
            "mos2": "MoS₂ (Liu 3-band GGA)",
        }
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(set(m.width for m in models))))
        width_colors = {w: c for w, c in zip(sorted(set(m.width for m in models)), colors)}

        for ax, mat in zip(axes, ["hbn", "phosphorene", "mos2"]):
            mat_models = [m for m in models if m.material == mat]
            for m in mat_models:
                e, T = store.read_pristine(m.model_id)
                ls = "-" if m.edge == "armchair" else "--"
                lbl = f"{m.edge[:3]} N={m.width}"
                ax.plot(e, T, lw=1.3, ls=ls, color=width_colors[m.width], label=lbl)
            ax.set_title(mat_titles[mat], fontsize=11, fontweight="bold")
            ax.set_xlabel("Energy E / t")
            ax.set_ylabel("Transmission T (G₀)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2, loc="upper right")

        fig.suptitle("Clean Fingerprints for Realistic Materials (Phase 3b)", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        print(f"Saved plot to {out_png}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="~/atlas_store/materials_v1")
    ap.add_argument("--out-png", default=str(REPO / "notebooks" / "tbribbon" / "fingerprints_real.png"))
    ap.add_argument("--widths", default="7,9,14,27")
    a = ap.parse_args()

    widths = [int(x.strip()) for x in a.widths.split(",") if x.strip()]
    models = []
    for mat in ["hbn", "phosphorene", "mos2"]:
        for edge in ["armchair", "zigzag"]:
            for w in widths:
                models.append(make_model(mat, edge, w))

    run(CloudStore(a.store), models, InputSpec(), out_png=a.out_png)
