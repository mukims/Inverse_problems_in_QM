#!/usr/bin/env python
"""Figures for docs/materials/<material>/figures/: ribbon structure, bands + clean transmission, disorder.

Run from the repo root (read-only on the stores):
  PYTHONPATH=notebooks/material_atlas:notebooks ~/miniconda3/envs/ml/bin/python docs/materials/make_figures.py
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from atlaslib.store import CloudStore  # noqa: E402
from tbribbon.bands import _bands  # noqa: E402
from tbribbon.disorder import impurity_shifts  # noqa: E402
from tbribbon.materials import hamiltonian_for, make_model  # noqa: E402

HERE = Path(__file__).resolve().parent
S = np.sqrt(3) / 2
STORE = {"graphene-ideal": ["~/atlas_store/engine_v1"],
         "square": ["~/atlas_store/novelty_v1"],
         "default": ["~/atlas_store/materials_ev_full", "~/atlas_store/materials_ev_v1"]}
FOLDER = {"graphene-ideal": "graphene", "hbn": "hbn", "phosphorene": "phosphorene", "mos2": "mos2",
          "triangular": "triangular", "square": "square"}
EDGES = {"square": ["strip"]}
WIDTHS = {"square": [10]}
ATOM = {"graphene-ideal": [("C", "#3d3d3d")], "hbn": [("B", "#e08a3c"), ("N", "#3b6fb6")],
        "phosphorene": [("P", "#c2702c")], "mos2": [("Mo", "#7a4f9a")],
        "triangular": [("site", "#4f7f8f")], "square": [("site", "#4f7f8f")]}
DENS_COL = {0.005: "#4c9be8", 0.01: "#3aa66b", 0.02: "#e0a03c", 0.04: "#d0453c"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def period(material, edge):
    if material in ("graphene-ideal", "hbn", "phosphorene"):
        return np.array([3.0, 0.0]) if edge == "armchair" else np.array([0.0, 2 * S])
    if material == "square":
        return np.array([1.0, 0.0])
    return np.array([1.0, 0.0]) if edge in ("zigzag", "strip") else np.array([0.0, 2 * S])


def store_for(mid, need_densities=4):
    mat = mid.split("/")[0]
    for root in STORE.get(mat, STORE["default"]):
        st = CloudStore(os.path.expanduser(root))
        if (st.root / mid / "pristine.npy").exists() and len(st.densities(mid)) >= need_densities:
            return st
    for root in STORE.get(mat, STORE["default"]):
        st = CloudStore(os.path.expanduser(root))
        if (st.root / mid / "pristine.npy").exists():
            return st
    return None


def draw_structure(ax, material, edge, N, n_show_len=12.0, density=0.04, seed=0):
    m = make_model(material, edge, N)
    h = hamiltonian_for(m)
    k = m.orbitals_per_site
    pos = h.positions
    per = period(material, edge)
    n_cells = int(np.ceil(n_show_len / np.linalg.norm(per)))
    rot = per[1] > per[0]                                   # draw transport horizontally
    tr = (lambda p: np.column_stack([p[:, 1], -p[:, 0]])) if rot else (lambda p: p)
    n_atoms = len(pos)
    A0 = np.abs(h.H0).reshape(n_atoms, k, n_atoms, k).sum((1, 3)) > 1e-12
    A1 = np.abs(h.H1).reshape(n_atoms, k, n_atoms, k).sum((1, 3)) > 1e-12
    for c in range(n_cells):
        P, Pn = pos + c * per, pos + (c + 1) * per
        for i in range(n_atoms):
            for j in range(n_atoms):
                for (mask, Q) in ((A0, P), (A1, Pn)):
                    if mask[i, j] and (mask is A1 and c < n_cells - 1 or mask is A0 and i < j):
                        d = Q[j] - P[i]
                        if abs(np.linalg.norm(d) - 1) > 1e-3:
                            continue                       # draw nearest-neighbour bonds only
                        col, lw = "#9a9a9a", 1.4
                        if material == "phosphorene":
                            col, lw = (("#c2702c", 2.2) if abs(d[1]) < 1e-3 else ("#9a9a9a", 1.4))
                        seg = tr(np.vstack([P[i], Q[j]]))
                        ax.plot(seg[:, 0], seg[:, 1], color=col, lw=lw, zorder=1)
    shifts = impurity_shifts(n_cells, h.H0.shape[0], max(1, int(round(density * n_cells * n_atoms))), seed,
                             1.0, k).reshape(n_cells, n_atoms, k)[:, :, 0]
    for c in range(n_cells):
        P = tr(pos + c * per)
        kinds = ATOM[material]
        for idx, (name, col) in enumerate(kinds):
            sel = (h.sublattice == idx) if len(kinds) > 1 else np.ones(n_atoms, bool)
            ax.scatter(P[sel, 0], P[sel, 1], s=34, color=col, edgecolor="white", linewidth=0.6, zorder=3,
                       label=name if c == 0 else None)
        imp = shifts[c] != 0
        ax.scatter(P[imp, 0], P[imp, 1], s=150, facecolor="none", edgecolor="#d0453c", linewidth=1.6, zorder=4,
                   label=f"impurity (example, {100 * density:.0f}%)" if c == 0 else None)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{edge}, N = {N}  ({m.n_sites // 100} atoms per cell; {n_cells} of 100 cells shown)", fontsize=9)
    xr = ax.get_xlim()
    yb = ax.get_ylim()[0] - 0.9
    ax.annotate("", xy=(xr[1], yb), xytext=(xr[0], yb), arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
                annotation_clip=False)
    ax.text(np.mean(xr), yb - 0.35, "transport direction (leads on both ends)", ha="center", va="top", fontsize=8,
            color="#555")
    ax.set_ylim(yb - 1.0, ax.get_ylim()[1])


def fig_structure(material):
    edges = EDGES.get(material, ["armchair", "zigzag"])
    N = WIDTHS.get(material, [7])[0]
    fig, axes = plt.subplots(len(edges), 1, figsize=(8, 2.6 * len(edges) + 0.6))
    axes = np.atleast_1d(axes)
    for ax, edge in zip(axes, edges):
        draw_structure(ax, material, edge, N)
    handles, labels = axes[0].get_legend_handles_labels()
    if material == "phosphorene":
        handles += [plt.Line2D([], [], color="#c2702c", lw=2.2), plt.Line2D([], [], color="#9a9a9a", lw=1.4)]
        labels += ["t2 bond (3.665 eV)", "t1 bond (−1.220 eV); t3–t5 not drawn"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8)
    note = {"mos2": "Mo atoms only: sulfur is integrated out of the 3-band model. Each Mo carries d_z², d_xy, d_x²−y².",
            "phosphorene": "Puckered lattice drawn on its honeycomb projection (Ezawa mapping).",
            "hbn": "Boron carries +Δ = 3.625 eV, nitrogen −Δ.",
            "graphene-ideal": "One p_z orbital per carbon; nearest-neighbour hopping t = 2.7 eV.",
            "triangular": "Toy lattice: one orbital per site, six nearest neighbours, t = 1.",
            "square": "Toy strip: one orbital per site, t = 1."}[material]
    fig.suptitle(f"{FOLDER[material]}: ribbon structure. {note}", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig


def fig_bands_transmission(material):
    edges = EDGES.get(material, ["armchair", "zigzag"])
    widths = WIDTHS.get(material, [7, 9, 14, 27])
    fig, axes = plt.subplots(len(edges), 2, figsize=(9, 3.4 * len(edges)), gridspec_kw={"width_ratios": [1, 1.6]},
                             squeeze=False)
    cmap = plt.get_cmap("viridis")
    for r, edge in enumerate(edges):
        nb = widths[min(1, len(widths) - 1)]
        m = make_model(material, edge, nb)
        h = hamiltonian_for(m)
        ks = np.linspace(-np.pi, np.pi, 301)
        b = _bands(h.H0, h.H1, 301) * m.t_ev
        top = m.band_top_t * m.t_ev
        ax = axes[r, 0]
        ax.plot(ks / np.pi, b, color="#3b6fb6", lw=0.8)
        ax.set_ylim(0, top * 1.06)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("k (π / period)")
        ax.set_ylabel("E (eV)")
        ax.set_title(f"{edge} N = {nb}: bands (E ≥ 0)", fontsize=9)
        ax = axes[r, 1]
        for i, n in enumerate(widths):
            mid = make_model(material, edge, n).model_id
            st = store_for(mid, need_densities=0)
            if st is None:
                continue
            e_t, pris = st.read_pristine(mid)
            mm = make_model(material, edge, n)
            ax.plot(np.round(pris, 3), e_t * mm.t_ev, color=cmap(i / max(1, len(widths) - 1)), lw=1.1, label=f"N = {n}")
        ax.set_ylim(0, top * 1.06)
        ax.set_xlabel("clean transmission T (G₀)")
        ax.set_title(f"{edge}: clean T(E) by width (T = open channels)", fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def fig_disorder(material):
    edges = EDGES.get(material, ["armchair", "zigzag"])
    n = WIDTHS.get(material, [9])[0] if material == "square" else 9
    fig, axes = plt.subplots(1, len(edges), figsize=(5.2 * len(edges), 3.4), squeeze=False)
    for ax, edge in zip(axes[0], edges):
        m = make_model(material, edge, n)
        st = store_for(m.model_id)
        e_t, pris = st.read_pristine(m.model_id)
        E = e_t * m.t_ev
        live = E <= m.band_top_t * m.t_ev + 0.05
        ax.plot(E[live], np.round(pris, 3)[live], color="black", lw=1.2, label="clean")
        n_seeds = 0
        for d in st.densities(m.model_id):
            c, s = st.read_cloud(m.model_id, d)
            n_seeds = len(s)
            c = np.round(c[:, live], 3)
            col = DENS_COL[min(DENS_COL, key=lambda x: abs(x - d))]   # stored density is n_imp / n_atoms
            lo, med, hi = np.percentile(c, [10, 50, 90], axis=0)
            ax.fill_between(E[live], lo, hi, color=col, alpha=0.18, lw=0)
            ax.plot(E[live], med, color=col, lw=1.0, label=f"{100 * d:.1f}% (median, 10–90%)")
        ax.set_ylim(0, max(1.0, pris[live].max()) * 1.12)
        ax.set_xlim(0, E[live][-1])
        ax.set_xlabel("E (eV)")
        ax.set_ylabel("T (G₀)")
        ax.set_title(f"{edge} N = {n}: {n_seeds} configurations per density ({st.root.name})", fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    note = " Legacy-formula spikes above the clean value are clipped by the axis." if material in ("graphene-ideal", "square") else ""
    fig.suptitle(f"{FOLDER[material]}: transmission with impurities.{note}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main():
    for material, folder in FOLDER.items():
        out = HERE / folder / "figures"
        out.mkdir(parents=True, exist_ok=True)
        for name, fn in (("structure", fig_structure), ("bands_transmission", fig_bands_transmission),
                         ("disorder", fig_disorder)):
            fig = fn(material)
            fig.savefig(out / f"{name}.png", dpi=140)
            plt.close(fig)
            print(f"wrote {out / name}.png", flush=True)


if __name__ == "__main__":
    main()
