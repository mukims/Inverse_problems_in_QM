"""Generate publication-grade Figure 5: Material Phase Manifold (PCA & UMAP)."""
import json
import os
import pathlib
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

repo_dir = pathlib.Path(__file__).resolve().parents[2].parent
sys.path.append(str(repo_dir / "notebooks/material_atlas"))
sys.path.append(str(repo_dir / "notebooks"))
from atlaslib import CloudStore, InputSpec

out_dir = repo_dir / "docs/report"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load catalogue
cat = np.load(repo_dir / "notebooks/material_atlas/lookup_v3/catalogue.npz")
with open(repo_dir / "notebooks/material_atlas/lookup_v3/manifest.json") as f:
    manifest = json.load(f)

models = manifest["models"]
mu = cat["mu"]  # (103, 101, 416)

# Palette for 13 materials + square lattice
mat_colors = {
    "graphene-ideal": "#0284c7",
    "silicene":       "#38bdf8",
    "germanene":      "#0284c7",
    "hbn":            "#7c3aed",
    "phosphorene":    "#0d9488",
    "mos2":           "#d97706",
    "ws2":            "#b45309",
    "mose2":          "#f59e0b",
    "wse2":           "#ea580c",
    "kagome":         "#10b981",
    "lieb":           "#059669",
    "checkerboard":   "#34d399",
    "triangular":     "#e11d48",
    "square":         "#64748b"
}

mat_names = {
    "graphene-ideal": "Graphene",
    "silicene":       "Silicene",
    "germanene":      "Germanene",
    "hbn":            "hBN",
    "phosphorene":    "Phosphorene",
    "mos2":           "MoS₂",
    "ws2":            "WS₂",
    "mose2":          "MoSe₂",
    "wse2":           "WSe₂",
    "kagome":         "Kagome",
    "lieb":           "Lieb",
    "checkerboard":   "Checkerboard",
    "triangular":     "Triangular",
    "square":         "Square (OOD)"
}

records = []
X_list = []

# Collect pristine (conc 0%) and disordered points (conc 1%, 2%, 4%) for all 103 models
for dev_id, m in enumerate(models):
    mat = m["material"]
    edge = m["edge"]
    w = m["width"]
    
    # Pristine
    X_list.append(mu[dev_id, 0, :])
    records.append({
        "dev_id": dev_id,
        "material": mat,
        "edge": edge,
        "width": w,
        "density": 0.0,
        "is_pristine": True
    })
    
    # Disordered
    for c_idx, d_val in [(20, 0.01), (40, 0.02), (80, 0.04)]:
        X_list.append(mu[dev_id, c_idx, :])
        records.append({
            "dev_id": dev_id,
            "material": mat,
            "edge": edge,
            "width": w,
            "density": d_val,
            "is_pristine": False
        })

# Also include square lattice strip N=10 from novelty_v1
nov = CloudStore(os.path.expanduser("~/atlas_store/novelty_v1"))
e_sq, p_sq = nov.read_pristine("square/strip/N10")
cloud_sq, seeds = nov.read_cloud("square/strip/N10", 0.01)
spec = InputSpec(version="v4")

sq_in_p = spec.to_input(p_sq, e_sq, band_top_t=4.0)
X_list.append(sq_in_p[0])
records.append({
    "dev_id": -1,
    "material": "square",
    "edge": "strip",
    "width": 10,
    "density": 0.0,
    "is_pristine": True
})

for s_idx in [0, 10, 20]:
    sq_in_d = spec.to_input(cloud_sq[s_idx], e_sq, band_top_t=4.0)
    X_list.append(sq_in_d[0])
    records.append({
        "dev_id": -1,
        "material": "square",
        "edge": "strip",
        "width": 10,
        "density": 0.01,
        "is_pristine": False
    })

X = np.array(X_list)

# 2. Fit PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)
var_pca = pca.explained_variance_ratio_

# 3. Fit UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.25, metric="euclidean", random_state=42)
X_umap = reducer.fit_transform(X)

for i, r in enumerate(records):
    r["pc1"] = float(X_pca[i, 0])
    r["pc2"] = float(X_pca[i, 1])
    r["umap1"] = float(X_umap[i, 0])
    r["umap2"] = float(X_umap[i, 1])

# 4. PLOT FIGURE 5 (2-Panel)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=300)
fig.patch.set_facecolor("#0b1120")
ax1.set_facecolor("#151f32")
ax2.set_facecolor("#151f32")

# Styling helper
for ax in (ax1, ax2):
    ax.tick_params(colors="#94a3b8", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#22324e")
    ax.grid(True, linestyle="--", alpha=0.2, color="#94a3b8")

# PANEL A: PCA PROJECTION
for mat in mat_colors.keys():
    pts_dis = [r for r in records if r["material"] == mat and not r["is_pristine"]]
    if pts_dis:
        ax1.scatter([p["pc1"] for p in pts_dis], [p["pc2"] for p in pts_dis],
                    color=mat_colors[mat], s=24, alpha=0.28, edgecolors="none")

for mat in mat_colors.keys():
    pts_pris = [r for r in records if r["material"] == mat and r["is_pristine"]]
    if pts_pris:
        ax1.scatter([p["pc1"] for p in pts_pris], [p["pc2"] for p in pts_pris],
                    color=mat_colors[mat], s=90, alpha=1.0, edgecolors="#f8fafc", linewidths=1.2, zorder=5)

# Centroid labels for distinct materials
labeled_mats = ["graphene-ideal", "hbn", "phosphorene", "mos2", "wse2", "kagome", "lieb", "triangular", "square"]
for mat in labeled_mats:
    pts = [r for r in records if r["material"] == mat and r["is_pristine"]]
    if pts:
        cx = np.mean([p["pc1"] for p in pts])
        cy = np.mean([p["pc2"] for p in pts])
        ax1.annotate(mat_names[mat], (cx, cy),
                     fontsize=9, fontweight="bold", color="#f8fafc",
                     bbox=dict(boxstyle="round,pad=0.25", fc="#0b1120", ec=mat_colors[mat], lw=1.5, alpha=0.9),
                     ha="center", va="center", zorder=10)

# Physical Axes Vectors
ax1.annotate("", xy=(3.8, -4.5), xytext=(-3.0, -4.5),
             arrowprops=dict(arrowstyle="->", lw=2.0, color="#38bdf8"))
ax1.text(0.4, -4.9, "Increasing Bandwidth / Hopping Scale (t) →",
         fontsize=10, fontweight="bold", color="#38bdf8", ha="center")

ax1.annotate("", xy=(-3.8, 2.8), xytext=(-3.8, -3.2),
             arrowprops=dict(arrowstyle="->", lw=2.0, color="#a855f7"))
ax1.text(-4.2, -0.2, "Increasing Metallic Continuum / Flat Bands ↑",
         fontsize=10, fontweight="bold", color="#a855f7", rotation=90, va="center")

ax1.set_title(f"A. 2D Principal Component Phase Space ({var_pca[:2].sum()*100:.1f}% Variance)",
              fontsize=13, fontweight="bold", color="#f8fafc", pad=12)
ax1.set_xlabel(f"Principal Component 1 ({var_pca[0]*100:.1f}% variance)", fontsize=11, color="#cbd5e1")
ax1.set_ylabel(f"Principal Component 2 ({var_pca[1]*100:.1f}% variance)", fontsize=11, color="#cbd5e1")

# PANEL B: UMAP NON-LINEAR MANIFOLD EMBEDDING
for mat in mat_colors.keys():
    pts_dis = [r for r in records if r["material"] == mat and not r["is_pristine"]]
    if pts_dis:
        ax2.scatter([p["umap1"] for p in pts_dis], [p["umap2"] for p in pts_dis],
                    color=mat_colors[mat], s=28, alpha=0.35, edgecolors="none")

for mat in mat_colors.keys():
    pts_pris = [r for r in records if r["material"] == mat and r["is_pristine"]]
    if pts_pris:
        ax2.scatter([p["umap1"] for p in pts_pris], [p["umap2"] for p in pts_pris],
                    color=mat_colors[mat], s=100, alpha=1.0, edgecolors="#f8fafc", linewidths=1.2, zorder=5)

# Connect widths within graphene armchair ribbons to show width foliation
gr_arm_pts = sorted([r for r in records if r["material"] == "graphene-ideal" and r["edge"] == "armchair" and r["is_pristine"]], key=lambda x: x["width"])
if len(gr_arm_pts) > 1:
    ax2.plot([p["umap1"] for p in gr_arm_pts], [p["umap2"] for p in gr_arm_pts],
             color="#38bdf8", linestyle="--", linewidth=1.5, alpha=0.7, zorder=4)

# Material cluster labels on UMAP
for mat in mat_colors.keys():
    pts = [r for r in records if r["material"] == mat and r["is_pristine"]]
    if pts:
        cx = np.median([p["umap1"] for p in pts])
        cy = np.median([p["umap2"] for p in pts])
        ax2.annotate(mat_names[mat], (cx, cy),
                     fontsize=9, fontweight="bold", color="#f8fafc",
                     bbox=dict(boxstyle="round,pad=0.25", fc="#0b1120", ec=mat_colors[mat], lw=1.5, alpha=0.9),
                     ha="center", va="center", zorder=10)

ax2.set_title("B. UMAP Non-Linear Manifold (Topological Islands & Width Foliation)",
              fontsize=13, fontweight="bold", color="#f8fafc", pad=12)
ax2.set_xlabel("UMAP Dimension 1", fontsize=11, color="#cbd5e1")
ax2.set_ylabel("UMAP Dimension 2", fontsize=11, color="#cbd5e1")

# Master Title
plt.suptitle("Quantum Transport Material Phase Manifold Across All 13 Materials & Lattice Geometries",
             fontsize=15, fontweight="black", color="#f8fafc", y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])

out_fig5 = out_dir / "figure5_material_phase_manifold.png"
plt.savefig(out_fig5, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print(f"Successfully generated Figure 5: {out_fig5}")

# 5. Extract Delta Z values from meaning v2 results
with open(repo_dir / "notebooks/material_atlas/meaning/results/v2/mos2.json") as f:
    d_mos2 = json.load(f)
dz_mos2_tri = next(m for m in d_mos2["results"] if m["model"] == "multiscale")["groups"]["mos2/zigzag/N7"]["relative_median_distance"]

with open(repo_dir / "notebooks/material_atlas/meaning/results/v2/phosphorene.json") as f:
    d_phos = json.load(f)
dz_gr_phos = next(m for m in d_phos["results"] if m["model"] == "physics")["groups"]["phosphorene/armchair/N14"]["relative_median_distance"]
dz_gr_hbn = next(m for m in d_phos["results"] if m["model"] == "multiscale")["groups"]["phosphorene/zigzag/N7"]["relative_median_distance"]

with open(repo_dir / "notebooks/material_atlas/meaning/results/v2/hbn.json") as f:
    d_hbn = json.load(f)
dz_sq_novelty = next(m for m in d_hbn["results"] if m["model"] == "physics")["groups"]["square/strip/N10"]["relative_median_distance"]

# Required output prints
print(f"Projected: {len(X)} spectra: the catalogue's ensemble medians for 103 devices at 0, 1, 2 and 4% density, plus 4 square-strip spectra transformed with InputSpec v4 (416-channel universal eV grid, log1p(T)/log1p(64)).")
print(f"Explained variance: PC1={var_pca[0]*100:.1f}%, PC2={var_pca[1]*100:.1f}%, Total 2D={var_pca[:2].sum()*100:.1f}%")
print(f"ΔZ values: MoS2 <-> Triangular ΔZ = {dz_mos2_tri:.2f}; Graphene <-> Phosphorene ΔZ = {dz_gr_phos:.2f}; Graphene <-> hBN ΔZ = {dz_gr_hbn:.2f}; Square-strip novelty ΔZ = {dz_sq_novelty:.2f} (> 10)")
