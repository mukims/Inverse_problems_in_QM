"""Generate publication-grade Figure 1: Pristine Transmission Spectra."""
import os
import pathlib
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

repo_dir = pathlib.Path(__file__).resolve().parents[2].parent
sys.path.append(str(repo_dir / "notebooks/material_atlas"))
sys.path.append(str(repo_dir / "notebooks"))
from atlaslib.store import MultiStore

# Set high-quality plotting aesthetics
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.0

out_dir = repo_dir / "docs/report"
out_dir.mkdir(parents=True, exist_ok=True)

store = MultiStore(os.path.expanduser("~/atlas_store/engine_v1"),
                   os.path.expanduser("~/atlas_store/materials_ev_full"),
                   os.path.expanduser("~/atlas_store/materials2_ev_full"))

materials = [
    ("graphene-ideal", "graphene-ideal/armchair/N9", "Dirac: Graphene (N9)", "#38bdf8"),
    ("silicene", "silicene/armchair/N9", "Dirac: Silicene (N9)", "#0ea5e9"),
    ("germanene", "germanene/armchair/N9", "Dirac: Germanene (N9)", "#0284c7"),
    ("hbn", "hbn/armchair/N9", "Insulator: hBN (N9)", "#818cf8"),
    ("phosphorene", "phosphorene/armchair/N9", "Semiconductor: Phosphorene (N9)", "#a855f7"),
    ("mos2", "mos2/armchair/N9", "TMD: MoS2 (N9)", "#f59e0b"),
    ("ws2", "ws2/armchair/N9", "TMD: WS2 (N9)", "#d97706"),
    ("mose2", "mose2/armchair/N9", "TMD: MoSe2 (N9)", "#b45309"),
    ("wse2", "wse2/armchair/N9", "TMD: WSe2 (N9)", "#fbbf24"),
    ("kagome", "kagome/armchair/N9", "Flat-Band: Kagome (N9)", "#10b981"),
    ("lieb", "lieb/strip/N9", "Flat-Band: Lieb (N9)", "#059669"),
    ("checkerboard", "checkerboard/strip/N9", "Flat-Band: Checkerboard (N9)", "#34d399"),
    ("triangular", "triangular/armchair/N9", "2D Metal: Triangular (N9)", "#f43f5e")
]

print("Generating Figure 1: Pristine Transmissions across 13 materials...")
fig, axes = plt.subplots(5, 3, figsize=(15, 16), facecolor='#0f172a')
axes = axes.flatten()

for idx, (mat, mid, title, color) in enumerate(materials):
    ax = axes[idx]
    ax.set_facecolor('#1e293b')
    e, p = store.read_pristine(mid)
    if mat == "graphene-ideal":
        e = e * 2.7
    
    # Clip energy to valid range
    valid = (e >= 0) & (e <= 8.32)
    ax.plot(e[valid], p[valid], color=color, lw=2.0, label='Pristine $T(E)$')
    ax.fill_between(e[valid], 0, p[valid], color=color, alpha=0.15)
    
    ax.set_title(title, fontsize=11, fontweight='bold', color='#f8fafc', pad=8)
    ax.set_xlabel("Energy (eV)", fontsize=9, color='#94a3b8')
    ax.set_ylabel(r"$T(E)\;[G_0]$", fontsize=9, color='#94a3b8')
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.grid(True, ls=':', color='#334155', alpha=0.6)
    ax.set_xlim(0, 8.32)
    ax.set_ylim(bottom=0)

# Hide unused axes
for i in range(len(materials), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
fig1_path = out_dir / "figure1_pristine_materials.png"
plt.savefig(fig1_path, dpi=200, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print(f"Figure 1 saved to {fig1_path}")
