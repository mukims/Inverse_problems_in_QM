"""Generate publication-grade Figure 2: Disorder Evolution."""
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

print("Generating Figure 2: Disorder Evolution Comparison...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#0f172a')
demo_systems = [
    ("graphene-ideal/armchair/N9", "Graphene Armchair N9 (Dirac Metal)", 4.0),
    ("mos2/armchair/N9", "MoS2 Armchair N9 (3-Band TMD)", 14.0),
    ("hbn/armchair/N9", "hBN Armchair N9 (Gapped Insulator)", 4.0),
    ("kagome/armchair/N9", "Kagome Armchair N9 (Flat-Band Lattice)", 9.0)
]

densities = [0.005, 0.01, 0.02, 0.04]
d_colors = ['#38bdf8', '#34d399', '#f59e0b', '#f43f5e']

for ax, (mid, title, y_max) in zip(axes.flatten(), demo_systems):
    ax.set_facecolor('#1e293b')
    e, pris = store.read_pristine(mid)
    is_graphene = mid.startswith("graphene-ideal")
    e_plot = e * 2.7 if is_graphene else e
    valid = (e_plot >= 0) & (e_plot <= 8.32)
    
    # Plot pristine
    ax.plot(e_plot[valid], pris[valid], color='#ffffff', lw=2.2, label='Pristine (0%)', zorder=5)
    
    # Plot disorder clouds
    for d, c_col in zip(densities, d_colors):
        cloud, _ = store.read_cloud(mid, d)
        med = np.median(cloud, axis=0)
        p10 = np.percentile(cloud, 10, axis=0)
        p90 = np.percentile(cloud, 90, axis=0)
        
        ax.plot(e_plot[valid], med[valid], color=c_col, lw=1.4, alpha=0.9, label=f'd = {d*100:.1f}%')
        ax.fill_between(e_plot[valid], p10[valid], p90[valid], color=c_col, alpha=0.1)

    ax.set_title(title, fontsize=12, fontweight='bold', color='#f8fafc', pad=8)
    ax.set_xlabel("Energy (eV)", fontsize=10, color='#94a3b8')
    ax.set_ylabel(r"Conductance $T(E)\;[G_0]$", fontsize=10, color='#94a3b8')
    ax.tick_params(colors='#94a3b8', labelsize=9)
    ax.grid(True, ls=':', color='#334155', alpha=0.6)
    ax.set_xlim(0, 8.32)
    ax.set_ylim(0, y_max * 1.05)
    ax.legend(loc='upper right', fontsize=8, facecolor='#0f172a', edgecolor='#334155')

plt.tight_layout()
fig2_path = out_dir / "figure2_disorder_evolution.png"
plt.savefig(fig2_path, dpi=200, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print(f"Figure 2 saved to {fig2_path}")
