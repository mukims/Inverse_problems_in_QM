"""Generate publication-grade Figure 3: Width Scaling."""
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

print("Generating Figure 3: Quantum Confinement vs Ribbon Width...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0f172a')

# 1. Graphene widths
ax1.set_facecolor('#1e293b')
g_widths = [5, 7, 9, 14, 20]
g_cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(g_widths)))
for w, col in zip(g_widths, g_cmap):
    mid = f"graphene-ideal/armchair/N{w}"
    if mid in store.models():
        e, p = store.read_pristine(mid)
        e_ev = e * 2.7
        v = (e_ev >= 0) & (e_ev <= 4.0)
        ax1.plot(e_ev[v], p[v], color=col, lw=1.8, label=f'N = {w}')
ax1.set_title("Graphene Armchair: Width $N$", fontsize=11, fontweight='bold', color='#f8fafc')
ax1.set_xlabel("Energy (eV)", fontsize=10, color='#94a3b8')
ax1.set_ylabel(r"$T(E)\;[G_0]$", fontsize=10, color='#94a3b8')
ax1.legend(fontsize=8, facecolor='#0f172a', edgecolor='#334155')
ax1.grid(True, ls=':', color='#334155')

# 2. MoS2 widths
ax2.set_facecolor('#1e293b')
m_widths = [7, 9, 14]
m_cmap = plt.cm.plasma(np.linspace(0.3, 0.9, len(m_widths)))
for w, col in zip(m_widths, m_cmap):
    mid = f"mos2/armchair/N{w}"
    if mid in store.models():
        e, p = store.read_pristine(mid)
        v = (e >= 0) & (e <= 4.0)
        ax2.plot(e[v], p[v], color=col, lw=2.0, label=f'N = {w}')
ax2.set_title("MoS2 Armchair: Width $N$", fontsize=11, fontweight='bold', color='#f8fafc')
ax2.set_xlabel("Energy (eV)", fontsize=10, color='#94a3b8')
ax2.legend(fontsize=8, facecolor='#0f172a', edgecolor='#334155')
ax2.grid(True, ls=':', color='#334155')

# 3. Kagome widths
ax3.set_facecolor('#1e293b')
k_widths = [7, 9, 14]
k_cmap = plt.cm.summer(np.linspace(0.2, 0.8, len(k_widths)))
for w, col in zip(k_widths, k_cmap):
    mid = f"kagome/armchair/N{w}"
    if mid in store.models():
        e, p = store.read_pristine(mid)
        v = (e >= 0) & (e <= 4.0)
        ax3.plot(e[v], p[v], color=col, lw=2.0, label=f'N = {w}')
ax3.set_title("Kagome Armchair: Width $N$", fontsize=11, fontweight='bold', color='#f8fafc')
ax3.set_xlabel("Energy (eV)", fontsize=10, color='#94a3b8')
ax3.legend(fontsize=8, facecolor='#0f172a', edgecolor='#334155')
ax3.grid(True, ls=':', color='#334155')

plt.tight_layout()
fig3_path = out_dir / "figure3_width_scaling.png"
plt.savefig(fig3_path, dpi=200, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print(f"Figure 3 saved to {fig3_path}")
