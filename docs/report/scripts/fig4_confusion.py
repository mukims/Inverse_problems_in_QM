"""Generate publication-grade Figure 4: 13x13 Material Confusion Heatmap."""
import os
import pathlib
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

repo_dir = pathlib.Path(__file__).resolve().parents[2].parent
results_path = repo_dir / "notebooks/material_atlas/lookup_v3/results.json"
out_dir = repo_dir / "docs/report"
out_dir.mkdir(parents=True, exist_ok=True)

with open(results_path) as f:
    r3 = json.load(f)

conf = r3["t1"]["material_confusion"]
mats = sorted(conf.keys())
n = len(mats)
matrix = np.zeros((n, n))

for i, r in enumerate(mats):
    for j, c in enumerate(mats):
        matrix[i, j] = conf[r].get(c, 0.0)

fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0f172a")
ax.set_facecolor("#1e293b")

im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(mats, rotation=45, ha="right", fontsize=9, color="#94a3b8")
ax.set_yticklabels(mats, fontsize=9, color="#94a3b8")

for i in range(n):
    for j in range(n):
        val = matrix[i, j]
        if val > 0:
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color="#ffffff" if val > 50 else "#38bdf8", fontsize=8, fontweight="bold")

ax.set_title("13 x 13 Material Confusion Matrix (BUILD-29: 61,800 Signatures)", fontsize=12, fontweight="bold", color="#f8fafc", pad=12)
ax.set_xlabel("Predicted Material", fontsize=10, color="#94a3b8", labelpad=8)
ax.set_ylabel("True Material", fontsize=10, color="#94a3b8", labelpad=8)

plt.tight_layout()
fig4_path = out_dir / "figure4_confusion_heatmap.png"
plt.savefig(fig4_path, dpi=200, bbox_inches="tight", facecolor="#0f172a")
plt.close()
print(f"Figure 4 saved to {fig4_path}")
