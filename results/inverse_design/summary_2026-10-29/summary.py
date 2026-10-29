"""Validation accuracy of every model compared on 2026-10-29 (from the compare_inverse / compare_nogn tables)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
rows = [  # name, top state, width mode, val loss
    ("model_v1\n(log, Adam)", 0.520, 0.350, 3.389), ("no GroupNorm", 0.480, 0.322, 3.734),
    ("linear input", 0.532, 0.334, 3.318), ("+ depth", 0.554, 0.358, 3.184),
    ("best MLP\n(band-top)", 0.533, 0.309, 3.496), ("+ depth\n+ band-top", 0.580, 0.377, 3.066)]
names, st, wd, vl = zip(*rows)
x = np.arange(len(rows)); w = 0.38
fig, ax = plt.subplots(figsize=(11, 5))
b1 = ax.bar(x - w / 2, st, w, color="#2f6fdf", label="top state right")
b2 = ax.bar(x + w / 2, wd, w, color="#e3a33b", label="width right (sample mode)")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006, f"{100 * b.get_height():.1f}%", ha="center", fontsize=8)
for i, v in enumerate(vl):
    ax.text(i, 0.02, f"val loss {v:.3f}", ha="center", fontsize=8, color="white")
ax.set_xticks(x, names, fontsize=9)
ax.set_ylim(0, 0.68)
ax.set_yticks(np.linspace(0, 0.6, 7), [f"{round(100 * t)}%" for t in np.linspace(0, 0.6, 7)], fontsize=8)
ax.set_ylabel("15,000 validation draws")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("Depth + band-top normalisation is the best model: 58.0% state, 37.7% width (model_v1: 52.0%, 35.0%)",
             fontsize=11, fontweight="bold", loc="left")
fig.tight_layout()
fig.savefig("summary.png", dpi=140)
