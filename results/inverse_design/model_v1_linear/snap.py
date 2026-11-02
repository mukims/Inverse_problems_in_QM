"""Snapshot of the loss curves: every run so far, validation (solid) and train (dotted), three panels."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = Path("/run/media/shardul/storage/machine_learning/transmission_github/transmissions/results/inverse_design")
runs = [("model_v1", "model_v1 (log input, Adam)", "#8b949e"), ("model_v1_nogn", "no GroupNorm", "#c9a227"),
        ("model_v1_sched", "weight decay + schedule (stopped)", "#9fb8e6"), ("model_v1_linear", "linear input", "#d1552b")]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (k, title) in zip(axes, [("loss", "Total validation loss"), ("ce", "Classifier (cross-entropy)"), ("flow", "Values (flow matching)")]):
    for name, label, c in runs:
        h = json.loads((R / name / "history.json").read_text())
        v = h[f"val_{k}"]
        lw = 2.4 if name == "model_v1_linear" else 1.4
        ax.plot(range(len(v)), v, color=c, lw=lw, label=label)
        if name in ("model_v1", "model_v1_linear"):
            ax.plot(range(len(v)), h[f"train_{k}"], color=c, lw=1, ls=":")
            b = int(min(range(len(v)), key=v.__getitem__))
            ax.plot(b, v[b], "o", color=c, ms=6)
            ax.annotate(f"{v[b]:.3f}", (b, v[b]), textcoords="offset points", xytext=(4, -12), fontsize=8, color=c)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(top={"loss": 5.0, "ce": 3.0, "flow": 2.3}[k])
axes[0].legend(fontsize=8, frameon=False)
n = len(json.loads((R / "model_v1_linear/history.json").read_text())["val_loss"])
fig.suptitle(f"Validation loss, lower is better (solid = validation, dotted = train for model_v1 and linear; dots = best). "
             f"Linear input: {n} epochs, finished", fontsize=11, x=0.01, ha="left")
fig.tight_layout()
fig.savefig("loss_snapshot.png", dpi=130)
print("ok", n)
