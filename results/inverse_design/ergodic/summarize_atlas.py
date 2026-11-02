"""ERGO-1 part A summary: atlas_clouds.csv by material and impurity count, and the figure ergodic_atlas.png.

    python results/inverse_design/ergodic/summarize_atlas.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
BINS = ((1, 15), (16, 40), (41, 100), (101, 10 ** 6))
STEP_EV = 0.02
COLUMNS = ("hwhm", "subgrid", "tau", "n_eff", "participation", "plateau", "pc1_share", "pc1_r2_arrangement",
           "R_mean", "R_var", "g_per_channel", "var_T")
# reference categorical palette (dataviz skill, light mode), slots 1-4 in fixed order; markers as secondary encoding
STYLE = {"germanene": ("#2a78d6", "o"), "silicene": ("#eb6834", "s"), "hbn": ("#1baf7a", "^"),
         "kagome": ("#eda100", "D")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load():
    rows = list(csv.DictReader(open(OUT / "atlas_clouds.csv")))
    for r in rows:
        for k, v in r.items():
            if k not in ("model_id", "material", "edge"):
                r[k] = v == "True" if v in ("True", "False") else float(v) if v else np.nan   # no _config.npz
    return rows


def table(rows):
    lines = [f"{'material':10s} {'n_imp':>9s} {'clouds':>6s} | " + " ".join(f"{c[:11]:>11s}" for c in COLUMNS)]
    summary = {}
    for mat in STYLE:
        for lo, hi in BINS:
            sel = [r for r in rows if r["material"] == mat and lo <= r["n_imp"] <= hi]
            if not sel:
                continue
            med = {c: float(np.nanmedian([float(r[c]) for r in sel])) for c in COLUMNS}
            med["hwhm_ev"] = med["hwhm"] * STEP_EV
            summary[f"{mat} n {lo}-{hi}"] = dict(clouds=len(sel), **med)
            lines.append(f"{mat:10s} {f'{lo}-{hi}':>9s} {len(sel):6d} | " + " ".join(f"{med[c]:11.3f}" for c in COLUMNS))
    allr = {c: float(np.nanmedian([float(r[c]) for r in rows])) for c in COLUMNS}
    lines.append(f"{'all':10s} {'':>9s} {len(rows):6d} | " + " ".join(f"{allr[c]:11.3f}" for c in COLUMNS))
    summary["all"] = dict(clouds=len(rows), **allr)
    return summary, lines


def figure(rows):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    panels = (("pc1_share", "Variance in the top mode of one spectrum", "share of residual variance"),
              ("n_eff", "Independent samples per spectrum (N_eff = channels / tau)", "N_eff"),
              ("pc1_r2_arrangement", "Top mode explained by the impurity arrangement", "R²"))
    for ax, (key, title, ylabel) in zip(axes, panels):
        for mat, (color, marker) in STYLE.items():
            sel = [r for r in rows if r["material"] == mat]
            ax.scatter([r["n_imp"] for r in sel], [r[key] for r in sel], s=26, color=color, marker=marker,
                       edgecolor="white", linewidth=0.8, label=mat, zorder=3)
        ax.set_xscale("log")
        if key == "n_eff":
            ax.set_yscale("log")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("impurities in the device", color=MUTED)
        ax.set_ylabel(ylabel, color=MUTED)
        ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.savefig(OUT / "ergodic_atlas.png", dpi=150, facecolor="#fcfcfb")


def main():
    rows = load()
    summary, lines = table(rows)
    cal = json.loads((OUT / "atlas_calibration.json").read_text())
    lines.append("\nlabel-free single-spectrum estimators against the ensemble truth (50 seeds per cloud):")
    for k, v in cal.items():
        lines.append(f"  {k:9s} rho1 bias {v['rho1_bias']:+.3f}, MAE {v['rho1_mae']:.3f}, corr {v['rho1_corr']:+.3f}; "
                     f"log tau bias {v['tau_log_bias']:+.3f}, corr {v['tau_log_corr']:+.3f}")
    print("\n".join(lines))
    (OUT / "atlas_summary.txt").write_text("\n".join(lines) + "\n")
    (OUT / "atlas_summary.json").write_text(json.dumps(summary, indent=1))
    figure(rows)
    print("wrote atlas_summary.{txt,json} and ergodic_atlas.png")


if __name__ == "__main__":
    main()
