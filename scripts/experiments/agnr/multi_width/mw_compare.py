#!/usr/bin/env python
"""
mw_compare.py — Assemble whatever technique results exist into one benchmark.

Reads every <tag>_metrics.json / <tag>_preds.npz in <out-dir>/mw_results and emits
a summary table plus plots. Missing techniques are simply skipped, so you can run
this after any subset of the four scripts.

It also warns if the stored predictions disagree on the test set, which is the
signature of results produced under different --samples-per-conc / --spectrum-len
(the split depends on both, so such results are not comparable).

Usage
-----
    python mw_compare.py
    python mw_compare.py --out-dir /path/to/run
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mw_common as mw
from mw_common import log, banner

ORDER = ["misfit", "xgboost", "mlp", "transformer"]
COLORS = {"misfit": "#7f7f7f", "xgboost": "#ff7f0e",
          "mlp": "#1f77b4", "transformer": "#2ca02c"}


def load_results(results_dir):
    found = {}
    for tag in ORDER:
        mpath = results_dir / f"{tag}_metrics.json"
        ppath = results_dir / f"{tag}_preds.npz"
        if not (mpath.exists() and ppath.exists()):
            continue
        with open(mpath) as fh:
            meta = json.load(fh)
        found[tag] = {"meta": meta, "npz": np.load(ppath)}
        log(f"  loaded {tag:12s} — written {meta.get('written', '?')}")
    return found


def check_consistency(found):
    """All techniques must share one test set, else the comparison is meaningless."""
    ref_tag, ref = next(iter(found.items()))
    ref_c = ref["npz"]["true_c"]
    ok = True
    for tag, r in found.items():
        tc = r["npz"]["true_c"]
        if len(tc) != len(ref_c) or not np.array_equal(tc, ref_c):
            log(f"  !! {tag} test set differs from {ref_tag} "
                f"({len(tc):,} vs {len(ref_c):,} samples) — re-run with matching "
                f"--samples-per-conc/--spectrum-len before trusting this table")
            ok = False
    if ok:
        log(f"  ✓ all {len(found)} technique(s) share an identical {len(ref_c):,}-sample test set")
    return ok


def summary_table(found):
    log("=" * 96)
    log(f"{'Technique':<26}{'Width Acc':>12}{'MAE':>10}{'RMSE':>10}"
        f"{'Max Err':>10}{'7-AGNR MAE':>13}{'9-AGNR MAE':>13}")
    log("-" * 96)
    rows = sorted(found.items(), key=lambda kv: kv[1]["meta"]["metrics"]["Conc_MAE_Overall"])
    for tag, r in rows:
        m = r["meta"]["metrics"]
        log(f"{r['meta']['display_name']:<26}{m['Width_Acc_Overall']:>11.3f}%"
            f"{m['Conc_MAE_Overall']:>10.3f}{m['Conc_RMSE_Overall']:>10.3f}"
            f"{m['Conc_Max_Error']:>10.2f}{m['Conc_MAE_7']:>13.3f}{m['Conc_MAE_9']:>13.3f}")
    log("=" * 96)
    best_tag, best = rows[0]
    log(f"Best concentration MAE: {best['meta']['display_name']} "
        f"({best['meta']['metrics']['Conc_MAE_Overall']:.3f})")
    return rows


def plot_scatter(found, out_path):
    n = len(found)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), dpi=140, squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, (tag, r) in zip(axes.flat, found.items()):
        z = r["npz"]
        true_c, pred_c, true_w = z["true_c"], z["pred_c"], z["true_w"]
        m7, m9 = true_w == 0, true_w == 1
        ax.scatter(true_c[m7], pred_c[m7], s=5, alpha=0.15, color=COLORS[tag], label="7-AGNR")
        ax.scatter(true_c[m9], pred_c[m9], s=5, alpha=0.15, color="#d62728", label="9-AGNR")
        ax.plot([0, 100], [0, 100], "k--", lw=1.4, alpha=0.7, label="Ideal")
        m = r["meta"]["metrics"]
        ax.set_title(f"{r['meta']['display_name']}\n"
                     f"MAE {m['Conc_MAE_Overall']:.2f} | RMSE {m['Conc_RMSE_Overall']:.2f}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("True concentration $c$")
        ax.set_ylabel(r"Predicted $\hat{c}$")
        ax.set_xlim(0, 102); ax.set_ylim(0, 102)
        ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    log(f"  wrote {out_path.name}")


def plot_error_dist(found, out_path):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    for tag, r in found.items():
        z = r["npz"]
        ax.hist(z["pred_c"] - z["true_c"], bins=80, range=(-15, 15), density=True,
                alpha=0.45, label=r["meta"]["display_name"], color=COLORS[tag])
    ax.axvline(0, color="k", ls="--", alpha=0.7)
    ax.set_xlabel(r"Prediction error ($\hat{c} - c$)")
    ax.set_ylabel("Probability density")
    ax.set_title("Multi-width error distributions (7-AGNR & 9-AGNR)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    log(f"  wrote {out_path.name}")


def plot_training_curves(found, out_path):
    have = {t: r for t, r in found.items() if "hist_val_mae" in r["npz"]}
    if not have:
        log("  (no training histories stored — skipping curves)")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=140)
    for tag, r in have.items():
        z, name = r["npz"], r["meta"]["display_name"]
        ax1.plot(z["hist_train_loss"], ls="--", color=COLORS[tag], label=f"{name} train")
        ax1.plot(z["hist_val_loss"], color=COLORS[tag], label=f"{name} val")
        ax2.plot(z["hist_val_mae"], color=COLORS[tag], label=name)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Huber + α·CE (normalised targets)")
    ax1.set_title("Training & validation loss", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation MAE (impurities)")
    ax2.set_title("Validation MAE — the selection metric", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    log(f"  wrote {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Assemble the multi-width benchmark.")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    out_dir, results_dir = mw.setup_run(args, "compare")
    banner("MULTI-WIDTH SUITE — COMPARATIVE BENCHMARK")
    log(f"Results dir: {results_dir}")

    found = load_results(results_dir)
    if not found:
        log("No results found. Run at least one of: mw_misfit.py, mw_xgboost.py, "
            "mw_mlp.py, mw_transformer.py")
        return

    check_consistency(found)
    summary_table(found)

    combined = {tag: r["meta"]["metrics"] for tag, r in found.items()}
    combined_path = results_dir / "mw_all_metrics.json"
    with open(combined_path, "w") as fh:
        json.dump(combined, fh, indent=2)
    log(f"✓ Combined metrics -> {combined_path.name}")

    plot_scatter(found, out_dir / "mw_scatter.png")
    plot_error_dist(found, out_dir / "mw_error_dist.png")
    plot_training_curves(found, out_dir / "mw_training_curves.png")
    log("[DONE] comparison complete")


if __name__ == "__main__":
    main()
