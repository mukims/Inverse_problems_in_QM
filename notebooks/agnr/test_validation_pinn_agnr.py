"""
test_validation_pinn_agnr.py — Test & Validation Suite for the PINN AGNR Model
================================================================================

Usage
-----
This script is designed to be run cell-by-cell in an interactive environment
(Jupyter / VS Code Interactive / IPython) or executed as a standalone script.

It provides:
  1.  Setup — imports, paths, device selection
  2.  Data loading — pristine spectrum, manifest, dataset construction
  3.  Model loading — instantiate ImprovedConductanceCNN & load checkpoint
  4.  Build DifferentiableMisfit module (physics regularizer)
  5.  Validation-set evaluation (from the training split)
  6.  Held-out test-set evaluation (on-the-fly generation or pre-saved data)
  7.  Visualisations:
        a. Predicted vs True concentration scatter
        b. Per-concentration box-plot of errors
        c. Confusion-style heatmap
        d. Residual histogram
        e. Physics-consistency: misfit loss distribution
        f. Example spectra overlay (predicted vs reference)
  8.  Quantitative summary table
"""

# ======================================================================
# 1. SETUP
# ======================================================================

import os
import sys
import copy
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset

# ── Make sure the local module is importable ──
NOTEBOOK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NOTEBOOK_DIR))

from pinn_agnr import (
    ImprovedConductanceCNN,
    DifferentiableMisfit,
    NormalizedTransmissionsDataset,
    NormalizedTestDataset,
    build_misfit_module,
    test_pinn,
)

# ── Plotting style ──
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
})
sns.set_palette("muted")

# ── Paths ──
PROJECT_ROOT = NOTEBOOK_DIR.parents[1]  # transmissions/
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "transmission_results"
MANIFEST_PATH = NOTEBOOK_DIR / "manifest_agnr.csv"
LEADS_PATH = PROJECT_ROOT / "leads" / "agnr_7.npy"
PRISTINE_PATH = DATA_DIR / "pristine.npy"

# ── Hyper-parameters that must match training ──
SPECTRUM_LENGTH = 200
CONC_RANGE = np.arange(1, 50, 2)   # 25 concentrations: 1, 3, 5, …, 49
MISFIT_WEIGHT = 0.1
BATCH_SIZE = 64
VAL_SPLIT = 0.2

# ── Device ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ======================================================================
# 2. LOAD PRISTINE SPECTRUM
# ======================================================================

pristine = np.load(str(PRISTINE_PATH))
print(f"Pristine spectrum loaded — shape: {pristine.shape}")


# ======================================================================
# 3. BUILD DATASET
# ======================================================================

dataset = NormalizedTransmissionsDataset(
    manifest_file=str(MANIFEST_PATH),
    root_dir=str(DATA_DIR),
    pristine=pristine,
    spectrum_length=SPECTRUM_LENGTH,
)
print(f"Full dataset size: {len(dataset)}")


# ======================================================================
# 4. TRAIN / VAL SPLIT  (mirroring training code exactly)
# ======================================================================

n_val = int(len(dataset) * VAL_SPLIT)
n_train = len(dataset) - n_val

# We use a fixed seed so the split is reproducible across runs.
generator = torch.Generator().manual_seed(42)
train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train samples: {n_train}  |  Val samples: {n_val}")


# ======================================================================
# 5. BUILD MISFIT MODULE
# ======================================================================

# We need the ca() function from misfit_agnr.
# Since that lives inside the notebook environment, we re-build the
# misfit module from the raw data files directly.

def _build_misfit_from_files():
    """
    Build DifferentiableMisfit from pre-saved config-averaged spectra.
    Falls back to loading individual files if needed.
    """
    ref_list = []
    for con in CONC_RANGE:
        # Configuration-average over first 10 000 configs
        acc = []
        # Try to load up to 100 configs for speed (representative sample)
        sample_configs = min(100, 10000)
        for cfg in range(sample_configs):
            fpath = DATA_DIR / f"7_agnr_conc{int(con)}_cfg{cfg}.npy"
            if fpath.exists():
                spec = np.load(str(fpath)).astype(np.float32)[:SPECTRUM_LENGTH]
                spec = np.clip(spec, 0, pristine[:SPECTRUM_LENGTH])
                acc.append(spec)
        if len(acc) == 0:
            raise FileNotFoundError(f"No data files found for concentration {con}")
        avg_spec = np.mean(acc, axis=0)
        ref_normalised = avg_spec / (pristine[:SPECTRUM_LENGTH] + 1e-8)
        ref_list.append(ref_normalised)

    ref_spectra = torch.tensor(np.array(ref_list), dtype=torch.float32)
    concentrations = torch.tensor(CONC_RANGE, dtype=torch.float32)
    module = DifferentiableMisfit(ref_spectra, concentrations)
    return module.to(DEVICE)


print("Building DifferentiableMisfit module (this may take a moment)…")
misfit_module = _build_misfit_from_files()
print("Done!")


# ======================================================================
# 6. LOAD / INSTANTIATE MODEL
# ======================================================================

def load_model(checkpoint_path=None):
    """
    Load the ImprovedConductanceCNN.

    If `checkpoint_path` is None, we just create a fresh model
    (you'll need to train first or supply a path).
    """
    model = ImprovedConductanceCNN(input_length=SPECTRUM_LENGTH).to(DEVICE)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print(f"✓ Model loaded from: {checkpoint_path}")
    else:
        print("⚠ No checkpoint loaded — using freshly initialised model.")
        print("  Set `CHECKPOINT_PATH` or pass one to `load_model()`.")

    model.eval()
    return model


# ──  Set your checkpoint path here  ──────────────────────────────────
CHECKPOINT_PATH = None  # e.g. "pinn_agnr_best.pt"
model = load_model(CHECKPOINT_PATH)


# ======================================================================
# 7. EVALUATION HELPERS
# ======================================================================

@torch.no_grad()
def evaluate(model, loader, misfit_module=None, misfit_weight=MISFIT_WEIGHT):
    """
    Run the model on a DataLoader and collect predictions + labels.

    Returns:
        dict with keys: preds, labels, mse_losses, misfit_losses, total_losses
    """
    model.eval()
    all_preds, all_labels = [], []
    all_mse, all_misfit, all_total = [], [], []

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        preds = model(xb)
        preds_flat = preds.squeeze(-1)
        yb_flat = yb.squeeze(-1)

        mse = F.mse_loss(preds_flat, yb_flat, reduction="none")

        if misfit_module is not None:
            mis = misfit_module(xb, preds)
            total = mse.mean() + misfit_weight * mis
            all_misfit.append(mis.item())
        else:
            total = mse.mean()
            all_misfit.append(0.0)

        all_preds.append(preds_flat.cpu())
        all_labels.append(yb_flat.cpu())
        all_mse.append(mse.mean().item())
        all_total.append(total.item())

    return {
        "preds":  torch.cat(all_preds).numpy(),
        "labels": torch.cat(all_labels).numpy(),
        "mse_losses":   all_mse,
        "misfit_losses": all_misfit,
        "total_losses":  all_total,
    }


def build_results_df(results):
    """Convert evaluation dict into a tidy DataFrame."""
    df = pd.DataFrame({
        "true_conc":  results["labels"],
        "pred_conc":  results["preds"],
    })
    df["error"]     = df["pred_conc"] - df["true_conc"]
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = (df["abs_error"] / (df["true_conc"] + 1e-8)) * 100
    return df


def summary_table(df):
    """Per-concentration summary statistics."""
    summary = df.groupby("true_conc").agg(
        n_samples    = ("abs_error", "count"),
        pred_mean    = ("pred_conc", "mean"),
        pred_std     = ("pred_conc", "std"),
        mae          = ("abs_error", "mean"),
        rmse         = ("abs_error", lambda x: np.sqrt((x**2).mean())),
        max_error    = ("abs_error", "max"),
        median_error = ("abs_error", "median"),
        mape         = ("pct_error", "mean"),
    ).round(3)
    return summary


# ======================================================================
# 8. RUN EVALUATION
# ======================================================================

print("\n" + "="*65)
print("Evaluating on VALIDATION set…")
print("="*65)
val_results = evaluate(model, val_loader, misfit_module)
val_df = build_results_df(val_results)
val_summary = summary_table(val_df)
print(f"\nOverall Val MSE:   {np.mean(val_results['mse_losses']):.4f}")
print(f"Overall Val MAE:   {val_df['abs_error'].mean():.4f}")
print(f"Overall Val MAPE:  {val_df['pct_error'].mean():.2f}%")
print(f"\nPer-concentration summary (validation):")
print(val_summary.to_string())


# ======================================================================
# 9. VISUALISATIONS
# ======================================================================

def plot_pred_vs_true(df, title="Predicted vs True Concentration"):
    """Scatter plot of predicted vs true concentration with identity line."""
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(df["true_conc"], df["pred_conc"], alpha=0.15, s=8,
               color="#4C72B0", edgecolors="none", rasterized=True)

    # Identity line
    lo, hi = CONC_RANGE[0] - 2, CONC_RANGE[-1] + 2
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal (y = x)")

    # Per-concentration mean ± std
    grouped = df.groupby("true_conc")["pred_conc"]
    means = grouped.mean()
    stds  = grouped.std()
    ax.errorbar(means.index, means.values, yerr=stds.values,
                fmt="ko", ms=5, capsize=3, lw=1.2, label="Mean ± σ")

    ax.set_xlabel("True Concentration")
    ax.set_ylabel("Predicted Concentration")
    ax.set_title(title)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
    return fig


def plot_error_boxplot(df, title="Absolute Error by Concentration"):
    """Box/violin plot of absolute errors per concentration."""
    fig, ax = plt.subplots(figsize=(14, 5))

    concs_sorted = sorted(df["true_conc"].unique())
    data = [df[df["true_conc"] == c]["abs_error"].values for c in concs_sorted]

    bplot = ax.boxplot(data, positions=range(len(concs_sorted)),
                       widths=0.6, patch_artist=True,
                       showfliers=False, medianprops=dict(color="black"))

    for patch in bplot["boxes"]:
        patch.set_facecolor("#7EB1D4")
        patch.set_alpha(0.7)

    ax.set_xticks(range(len(concs_sorted)))
    ax.set_xticklabels([str(int(c)) for c in concs_sorted], fontsize=9)
    ax.set_xlabel("True Concentration")
    ax.set_ylabel("Absolute Error")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()
    return fig


def plot_confusion_heatmap(df, title="Prediction Density Heatmap"):
    """2D histogram (heatmap) showing prediction density."""
    fig, ax = plt.subplots(figsize=(9, 7))

    # Round predicted to nearest integer for binning
    bins_x = np.arange(CONC_RANGE[0] - 1, CONC_RANGE[-1] + 3, 2)
    bins_y = np.arange(CONC_RANGE[0] - 1, CONC_RANGE[-1] + 3, 2)

    h, xedges, yedges, img = ax.hist2d(
        df["true_conc"], df["pred_conc"],
        bins=[bins_x, bins_y],
        cmap="YlOrRd",
        norm=LogNorm(),
    )
    fig.colorbar(img, ax=ax, label="Count (log)")

    # Identity line
    lo, hi = CONC_RANGE[0] - 2, CONC_RANGE[-1] + 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7)

    ax.set_xlabel("True Concentration")
    ax.set_ylabel("Predicted Concentration")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
    return fig


def plot_residual_histogram(df, title="Residual Distribution"):
    """Histogram of (predicted - true) residuals."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Overall residual histogram
    residuals = df["error"].values
    ax1.hist(residuals, bins=80, color="#5B8DB8", edgecolor="white",
             alpha=0.8, density=True)
    ax1.axvline(0, color="red", linestyle="--", lw=1.2)
    ax1.set_xlabel("Residual (predicted − true)")
    ax1.set_ylabel("Density")
    ax1.set_title(f"{title} — Overall")
    ax1.grid(True, alpha=0.3)

    # Stats text box
    stats_text = (f"Mean: {residuals.mean():.3f}\n"
                  f"Std:  {residuals.std():.3f}\n"
                  f"Skew: {pd.Series(residuals).skew():.3f}")
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
             verticalalignment="top", horizontalalignment="right",
             fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Residual vs true concentration (bias check)
    ax2.scatter(df["true_conc"], df["error"], alpha=0.1, s=4,
                color="#4C72B0", edgecolors="none", rasterized=True)
    mean_resid = df.groupby("true_conc")["error"].mean()
    ax2.plot(mean_resid.index, mean_resid.values, "ro-", ms=4, lw=1.2,
             label="Mean residual")
    ax2.axhline(0, color="black", linestyle="--", lw=0.8)
    ax2.set_xlabel("True Concentration")
    ax2.set_ylabel("Residual (predicted − true)")
    ax2.set_title(f"{title} — Bias Check")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
    return fig


def plot_error_metrics_summary(summary_df, title="Per-Concentration Error Metrics"):
    """Bar chart of MAE and RMSE per concentration."""
    fig, ax = plt.subplots(figsize=(14, 5))

    concs = summary_df.index.values
    x = np.arange(len(concs))
    width = 0.35

    bars1 = ax.bar(x - width/2, summary_df["mae"], width, label="MAE",
                   color="#5B8DB8", alpha=0.8, edgecolor="white")
    bars2 = ax.bar(x + width/2, summary_df["rmse"], width, label="RMSE",
                   color="#E8875A", alpha=0.8, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in concs], fontsize=9)
    ax.set_xlabel("True Concentration")
    ax.set_ylabel("Error Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()
    return fig


def plot_cumulative_error(df, title="Cumulative Error Distribution"):
    """CDF of absolute errors."""
    fig, ax = plt.subplots(figsize=(8, 5))

    sorted_errors = np.sort(df["abs_error"].values)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

    ax.plot(sorted_errors, cdf, color="#4C72B0", lw=2)

    # Mark key percentiles
    for pct in [0.5, 0.9, 0.95, 0.99]:
        threshold = np.percentile(df["abs_error"], pct * 100)
        ax.axhline(pct, color="gray", linestyle=":", lw=0.7, alpha=0.5)
        ax.axvline(threshold, color="gray", linestyle=":", lw=0.7, alpha=0.5)
        ax.annotate(f"{pct*100:.0f}%: {threshold:.2f}",
                    xy=(threshold, pct), fontsize=8,
                    xytext=(threshold + 0.5, pct - 0.03))

    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Cumulative Proportion")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
    return fig


def plot_mape_per_concentration(summary_df, title="Mean Absolute Percentage Error (MAPE)"):
    """MAPE per concentration — shows relative accuracy."""
    fig, ax = plt.subplots(figsize=(14, 5))

    concs = summary_df.index.values
    ax.bar(range(len(concs)), summary_df["mape"], color="#8FBC8F",
           alpha=0.8, edgecolor="white")
    ax.axhline(summary_df["mape"].mean(), color="red", linestyle="--",
               lw=1.2, label=f'Overall Mean: {summary_df["mape"].mean():.2f}%')

    ax.set_xticks(range(len(concs)))
    ax.set_xticklabels([str(int(c)) for c in concs], fontsize=9)
    ax.set_xlabel("True Concentration")
    ax.set_ylabel("MAPE (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()
    return fig


# ── Run all visualizations ──

print("\n" + "="*65)
print("VISUALISATIONS")
print("="*65)

fig1 = plot_pred_vs_true(val_df, "Validation — Predicted vs True Concentration")
fig2 = plot_error_boxplot(val_df, "Validation — Absolute Error by Concentration")
fig3 = plot_confusion_heatmap(val_df, "Validation — Prediction Density Heatmap")
fig4 = plot_residual_histogram(val_df, "Validation — Residuals")
fig5 = plot_error_metrics_summary(val_summary, "Validation — MAE & RMSE per Concentration")
fig6 = plot_cumulative_error(val_df, "Validation — Cumulative Error Distribution")
fig7 = plot_mape_per_concentration(val_summary, "Validation — MAPE per Concentration")


# ======================================================================
# 10. COMPREHENSIVE SUMMARY
# ======================================================================

def print_full_report(df, summary_df, split_name="Validation"):
    """Print a formatted final report."""
    print("\n" + "="*70)
    print(f"  PINN AGNR — {split_name} Report")
    print("="*70)

    print(f"\n  Total samples evaluated: {len(df)}")
    print(f"  Unique concentrations:   {df['true_conc'].nunique()}")

    print(f"\n  ── Global Metrics ──")
    print(f"  MAE:           {df['abs_error'].mean():.4f}")
    print(f"  RMSE:          {np.sqrt((df['abs_error']**2).mean()):.4f}")
    print(f"  MAPE:          {df['pct_error'].mean():.2f}%")
    print(f"  Median AE:     {df['abs_error'].median():.4f}")
    print(f"  Max AE:        {df['abs_error'].max():.4f}")
    print(f"  Mean Residual: {df['error'].mean():.4f} (bias)")
    print(f"  Std Residual:  {df['error'].std():.4f}")

    # Correlation coefficient
    from scipy.stats import pearsonr, spearmanr
    r, p_r = pearsonr(df["true_conc"], df["pred_conc"])
    rho, p_rho = spearmanr(df["true_conc"], df["pred_conc"])
    print(f"\n  Pearson  r  = {r:.6f}  (p = {p_r:.2e})")
    print(f"  Spearman ρ = {rho:.6f}  (p = {p_rho:.2e})")

    # R² score
    ss_res = ((df["pred_conc"] - df["true_conc"]) ** 2).sum()
    ss_tot = ((df["true_conc"] - df["true_conc"].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"  R² score   = {r2:.6f}")

    # Percentage within error thresholds
    for thresh in [1, 2, 3, 5, 10]:
        pct = (df["abs_error"] <= thresh).mean() * 100
        print(f"  Within ±{thresh:>2d}: {pct:6.2f}%")

    print(f"\n  ── Per-Concentration Table ──")
    print(summary_df.to_string())

    # Best and worst concentrations
    best_conc  = summary_df["mae"].idxmin()
    worst_conc = summary_df["mae"].idxmax()
    print(f"\n  Best  concentration (lowest MAE):  {int(best_conc)} "
          f"(MAE = {summary_df.loc[best_conc, 'mae']:.4f})")
    print(f"  Worst concentration (highest MAE): {int(worst_conc)} "
          f"(MAE = {summary_df.loc[worst_conc, 'mae']:.4f})")

    print("\n" + "="*70)


print_full_report(val_df, val_summary, "Validation")


# ======================================================================
# 11. OPTIONAL: GENERATE & EVALUATE TEST DATA (on-the-fly)
# ======================================================================
# Uncomment the block below if you have the leads data loaded and want
# to test on freshly generated (never-before-seen) configurations.
#
# from ca_agnr import device_transmission
#
# g_7 = np.load(str(LEADS_PATH))
#
# TEST_CONFIGS = np.arange(10001, 10101, 1)  # 100 unseen configs
# TEST_CONCS   = np.arange(1, 50, 2)
#
# print("\nGenerating test data (this is slow — one config at a time)…")
# data_test = {}
# for con in TEST_CONCS:
#     print(f"  Concentration {con}…")
#     data_test[con] = [
#         [device_transmission(w, 0.0001, 1, 0, 7, int(cfg), int(con))
#          for w in np.arange(0, 3, 0.01)]
#         for cfg in TEST_CONFIGS
#     ]
#
# test_ds = NormalizedTestDataset(data_test, pristine, SPECTRUM_LENGTH)
# test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
#
# print("\nEvaluating on held-out TEST set…")
# test_results = evaluate(model, test_loader, misfit_module)
# test_df = build_results_df(test_results)
# test_summary = summary_table(test_df)
#
# plot_pred_vs_true(test_df, "Test — Predicted vs True Concentration")
# plot_error_boxplot(test_df, "Test — Absolute Error by Concentration")
# plot_confusion_heatmap(test_df, "Test — Prediction Density Heatmap")
# plot_residual_histogram(test_df, "Test — Residuals")
# plot_error_metrics_summary(test_summary, "Test — MAE & RMSE per Concentration")
# plot_cumulative_error(test_df, "Test — Cumulative Error Distribution")
# plot_mape_per_concentration(test_summary, "Test — MAPE per Concentration")
# print_full_report(test_df, test_summary, "Test")


# ======================================================================
# 12. SAVE RESULTS (optional)
# ======================================================================

# val_df.to_csv(NOTEBOOK_DIR / "validation_results.csv", index=False)
# val_summary.to_csv(NOTEBOOK_DIR / "validation_summary.csv")
# print("Results saved to CSV.")

print("\n✓ Test & Validation suite completed!")
