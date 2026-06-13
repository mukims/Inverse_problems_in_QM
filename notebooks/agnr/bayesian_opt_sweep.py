"""
bayesian_opt_sweep.py — Bayesian Optimisation of PINN Hyperparameters
=====================================================================

Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler to efficiently
search the hyperparameter space of the ConductanceMLP PINN with curvature-
weighted misfit regulariser.

Search space (7 dimensions):
  - lr:             [1e-5, 1e-2]    (log-uniform)
  - misfit_weight:  [1e-4, 10.0]    (log-uniform)
  - temperature:    [0.1, 10.0]     (log-uniform)
  - dropout:        [0.0, 0.5]
  - noise_std:      [0.0, 0.10]
  - weight_decay:   [1e-6, 1e-2]    (log-uniform)
  - hidden_arch:    3 candidate MLP widths

Objective: Validation MAE (impurity count) — evaluated directly, not via
the composite training loss. This gives a metric comparable across λ values.

Features:
  - Optuna MedianPruner kills bad trials early (saves ~40-60% compute)
  - Fixed train/val split (seed=42) across all trials for fair comparison
  - After optimisation, retrains the best config and evaluates on 2100 test set
  - Produces 4 publication-quality plots + results summary

Usage:
    conda run -n ml python bayesian_opt_sweep.py [--n-trials 50] [--epochs 200]

Dependencies:
    pip install optuna
"""

import os
import sys
import copy
import json
import glob
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# ── Import from the existing PINN module ────────────────────────────────────
from pinn_agnr_curvature import (
    ConductanceMLP,
    CurvatureMisfit,
    NormalizedTransmissionsDataset,
)


# ======================================================================
# 1. CONFIGURATION
# ======================================================================

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR     = PROJECT_ROOT / "data" / "raw" / "transmission_results"
TEST_DIR     = PROJECT_ROOT / "data" / "test" / "transmission_results"
MANIFEST     = SCRIPT_DIR / "manifest_agnr.csv"
PRISTINE_PATH = DATA_DIR / "pristine.npy"
OUTPUT_DIR   = SCRIPT_DIR / "bo_sweep_results"

SPECTRUM_LEN  = 200
CONC_RANGE    = np.arange(1, 50, 2)  # matches main training script
BATCH_SIZE    = 64
VAL_SPLIT     = 0.2
SPLIT_SEED    = 42   # fixed split across all trials

# Architecture candidates
HIDDEN_ARCHITECTURES = {
    "wide":     [512, 256, 128, 64],
    "standard": [256, 128, 64, 32],
    "narrow":   [128, 64, 32, 16],
}


# ======================================================================
# 2. DATA & MISFIT MODULE (loaded once, shared across trials)
# ======================================================================

def load_shared_resources(device):
    """Load dataset, pristine, and misfit module once."""
    pristine = np.load(str(PRISTINE_PATH))
    dataset = NormalizedTransmissionsDataset(
        str(MANIFEST), str(DATA_DIR), pristine, SPECTRUM_LEN
    )
    print(f"Dataset: {len(dataset)} samples")

    # Build CurvatureMisfit from raw files
    pristine_clip = pristine[:SPECTRUM_LEN].astype(np.float32)
    ref_list = []
    for con in CONC_RANGE:
        acc = []
        for cfg in range(100):
            fpath = DATA_DIR / f"7_agnr_conc{int(con)}_cfg{cfg}.npy"
            if fpath.exists():
                spec = np.load(str(fpath)).astype(np.float32)[:SPECTRUM_LEN]
                spec = np.clip(spec, 0, pristine_clip)
                acc.append(spec)
        if len(acc) == 0:
            raise FileNotFoundError(f"No files for conc {con}")
        ref_list.append(np.mean(acc, axis=0) / (pristine_clip + 1e-8))

    ref_spectra_np = np.array(ref_list)
    ref_spectra = torch.tensor(ref_spectra_np, dtype=torch.float32)
    concentrations = torch.tensor(CONC_RANGE, dtype=torch.float32)
    pristine_tensor = torch.tensor(pristine_clip, dtype=torch.float32)

    # Fixed train/val split
    n_val = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    return {
        "dataset": dataset,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "ref_spectra": ref_spectra,
        "concentrations": concentrations,
        "pristine_tensor": pristine_tensor,
        "pristine_np": pristine_clip,
        "ref_spectra_np": ref_spectra_np,
    }


# ======================================================================
# 3. OBJECTIVE FUNCTION (one Optuna trial)
# ======================================================================

def objective(trial, resources, device, max_epochs, patience):
    """
    Single Optuna trial: train a ConductanceMLP with sampled hyperparameters,
    return validation MAE.
    """
    # ── Sample hyperparameters ──
    lr            = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    misfit_weight = trial.suggest_float("misfit_weight", 1e-4, 10.0, log=True)
    temperature   = trial.suggest_float("temperature", 0.1, 10.0, log=True)
    dropout       = trial.suggest_float("dropout", 0.0, 0.5)
    noise_std     = trial.suggest_float("noise_std", 0.0, 0.10)
    weight_decay  = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    arch_name     = trial.suggest_categorical("hidden_arch",
                                              list(HIDDEN_ARCHITECTURES.keys()))
    hidden_dims = HIDDEN_ARCHITECTURES[arch_name]

    # ── Build model + misfit module ──
    model = ConductanceMLP(
        input_length=SPECTRUM_LEN,
        hidden_dims=hidden_dims,
        dropout=dropout,
        noise_std=noise_std,
    ).to(device)

    misfit_module = CurvatureMisfit(
        ref_spectra=resources["ref_spectra"],
        concentrations=resources["concentrations"],
        pristine=resources["pristine_tensor"],
        temperature=temperature,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    train_loader = DataLoader(resources["train_ds"], batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(resources["val_ds"], batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    best_val_mae = float("inf")
    best_weights = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # ── Train ──
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            yb_flat = yb.squeeze(-1)
            preds_flat = preds.squeeze(-1)

            mse_loss = F.mse_loss(preds_flat, yb_flat)
            misfit_loss = misfit_module(xb, preds)
            loss = mse_loss + misfit_weight * misfit_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # ── Validate (MAE, not composite loss) ──
        model.eval()
        all_errors = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).squeeze(-1)
                yb_flat = yb.squeeze(-1)
                errors = torch.abs(preds - yb_flat)
                all_errors.append(errors)

        val_mae = torch.cat(all_errors).mean().item()

        # Report to Optuna for pruning
        trial.report(val_mae, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    # Store best weights for later retrieval
    trial.set_user_attr("best_epoch", epoch - epochs_no_improve)
    trial.set_user_attr("best_val_mae", best_val_mae)

    # Save the best model weights to disk so we can reload the winner
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_path = OUTPUT_DIR / f"trial_{trial.number}.pt"
    if best_weights is not None:
        torch.save({
            "model_state_dict": best_weights,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "noise_std": noise_std,
        }, ckpt_path)

    return best_val_mae


# ======================================================================
# 4. TEST SET EVALUATION
# ======================================================================

def evaluate_on_test_set(model, device, pristine_np, ref_spectra_np):
    """Evaluate a trained model on the 2100 held-out test spectra."""
    test_files = sorted(glob.glob(str(TEST_DIR / "*.npy")))
    if not test_files:
        print("⚠ No test files found. Skipping test evaluation.")
        return None, None, None

    true_concs, model_preds, misfit_preds = [], [], []
    conc_range = CONC_RANGE

    model.eval()
    for fpath in test_files:
        filename = os.path.basename(fpath)
        true_c = float(filename.split("_conc")[1].split("_")[0])
        true_concs.append(true_c)

        spec = np.load(fpath).astype(np.float32)[:SPECTRUM_LEN]
        spec = np.clip(spec, 0, pristine_np)
        spec_norm = spec / (pristine_np + 1e-8)

        x = torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            model_preds.append(model(x).item())

        # Physical misfit baseline for comparison
        test_crop = spec_norm[20:150]
        ref_crop = ref_spectra_np[:, 20:150]
        pristine_crop = pristine_np[20:150]
        t_un = np.clip(test_crop * pristine_crop, 0.0, pristine_crop)
        r_un = ref_crop * pristine_crop
        mis = np.sum((t_un[None, :] - r_un) ** 2, axis=1) / 150.0
        misfit_preds.append(conc_range[np.argmin(mis)])

    true_concs = np.array(true_concs)
    model_preds = np.array(model_preds)
    misfit_preds = np.array(misfit_preds)

    return true_concs, model_preds, misfit_preds


# ======================================================================
# 5. PLOTTING
# ======================================================================

def plot_results(study, true_concs, model_preds, misfit_preds):
    """Generate 4 publication-quality plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

    # ── 5a. Optimization history ──
    fig, ax = plt.subplots(figsize=(10, 5))
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trial_nums = [t.number for t in trials]
    trial_vals = [t.value for t in trials]
    best_so_far = np.minimum.accumulate(trial_vals)

    ax.scatter(trial_nums, trial_vals, alpha=0.6, s=30, color="#5C6BC0",
               edgecolors="white", linewidths=0.5, label="Trial MAE", zorder=3)
    ax.plot(trial_nums, best_so_far, color="#E53935", lw=2.0,
            label="Best MAE so far", zorder=4)
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Validation MAE (impurities)")
    ax.set_title("Bayesian Optimisation: Convergence History")
    ax.legend()
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bo_optimisation_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 5b. Parameter importance ──
    try:
        importance = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(10, 5))
        params = list(importance.keys())
        values = list(importance.values())
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(params)))
        bars = ax.barh(params[::-1], values[::-1], color=colors[::-1], edgecolor="white")
        for bar, val in zip(bars, values[::-1]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=10)
        ax.set_xlabel("Importance (fANOVA)")
        ax.set_title("Hyperparameter Importance")
        ax.set_xlim(0, max(values) * 1.15)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "bo_param_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"⚠ Could not compute parameter importance: {e}")

    # ── 5c. Parallel coordinates (top 30% of trials) ──
    try:
        fig = optuna.visualization.matplotlib.plot_parallel_coordinate(
            study,
            params=["lr", "misfit_weight", "temperature", "dropout",
                     "noise_std", "weight_decay"],
        )
        plt.title("Parallel Coordinates (all complete trials)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "bo_parallel_coordinates.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"⚠ Could not plot parallel coordinates: {e}")

    # ── 5d. Test set scatter (BO-tuned vs Physical Misfit) ──
    if true_concs is not None:
        model_mae = np.mean(np.abs(model_preds - true_concs))
        model_rmse = np.sqrt(np.mean((model_preds - true_concs) ** 2))
        mis_mae = np.mean(np.abs(misfit_preds - true_concs))
        mis_rmse = np.sqrt(np.mean((misfit_preds - true_concs) ** 2))

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        lo, hi = true_concs.min(), true_concs.max()

        for ax, (name, preds, mae, rmse, color) in zip(axes, [
            ("BO-Tuned MLP (PINN)", model_preds, model_mae, model_rmse, "#1565C0"),
            ("Physical Misfit Baseline", misfit_preds, mis_mae, mis_rmse, "#2E7D32"),
        ]):
            ax.scatter(true_concs, preds, alpha=0.45, s=18, color=color, edgecolors="none")
            ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect")
            ax.set_title(f"{name}\nMAE: {mae:.2f}  |  RMSE: {rmse:.2f}", fontsize=12)
            ax.set_xlabel("True Concentration")
            ax.set_ylabel("Predicted Concentration")
            ax.set_xlim(lo - 2, hi + 2)
            ax.set_ylim(lo - 5, hi + 5)
            ax.grid(True, alpha=0.2)
            ax.legend(loc="upper left")

        plt.suptitle("Test Set: BO-Tuned PINN vs Physical Misfit (2100 spectra)",
                      fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "bo_test_scatter.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ======================================================================
# 6. MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Bayesian Optimisation for PINN hyperparameters")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of BO trials (default: 50)")
    parser.add_argument("--epochs",   type=int, default=200,
                        help="Max epochs per trial (default: 200)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Optuna sampler seed (default: 42)")
    parser.add_argument("--device",   type=str, default=None)
    args = parser.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n{'='*70}")
    print(f"  Bayesian Optimisation — PINN Hyperparameter Sweep")
    print(f"  Device: {device}  |  Trials: {args.n_trials}  |  Max epochs: {args.epochs}")
    print(f"{'='*70}\n")

    # ── Load data once ──
    print("Loading shared resources...")
    resources = load_shared_resources(device)
    print(f"  Train: {len(resources['train_ds'])}  |  Val: {len(resources['val_ds'])}")

    # ── Create Optuna study ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    db_path = OUTPUT_DIR / "optuna_study.db"
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name="pinn_bo_sweep",
        direction="minimize",
        sampler=TPESampler(seed=args.seed, n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=30),
        storage=storage,
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    n_remaining = max(0, args.n_trials - n_existing)
    if n_existing > 0:
        print(f"  Resuming study: {n_existing} trials already completed, "
              f"running {n_remaining} more.")

    # ── Run optimisation ──
    if n_remaining > 0:
        study.optimize(
            lambda trial: objective(trial, resources, device, args.epochs, args.patience),
            n_trials=n_remaining,
            show_progress_bar=True,
        )

    # ── Print results ──
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print(f"\n{'='*70}")
    print(f"  OPTIMISATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Completed trials: {len(completed)}")
    print(f"  Pruned trials:    {len(pruned)}")
    print(f"  Total trials:     {len(study.trials)}")
    print(f"\n  Best trial #{study.best_trial.number}:")
    print(f"    Validation MAE: {study.best_trial.value:.4f}")
    print(f"    Parameters:")
    for k, v in study.best_trial.params.items():
        if isinstance(v, float):
            print(f"      {k:>20s}: {v:.6g}")
        else:
            print(f"      {k:>20s}: {v}")

    # ── Save summary JSON ──
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_completed": len(completed),
        "n_pruned": len(pruned),
        "best_trial_number": study.best_trial.number,
        "best_val_mae": study.best_trial.value,
        "best_params": study.best_trial.params,
        "top_5_trials": [
            {"number": t.number, "val_mae": t.value, "params": t.params}
            for t in sorted(completed, key=lambda t: t.value)[:5]
        ],
    }
    summary_path = OUTPUT_DIR / "bo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")

    # ── Reload best model and evaluate on test set ──
    print(f"\n{'='*70}")
    print(f"  EVALUATING BEST MODEL ON HELD-OUT TEST SET")
    print(f"{'='*70}")

    best_ckpt_path = OUTPUT_DIR / f"trial_{study.best_trial.number}.pt"
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        best_model = ConductanceMLP(
            input_length=SPECTRUM_LEN,
            hidden_dims=ckpt["hidden_dims"],
            dropout=ckpt["dropout"],
            noise_std=ckpt["noise_std"],
        ).to(device)
        best_model.load_state_dict(ckpt["model_state_dict"])
        best_model.eval()

        true_concs, model_preds, misfit_preds = evaluate_on_test_set(
            best_model, device, resources["pristine_np"], resources["ref_spectra_np"]
        )

        if true_concs is not None:
            bo_mae = np.mean(np.abs(model_preds - true_concs))
            bo_rmse = np.sqrt(np.mean((model_preds - true_concs) ** 2))
            mis_mae = np.mean(np.abs(misfit_preds - true_concs))
            mis_rmse = np.sqrt(np.mean((misfit_preds - true_concs) ** 2))

            print(f"\n  {'Method':<30s} {'MAE':>8s} {'RMSE':>8s}")
            print(f"  {'-'*30} {'-'*8} {'-'*8}")
            print(f"  {'BO-Tuned PINN':<30s} {bo_mae:>8.3f} {bo_rmse:>8.3f}")
            print(f"  {'Physical Misfit':<30s} {mis_mae:>8.3f} {mis_rmse:>8.3f}")
            improvement = (1 - bo_mae / mis_mae) * 100
            print(f"\n  BO-tuned PINN achieves {improvement:.1f}% lower MAE vs misfit baseline")

            # Update summary with test results
            summary["test_results"] = {
                "bo_mae": float(bo_mae),
                "bo_rmse": float(bo_rmse),
                "misfit_mae": float(mis_mae),
                "misfit_rmse": float(mis_rmse),
            }
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
    else:
        true_concs, model_preds, misfit_preds = None, None, None
        print("  ⚠ Best trial checkpoint not found. Skipping test evaluation.")

    # ── Generate plots ──
    print(f"\nGenerating plots...")
    plot_results(study, true_concs, model_preds, misfit_preds)
    print(f"✓ Plots saved to: {OUTPUT_DIR}/")

    # ── Save best model as the main checkpoint ──
    if best_ckpt_path.exists():
        best_save_path = OUTPUT_DIR / "pinn_bo_best.pt"
        ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=True)
        torch.save({
            "model_state_dict": ckpt["model_state_dict"],
            "hidden_dims": ckpt["hidden_dims"],
            "dropout": ckpt["dropout"],
            "noise_std": ckpt["noise_std"],
            "best_params": study.best_trial.params,
            "best_val_mae": study.best_trial.value,
        }, best_save_path)
        print(f"✓ Best model saved to: {best_save_path}")

    print(f"\n{'='*70}")
    print(f"  DONE — {len(completed)} trials completed, {len(pruned)} pruned.")
    print(f"  Best validation MAE: {study.best_trial.value:.4f}")
    print(f"  Study DB: {db_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
