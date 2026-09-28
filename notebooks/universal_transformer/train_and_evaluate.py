#!/usr/bin/env python
"""
train_and_evaluate.py — Training, Evaluation & Benchmarking for Universal Transformer

Runs end-to-end training and evaluation on combined 7-AGNR, 9-AGNR, and Square-10 data.
Produces:
  - Checkpoint: universal_transformer.pt
  - Metrics JSON: universal_metrics.json
  - Visualizations: universal_scatter.png, universal_confusion.png, universal_training_curves.png
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Ensure modules are on path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import universal_data as ud
from universal_transformer import UniversalPatchedTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Universal Multi-Task Patched Transformer.")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum epochs (training stops earlier on convergence)")
    parser.add_argument("--patience", type=int, default=12, help="Stop after this many epochs without a val-MAE improvement")
    parser.add_argument("--plateau-patience", type=int, default=4, help="Halve the LR after this many epochs without improvement")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Warmup epochs")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--samples-per-conc", type=int, default=3000, help="Config seeds per concentration per system")
    parser.add_argument("--threads", type=int, default=16, help="Torch CPU threads")
    parser.add_argument("--out-dir", type=str, default=str(SCRIPT_DIR), help="where checkpoint, metrics and plots are written")
    parser.add_argument("--cache-path", type=str, default=None, help="default: universal_cache_seed_<samples-per-conc>.pt")
    args = parser.parse_args()
    if args.cache_path is None:
        args.cache_path = str(SCRIPT_DIR / f"universal_cache_seed_{args.samples_per_conc}.pt")
    return args


def compute_metrics(preds_c, true_c):
    err = np.asarray(preds_c, float) - np.asarray(true_c, float)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "Max_Error": float(np.max(np.abs(err))),
    }


def main():
    args = parse_args()
    global OUT_DIR
    OUT_DIR = Path(args.out_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    print(f"Setting PyTorch threads: {args.threads}")

    repo_root = SCRIPT_DIR.parents[1]

    # 1. Load Dataset
    t0_data = time.time()
    train_data, val_data, test_data, scaler, meta = ud.load_universal_data(
        repo_root=repo_root,
        samples_per_conc=args.samples_per_conc,
        spectrum_len=150,
        seed=42,
        cache_path=args.cache_path,
        num_workers=args.threads,
    )
    print(f"Data ready in {time.time() - t0_data:.2f}s.")

    # Target normalization
    y_conc_tr_norm = torch.tensor(scaler.transform(train_data["y_conc"].numpy()), dtype=torch.float32).view(-1, 1)
    y_conc_va_norm = torch.tensor(scaler.transform(val_data["y_conc"].numpy()), dtype=torch.float32).view(-1, 1)
    y_conc_te_norm = torch.tensor(scaler.transform(test_data["y_conc"].numpy()), dtype=torch.float32).view(-1, 1)

    # PyTorch DataLoaders
    train_ds = TensorDataset(train_data["x"], train_data["y_type"], train_data["y_width"], y_conc_tr_norm, train_data["y_conc"])
    val_ds = TensorDataset(val_data["x"], val_data["y_type"], val_data["y_width"], y_conc_va_norm, val_data["y_conc"])
    test_ds = TensorDataset(test_data["x"], test_data["y_type"], test_data["y_width"], y_conc_te_norm, test_data["y_conc"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Build Model
    model = UniversalPatchedTransformer(
        seq_len=150,
        patch_size=10,
        stem_channels=32,
        embed_dim=128,
        depth=3,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.05,
        drop_path_rate=0.05,
        pos_embed_std=0.10,
        condition_heads=True,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel initialized: UniversalPatchedTransformer | Trainable Parameters: {total_params:,}")

    # 3. Optimizer & Schedulers
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Linear warmup, then halve the LR whenever val MAE plateaus; stop once it stops improving.
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: min(1.0, float(e + 1) / args.warmup_epochs))
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                         patience=args.plateau_patience, min_lr=1e-6)
    huber_loss_fn = nn.SmoothL1Loss(beta=0.05)
    ce_loss_fn = nn.CrossEntropyLoss()

    # 4. Training Loop
    history = {
        "train_loss": [], "val_loss": [], "lr": [],
        "val_type_acc": [], "val_width_acc": [], "val_conc_mae": [],
    }

    best_val_mae = float("inf")
    best_state = None
    epochs_since_best = 0

    print("\n" + "=" * 80)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<10} {'Type Acc':<10} {'Width Acc':<10} {'Conc MAE':<10} {'LR':<10} {'Time':<8}")
    print("-" * 80)

    t_train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep_start = time.time()
        model.train()
        total_loss = 0.0

        for bx, b_type, b_width, b_conc_norm, _ in train_loader:
            optimizer.zero_grad()
            out_type, out_width, out_conc = model(bx)

            l_type = ce_loss_fn(out_type, b_type)
            l_width = ce_loss_fn(out_width, b_width)
            l_conc = huber_loss_fn(out_conc, b_conc_norm)

            loss = l_type + l_width + l_conc
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(bx)

        train_loss = total_loss / len(train_ds)
        current_lr = optimizer.param_groups[0]["lr"]

        # Validation
        model.eval()
        v_total_loss = 0.0
        v_type_corr = 0
        v_width_corr = 0
        v_preds_c = []
        v_true_c = []

        with torch.no_grad():
            for bx, b_type, b_width, b_conc_norm, b_conc_raw in val_loader:
                out_type, out_width, out_conc = model(bx)

                l_type = ce_loss_fn(out_type, b_type)
                l_width = ce_loss_fn(out_width, b_width)
                l_conc = huber_loss_fn(out_conc, b_conc_norm)
                v_loss = l_type + l_width + l_conc
                v_total_loss += v_loss.item() * len(bx)

                v_type_corr += (out_type.argmax(dim=1) == b_type).sum().item()
                v_width_corr += (out_width.argmax(dim=1) == b_width).sum().item()

                pred_c_unnorm = scaler.inverse_transform(out_conc.squeeze(1).numpy())
                v_preds_c.append(pred_c_unnorm)
                v_true_c.append(b_conc_raw.numpy())

        val_loss = v_total_loss / len(val_ds)
        val_type_acc = 100.0 * v_type_corr / len(val_ds)
        val_width_acc = 100.0 * v_width_corr / len(val_ds)

        v_preds_arr = np.concatenate(v_preds_c)
        v_true_arr = np.concatenate(v_true_c)
        val_conc_mae = np.mean(np.abs(v_preds_arr - v_true_arr))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        history["val_type_acc"].append(val_type_acc)
        history["val_width_acc"].append(val_width_acc)
        history["val_conc_mae"].append(val_conc_mae)

        ep_time = time.time() - t_ep_start
        if epoch < args.warmup_epochs:
            warmup.step()
        else:
            plateau.step(val_conc_mae)
        is_best = val_conc_mae < best_val_mae - 1e-4
        if is_best:
            best_val_mae = val_conc_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        mark = " *" if is_best else ""
        print(f"{epoch:02d}/{args.epochs:02d}    {train_loss:<12.4f} {val_loss:<10.4f} {val_type_acc:>6.2f}%    {val_width_acc:>6.2f}%    {val_conc_mae:>6.3f}    {current_lr:<10.2e} {ep_time:>5.1f}s{mark}", flush=True)
        if epochs_since_best >= args.patience:
            print(f"Converged: no val-MAE improvement for {args.patience} epochs (stopped at epoch {epoch}).")
            break

    total_train_time = time.time() - t_train_start
    print("=" * 80)
    print(f"Training completed in {total_train_time/60:.2f} min. Best Val Conc MAE: {best_val_mae:.3f}")

    # Save Checkpoint
    ckpt_path = OUT_DIR / "universal_transformer.pt"
    torch.save({
        "model_state_dict": best_state,
        "scaler": scaler.as_dict(),
        "meta": meta,
        "args": vars(args),
        "best_val_mae": best_val_mae,
    }, ckpt_path)
    print(f"✓ Saved best model checkpoint to {ckpt_path}")

    # 5. Full Evaluation on Held-Out Test Set
    print("\n" + "=" * 80)
    print(f"EVALUATING ON HELD-OUT TEST SET ({len(test_ds):,} spectra, unseen config seeds)")
    print("=" * 80)

    model.load_state_dict(best_state)
    model.eval()

    test_type_preds, test_type_true = [], []
    test_width_preds, test_width_true = [], []
    test_conc_preds, test_conc_true = [], []

    with torch.no_grad():
        for bx, b_type, b_width, _, b_conc_raw in test_loader:
            out_type, out_width, out_conc = model(bx)
            test_type_preds.append(out_type.argmax(dim=1).numpy())
            test_type_true.append(b_type.numpy())
            test_width_preds.append(out_width.argmax(dim=1).numpy())
            test_width_true.append(b_width.numpy())
            pred_unnorm = scaler.inverse_transform(out_conc.squeeze(1).numpy())
            test_conc_preds.append(pred_unnorm)
            test_conc_true.append(b_conc_raw.numpy())

    y_type_p = np.concatenate(test_type_preds)
    y_type_t = np.concatenate(test_type_true)
    y_width_p = np.concatenate(test_width_preds)
    y_width_t = np.concatenate(test_width_true)
    y_conc_p = np.concatenate(test_conc_preds)
    y_conc_t = np.concatenate(test_conc_true)

    # Overall metrics
    type_acc = float(100.0 * np.mean(y_type_p == y_type_t))
    width_acc = float(100.0 * np.mean(y_width_p == y_width_t))
    overall_conc_metrics = compute_metrics(y_conc_p, y_conc_t)

    # Per-system metrics
    mask_7 = (y_width_t == 0)
    mask_9 = (y_width_t == 1)
    mask_sq = (y_width_t == 2)

    metrics_7 = compute_metrics(y_conc_p[mask_7], y_conc_t[mask_7])
    metrics_7["Width_Accuracy"] = float(100.0 * np.mean(y_width_p[mask_7] == 0))
    metrics_7["Type_Accuracy"] = float(100.0 * np.mean(y_type_p[mask_7] == 0))

    metrics_9 = compute_metrics(y_conc_p[mask_9], y_conc_t[mask_9])
    metrics_9["Width_Accuracy"] = float(100.0 * np.mean(y_width_p[mask_9] == 1))
    metrics_9["Type_Accuracy"] = float(100.0 * np.mean(y_type_p[mask_9] == 0))

    metrics_sq = compute_metrics(y_conc_p[mask_sq], y_conc_t[mask_sq])
    metrics_sq["Width_Accuracy"] = float(100.0 * np.mean(y_width_p[mask_sq] == 2))
    metrics_sq["Type_Accuracy"] = float(100.0 * np.mean(y_type_p[mask_sq] == 1))

    full_results = {
        "Overall": {
            "Type_Accuracy": type_acc,
            "Width_Accuracy": width_acc,
            "Conc_MAE": overall_conc_metrics["MAE"],
            "Conc_RMSE": overall_conc_metrics["RMSE"],
            "Conc_Max_Error": overall_conc_metrics["Max_Error"],
            "Test_Samples": len(y_conc_t),
        },
        "7-AGNR": metrics_7,
        "9-AGNR": metrics_9,
        "Square-10": metrics_sq,
        "Training_Time_Sec": round(total_train_time, 1),
        "Epochs_Run": len(history["val_conc_mae"]),
        "Best_Val_Conc_MAE": float(best_val_mae),
        "Split": "config-seed 70/15/15",
    }

    # Print Summary Table
    print(f"{'System Target':<18} {'Type Acc (%)':>12} {'Width Acc (%)':>14} {'Conc MAE':>10} {'Conc RMSE':>10} {'Max Error':>10}")
    print("-" * 78)
    print(f"{'7-AGNR':<18} {metrics_7['Type_Accuracy']:>12.2f} {metrics_7['Width_Accuracy']:>14.2f} {metrics_7['MAE']:>10.3f} {metrics_7['RMSE']:>10.3f} {metrics_7['Max_Error']:>10.3f}")
    print(f"{'9-AGNR':<18} {metrics_9['Type_Accuracy']:>12.2f} {metrics_9['Width_Accuracy']:>14.2f} {metrics_9['MAE']:>10.3f} {metrics_9['RMSE']:>10.3f} {metrics_9['Max_Error']:>10.3f}")
    print(f"{'Square-10':<18} {metrics_sq['Type_Accuracy']:>12.2f} {metrics_sq['Width_Accuracy']:>14.2f} {metrics_sq['MAE']:>10.3f} {metrics_sq['RMSE']:>10.3f} {metrics_sq['Max_Error']:>10.3f}")
    print("-" * 78)
    print(f"{'OVERALL COMBINED':<18} {type_acc:>12.2f} {width_acc:>14.2f} {overall_conc_metrics['MAE']:>10.3f} {overall_conc_metrics['RMSE']:>10.3f} {overall_conc_metrics['Max_Error']:>10.3f}")
    print("=" * 78)

    # Save Metrics JSON
    metrics_file = OUT_DIR / "universal_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"✓ Saved numerical benchmark metrics to {metrics_file}")

    # 6. Generate Diagnostic Plots
    print("\nGenerating benchmark plots...")

    # --- Plot 1: 3-Panel Scatter Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    systems = [
        ("7-AGNR (Width 7)", mask_7, metrics_7, "#2196F3"),
        ("9-AGNR (Width 9)", mask_9, metrics_9, "#FF9800"),
        ("Square Lattice (Size 10)", mask_sq, metrics_sq, "#4CAF50"),
    ]

    for ax, (title, mask, m, col) in zip(axes, systems):
        tc = y_conc_t[mask]
        pc = y_conc_p[mask]
        ax.scatter(tc, pc, alpha=0.35, s=16, color=col, edgecolors="none")
        lo = min(tc.min(), pc.min())
        hi = max(tc.max(), pc.max())
        ax.plot([0, 100], [0, 100], "r--", lw=1.5, label="Perfect")
        ax.set_title(f"{title}\nMAE: {m['MAE']:.3f} | RMSE: {m['RMSE']:.3f} | Acc: {m['Width_Accuracy']:.1f}%", fontsize=11)
        ax.set_xlabel("True Impurity Concentration (c)")
        ax.set_ylabel("Predicted Concentration (ĉ)")
        ax.set_xlim(0, 102)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")

    plt.suptitle("Universal Multi-Task Transformer: Concentration Prediction Across 3 Quantum Systems", fontsize=13, y=1.02)
    plt.tight_layout()
    scatter_path = OUT_DIR / "universal_scatter.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved scatter plot to {scatter_path}")

    # --- Plot 2: Training Curves ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    axes[0].plot(history["train_loss"], label="Train Loss", color="#1976D2")
    axes[0].plot(history["val_loss"], label="Val Loss", color="#D32F2F", linestyle="--")
    axes[0].set_title("Multi-Task Loss Convergence")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(history["val_type_acc"], label="Type Accuracy", color="#388E3C")
    axes[1].plot(history["val_width_acc"], label="Width Accuracy", color="#7B1FA2", linestyle="--")
    axes[1].set_title("Validation Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(95, 100.5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(history["val_conc_mae"], label="Val Conc MAE", color="#F57C00")
    axes[2].set_title("Validation Concentration Error")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MAE (impurities)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    plt.tight_layout()
    training_path = OUT_DIR / "universal_training_curves.png"
    plt.savefig(training_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved training curves to {training_path}")

    # --- Plot 3: Confusion Matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Type Confusion Matrix
    type_cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_type_t, y_type_p):
        type_cm[t, p] += 1
    im0 = axes[0].imshow(type_cm, cmap="Blues")
    axes[0].set_title(f"System Type Confusion (Acc: {type_acc:.2f}%)")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["AGNR", "Square Lattice"])
    axes[0].set_yticklabels(["AGNR", "Square Lattice"])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{type_cm[i, j]:,}", ha="center", va="center", color="white" if type_cm[i, j] > type_cm.max()/2 else "black")

    # Width Confusion Matrix
    width_cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_width_t, y_width_p):
        width_cm[t, p] += 1
    im1 = axes[1].imshow(width_cm, cmap="Purples")
    axes[1].set_title(f"System Width/Size Confusion (Acc: {width_acc:.2f}%)")
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_xticklabels(["7-AGNR", "9-AGNR", "Square-10"])
    axes[1].set_yticklabels(["7-AGNR", "9-AGNR", "Square-10"])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{width_cm[i, j]:,}", ha="center", va="center", color="white" if width_cm[i, j] > width_cm.max()/2 else "black")

    plt.tight_layout()
    confusion_path = OUT_DIR / "universal_confusion.png"
    plt.savefig(confusion_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved confusion matrices to {confusion_path}")

    print("\n[ALL DONE] Universal Transformer pipeline finished successfully!")


if __name__ == "__main__":
    main()
