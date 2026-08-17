#!/usr/bin/env python
"""
train_multi_width.py — Multi-Width Joint Model Training for 7-AGNR & 9-AGNR.

Simultaneously learns:
  1. Width Identification: 7-AGNR (width=7) vs 9-AGNR (width=9)
  2. Impurity Concentration Prediction: continuous scalar c in [2, 98]

Uses consolidated data from:
  /run/media/shardul/storage/machine_learning/transmission_data/transmission_results/consolidated_data/
Following xgb.ipynb data cleaning and normalization standards.
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
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import xgboost as xgb

# Multi-threaded CPU acceleration & unbuffered live logging
torch.set_num_threads(min(20, max(1, os.cpu_count() - 2)))
sys.stdout.reconfigure(line_buffering=True)


# ======================================================================
# 1. MULTI-TASK NEURAL NETWORK ARCHITECTURES
# ======================================================================

class MultiTaskConductanceMLP(nn.Module):
    """
    Multi-Task MLP for joint width classification (7 vs 9) and
    impurity concentration regression.
    """
    def __init__(self, input_length: int = 150, hidden_dims: list = None,
                 dropout: float = 0.2, noise_std: float = 0.01):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        self.noise_std = noise_std

        # Shared feature extractor backbone
        layers = []
        in_dim = input_length
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        # Head 1: Width classification (2 classes: 7-AGNR vs 9-AGNR)
        self.width_head = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 2)
        )

        # Head 2: Concentration regression (scalar c)
        self.conc_head = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor):
        if self.training and self.noise_std > 0:
            noise = 1.0 + torch.randn_like(x) * self.noise_std
            x = x * noise
        feat = self.backbone(x)
        w_logits = self.width_head(feat)
        c_pred = self.conc_head(feat)
        return w_logits, c_pred


class DropPath(nn.Module):
    """Stochastic Depth per-sample."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device))
        return x * mask / keep


class ConvStem(nn.Module):
    """1D ConvStem for rich spectral feature representation."""
    def __init__(self, out_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3),
            nn.GELU(),
            nn.Conv1d(16, out_channels, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchEmbedding1D(nn.Module):
    """1D non-overlapping patching with [CLS] token and positional encoding."""
    def __init__(self, seq_len: int = 150, patch_size: int = 10,
                 in_channels: int = 32, embed_dim: int = 128):
        super().__init__()
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.proj(x).transpose(1, 2)  # [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.norm(x) + self.pos_embed
        return x


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer block with multi-head attention and DropPath."""
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class MultiTaskPatchedTransformerV2(nn.Module):
    """
    Multi-Task 1D-Patched Transformer v2 with ConvStem, self-attention blocks,
    and dual heads for width classification and concentration regression.
    """
    def __init__(self, seq_len: int = 150, patch_size: int = 10,
                 stem_channels: int = 32, embed_dim: int = 128, depth: int = 4,
                 num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1,
                 drop_path_rate: float = 0.1):
        super().__init__()
        self.stem = ConvStem(out_channels=stem_channels)
        self.patch_embed = PatchEmbedding1D(
            seq_len=seq_len, patch_size=patch_size,
            in_channels=stem_channels, embed_dim=embed_dim
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                             dropout=dropout, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Dual Heads operating on [CLS] token
        self.width_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )
        self.conc_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.stem(x)
        tokens = self.patch_embed(feat)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        cls_repr = tokens[:, 0]
        w_logits = self.width_head(cls_repr)
        c_pred = self.conc_head(cls_repr)
        return w_logits, c_pred


# ======================================================================
# 2. DATA LOADING & CLEANING (7-AGNR + 9-AGNR)
# ======================================================================

def load_multi_width_data(consolidated_dir: str, pristine_dir: str,
                          samples_per_conc=5000, spectrum_len=150, seed=42):
    print("=" * 75)
    print("MULTI-WIDTH DATA LOADING & CLEANING (7-AGNR & 9-AGNR)")
    print("=" * 75)

    # 1. Load Pristine References (prefer newly calculated base directory files)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]

    cand_p7 = [
        project_root / "7_agnr_pris.npy",
        Path(pristine_dir) / "7_agnr_pris.npy",
        Path(pristine_dir) / "pristine.npy",
    ]
    cand_p9 = [
        project_root / "9_agnr_pris.npy",
        Path(pristine_dir) / "9_agnr_pris.npy",
        Path(pristine_dir) / "pristine_9.npy",
    ]

    p7_path = next(p for p in cand_p7 if p.exists())
    p9_path = next(p for p in cand_p9 if p.exists())

    p7 = np.load(str(p7_path))[:spectrum_len].astype(np.float32)
    p9 = np.load(str(p9_path))[:spectrum_len].astype(np.float32)

    p7_safe = np.where(p7 > 1e-12, p7, 1.0)
    p9_safe = np.where(p9 > 1e-12, p9, 1.0)

    print(f"✓ Loaded Pristine 7-AGNR from {p7_path}: max={np.max(p7):.4f}, length={spectrum_len}")
    print(f"✓ Loaded Pristine 9-AGNR from {p9_path}: max={np.max(p9):.4f}, length={spectrum_len}")

    # 2. Open Memory-Mapped Consolidated Files
    s7_path = Path(consolidated_dir) / "size_7.npy"
    s9_path = Path(consolidated_dir) / "size_9.npy"

    s7_mmap = np.load(str(s7_path), mmap_mode="r")  # (34, 10000, 300)
    s9_mmap = np.load(str(s9_path), mmap_mode="r")  # (49, 10000, 300)

    print(f"✓ Opened size_7.npy: shape={s7_mmap.shape} (34 concentrations, 10k configs each)")
    print(f"✓ Opened size_9.npy: shape={s9_mmap.shape} (49 concentrations, 10k configs each)")

    X_all, y_w_all, y_c_all = [], [], []

    # Process 7-AGNR (width label = 0)
    concs_7 = np.arange(2, 70, 2)  # 34 concentrations (2, 4, ..., 68)
    print(f"\n[INFO] Loading 7-AGNR ({len(concs_7)} concentrations, {samples_per_conc} samples each)...")
    for idx, c in enumerate(concs_7):
        if idx >= s7_mmap.shape[0]:
            break
        raw = np.array(s7_mmap[idx, :samples_per_conc, :spectrum_len], dtype=np.float32)
        norm = np.clip(raw / p7_safe, 0.0, 1.0)
        X_all.append(norm)
        y_w_all.append(np.zeros(samples_per_conc, dtype=np.int64))  # 0 -> size 7
        y_c_all.append(np.full(samples_per_conc, c, dtype=np.float32))

    # Process 9-AGNR (width label = 1)
    concs_9 = np.arange(2, 100, 2)  # 49 concentrations (2, 4, ..., 98)
    print(f"[INFO] Loading 9-AGNR ({len(concs_9)} concentrations, {samples_per_conc} samples each)...")
    for idx, c in enumerate(concs_9):
        if idx >= s9_mmap.shape[0]:
            break
        raw = np.array(s9_mmap[idx, :samples_per_conc, :spectrum_len], dtype=np.float32)
        norm = np.clip(raw / p9_safe, 0.0, 1.0)
        X_all.append(norm)
        y_w_all.append(np.ones(samples_per_conc, dtype=np.int64))   # 1 -> size 9
        y_c_all.append(np.full(samples_per_conc, c, dtype=np.float32))

    X = np.concatenate(X_all, axis=0)
    y_width = np.concatenate(y_w_all, axis=0)
    y_conc = np.concatenate(y_c_all, axis=0)

    print(f"\n✓ Combined Dataset: {len(X):,} total samples")
    print(f"  7-AGNR samples: {np.sum(y_width == 0):,} | 9-AGNR samples: {np.sum(y_width == 1):,}")
    print(f"  Feature shape: {X.shape}, Conc range: [{y_conc.min():.0f}, {y_conc.max():.0f}]")

    # 3. Train / Val / Test Partitioning (70% / 15% / 15%)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))
    n_train = int(len(X) * 0.70)
    n_val = int(len(X) * 0.15)

    idx_tr = perm[:n_train]
    idx_va = perm[n_train:n_train + n_val]
    idx_te = perm[n_train + n_val:]

    train_data = (X[idx_tr], y_width[idx_tr], y_conc[idx_tr])
    val_data = (X[idx_va], y_width[idx_va], y_conc[idx_va])
    test_data = (X[idx_te], y_width[idx_te], y_conc[idx_te])

    print(f"✓ Splits: Train={len(idx_tr):,}, Val={len(idx_va):,}, Test={len(idx_te):,}")
    print("=" * 75 + "\n")

    return train_data, val_data, test_data, p7, p9


# ======================================================================
# 3. TRAINING & EVALUATION ROUTINES
# ======================================================================

def compute_multi_metrics(w_preds, w_true, c_preds, c_true):
    w_acc = float(np.mean(w_preds == w_true)) * 100.0
    err = np.asarray(c_preds, float) - np.asarray(c_true, float)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    max_err = float(np.max(np.abs(err)))

    # Per-width breakdown
    mask7 = (w_true == 0)
    mask9 = (w_true == 1)

    m7_acc = float(np.mean(w_preds[mask7] == 0)) * 100.0 if np.any(mask7) else 0.0
    m9_acc = float(np.mean(w_preds[mask9] == 1)) * 100.0 if np.any(mask9) else 0.0

    m7_mae = float(np.mean(np.abs(err[mask7]))) if np.any(mask7) else 0.0
    m7_rmse = float(np.sqrt(np.mean(err[mask7] ** 2))) if np.any(mask7) else 0.0
    m9_mae = float(np.mean(np.abs(err[mask9]))) if np.any(mask9) else 0.0
    m9_rmse = float(np.sqrt(np.mean(err[mask9] ** 2))) if np.any(mask9) else 0.0

    return {
        "Width_Acc_Overall": w_acc,
        "Width_Acc_7": m7_acc,
        "Width_Acc_9": m9_acc,
        "Conc_MAE_Overall": mae,
        "Conc_RMSE_Overall": rmse,
        "Conc_Max_Error": max_err,
        "Conc_MAE_7": m7_mae,
        "Conc_RMSE_7": m7_rmse,
        "Conc_MAE_9": m9_mae,
        "Conc_RMSE_9": m9_rmse,
    }


def train_multi_mlp(train_data, val_data, test_data, save_path,
                    epochs=70, batch_size=256, lr=1e-3, alpha_width=10.0, patience=15):
    print("=" * 75)
    print("TRAINING MULTI-TASK CONDUCTANCE MLP")
    print("=" * 75)
    device = torch.device("cpu")

    X_tr, yw_tr, yc_tr = train_data
    X_va, yw_va, yc_va = val_data
    X_te, yw_te, yc_te = test_data

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(yw_tr, dtype=torch.long),
        torch.tensor(yc_tr, dtype=torch.float32).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va, dtype=torch.float32),
        torch.tensor(yw_va, dtype=torch.long),
        torch.tensor(yc_va, dtype=torch.float32).unsqueeze(1),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MultiTaskConductanceMLP(input_length=X_tr.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_mae": []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_ce = 0.0
        train_mse = 0.0

        for xb, ywb, ycb in train_loader:
            optimizer.zero_grad()
            w_logits, c_pred = model(xb)
            l_ce = ce_loss_fn(w_logits, ywb)
            l_mse = mse_loss_fn(c_pred, ycb)
            loss = l_mse + alpha_width * l_ce
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(xb)
            train_ce += l_ce.item() * len(xb)
            train_mse += l_mse.item() * len(xb)

        train_loss /= len(train_ds)
        train_ce /= len(train_ds)
        train_mse /= len(train_ds)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        correct_w = 0
        total_c_err = 0.0

        with torch.no_grad():
            for xb, ywb, ycb in val_loader:
                w_logits, c_pred = model(xb)
                l_ce = ce_loss_fn(w_logits, ywb)
                l_mse = mse_loss_fn(c_pred, ycb)
                v_loss = l_mse + alpha_width * l_ce
                val_loss += v_loss.item() * len(xb)

                pred_w = torch.argmax(w_logits, dim=1)
                correct_w += (pred_w == ywb).sum().item()
                total_c_err += torch.abs(c_pred - ycb).sum().item()

        val_loss /= len(val_ds)
        val_acc = (correct_w / len(val_ds)) * 100.0
        val_mae = total_c_err / len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_mae"].append(val_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            star = " * Best"
        else:
            no_improve += 1
            star = ""

        if epoch % 5 == 0 or star or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train: {train_loss:.3f} (MSE:{train_mse:.2f}, CE:{train_ce:.3f}) | "
                  f"Val: {val_loss:.3f} (Width Acc: {val_acc:.2f}%, Conc MAE: {val_mae:.3f}) | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}{star}")

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    elapsed = time.time() - start_time
    print(f"\n✓ MLP training completed in {elapsed:.1f}s (Best Val Loss: {best_val_loss:.4f})")

    # Load and save best weights
    model.load_state_dict(best_state)
    checkpoint = {
        "model_state_dict": best_state,
        "args": {"input_length": X_tr.shape[1], "hidden_dims": [256, 128, 64]},
        "val_loss": best_val_loss,
    }
    torch.save(checkpoint, save_path)
    print(f"✓ Saved MLP checkpoint to: {save_path}")

    # Evaluate on held-out test set
    model.eval()
    with torch.no_grad():
        w_logits, c_pred = model(torch.tensor(X_te, dtype=torch.float32))
        pred_w = torch.argmax(w_logits, dim=1).numpy()
        pred_c = c_pred.squeeze().numpy()

    metrics = compute_multi_metrics(pred_w, yw_te, pred_c, yc_te)
    print(f"✓ MLP Test Results: Width Acc = {metrics['Width_Acc_Overall']:.2f}% (7: {metrics['Width_Acc_7']:.1f}%, 9: {metrics['Width_Acc_9']:.1f}%) | Conc MAE = {metrics['Conc_MAE_Overall']:.3f}, RMSE = {metrics['Conc_RMSE_Overall']:.3f}\n")
    return metrics, (pred_w, pred_c), history


def train_multi_transformer(train_data, val_data, test_data, save_path,
                            epochs=40, batch_size=256, lr=5e-4, alpha_width=10.0, patience=12):
    print("=" * 75)
    print("TRAINING MULTI-TASK PATCHED TRANSFORMER V2")
    print("=" * 75)
    device = torch.device("cpu")

    X_tr, yw_tr, yc_tr = train_data
    X_va, yw_va, yc_va = val_data
    X_te, yw_te, yc_te = test_data

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(yw_tr, dtype=torch.long),
        torch.tensor(yc_tr, dtype=torch.float32).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va, dtype=torch.float32),
        torch.tensor(yw_va, dtype=torch.long),
        torch.tensor(yc_va, dtype=torch.float32).unsqueeze(1),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MultiTaskPatchedTransformerV2(
        seq_len=X_tr.shape[1], patch_size=10, stem_channels=32,
        embed_dim=128, depth=4, num_heads=4, mlp_ratio=4.0,
        dropout=0.1, drop_path_rate=0.1
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_mae": []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_ce = 0.0
        train_mse = 0.0

        for xb, ywb, ycb in train_loader:
            optimizer.zero_grad()
            w_logits, c_pred = model(xb)
            l_ce = ce_loss_fn(w_logits, ywb)
            l_mse = mse_loss_fn(c_pred, ycb)
            loss = l_mse + alpha_width * l_ce
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(xb)
            train_ce += l_ce.item() * len(xb)
            train_mse += l_mse.item() * len(xb)

        train_loss /= len(train_ds)
        train_ce /= len(train_ds)
        train_mse /= len(train_ds)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        correct_w = 0
        total_c_err = 0.0

        with torch.no_grad():
            for xb, ywb, ycb in val_loader:
                w_logits, c_pred = model(xb)
                l_ce = ce_loss_fn(w_logits, ywb)
                l_mse = mse_loss_fn(c_pred, ycb)
                v_loss = l_mse + alpha_width * l_ce
                val_loss += v_loss.item() * len(xb)

                pred_w = torch.argmax(w_logits, dim=1)
                correct_w += (pred_w == ywb).sum().item()
                total_c_err += torch.abs(c_pred - ycb).sum().item()

        val_loss /= len(val_ds)
        val_acc = (correct_w / len(val_ds)) * 100.0
        val_mae = total_c_err / len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_mae"].append(val_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            star = " * Best"
        else:
            no_improve += 1
            star = ""

        if epoch % 5 == 0 or star or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train: {train_loss:.3f} (MSE:{train_mse:.2f}, CE:{train_ce:.3f}) | "
                  f"Val: {val_loss:.3f} (Width Acc: {val_acc:.2f}%, Conc MAE: {val_mae:.3f}) | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}{star}")

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    elapsed = time.time() - start_time
    print(f"\n✓ Transformer training completed in {elapsed:.1f}s (Best Val Loss: {best_val_loss:.4f})")

    # Load and save best weights
    model.load_state_dict(best_state)
    checkpoint = {
        "model_state_dict": best_state,
        "args": {
            "seq_len": X_tr.shape[1],
            "patch_size": 10,
            "embed_dim": 128,
            "depth": 4,
            "num_heads": 4,
        },
        "val_loss": best_val_loss,
    }
    torch.save(checkpoint, save_path)
    print(f"✓ Saved Transformer checkpoint to: {save_path}")

    # Evaluate on held-out test set
    model.eval()
    with torch.no_grad():
        w_logits, c_pred = model(torch.tensor(X_te, dtype=torch.float32))
        pred_w = torch.argmax(w_logits, dim=1).numpy()
        pred_c = c_pred.squeeze().numpy()

    metrics = compute_multi_metrics(pred_w, yw_te, pred_c, yc_te)
    print(f"✓ Transformer Test Results: Width Acc = {metrics['Width_Acc_Overall']:.2f}% (7: {metrics['Width_Acc_7']:.1f}%, 9: {metrics['Width_Acc_9']:.1f}%) | Conc MAE = {metrics['Conc_MAE_Overall']:.3f}, RMSE = {metrics['Conc_RMSE_Overall']:.3f}\n")
    return metrics, (pred_w, pred_c), history


def train_multi_xgb(train_data, test_data, save_dir):
    print("=" * 75)
    print("TRAINING MULTI-TASK XGBOOST BASELINES")
    print("=" * 75)
    X_tr, yw_tr, yc_tr = train_data
    X_te, yw_te, yc_te = test_data

    # 1. Width Classifier
    print("[INFO] Training XGBoost Width Classifier...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        tree_method="hist", random_state=42, n_jobs=-1
    )
    xgb_clf.fit(X_tr, yw_tr)
    pred_w = xgb_clf.predict(X_te)
    xgb_clf.save_model(os.path.join(save_dir, "multi_width_xgb_width.json"))

    # 2. Concentration Regressor
    print("[INFO] Training XGBoost Concentration Regressor...")
    xgb_reg = xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", random_state=42, n_jobs=-1
    )
    # Augment input with predicted width probability for joint inference
    prob_w_tr = xgb_clf.predict_proba(X_tr)[:, 1:2]
    prob_w_te = xgb_clf.predict_proba(X_te)[:, 1:2]

    X_tr_aug = np.hstack([X_tr, prob_w_tr])
    X_te_aug = np.hstack([X_te, prob_w_te])

    xgb_reg.fit(X_tr_aug, yc_tr)
    pred_c = xgb_reg.predict(X_te_aug)
    xgb_reg.save_model(os.path.join(save_dir, "multi_width_xgb_conc.json"))

    metrics = compute_multi_metrics(pred_w, yw_te, pred_c, yc_te)
    print(f"✓ XGBoost Test Results: Width Acc = {metrics['Width_Acc_Overall']:.2f}% | Conc MAE = {metrics['Conc_MAE_Overall']:.3f}, RMSE = {metrics['Conc_RMSE_Overall']:.3f}\n")
    return metrics, (pred_w, pred_c)


# ======================================================================
# 4. PLOTTING & VISUALIZATION
# ======================================================================

def plot_benchmark_results(test_data, mlp_preds, tf_preds, xgb_preds,
                           mlp_hist, tf_hist, out_dir):
    print("Generating comprehensive multi-width comparison plots...")
    _, yw_te, yc_te = test_data
    mlp_pw, mlp_pc = mlp_preds
    tf_pw, tf_pc = tf_preds
    xgb_pw, xgb_pc = xgb_preds

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Plot 1: Concentration Scatter Plot (Predicted vs True) for 7-AGNR and 9-AGNR
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
    models = [
        ("ConductanceMLP (PINN)", mlp_pc, "#1f77b4"),
        ("Patched Transformer v2", tf_pc, "#2ca02c"),
        ("XGBoost Regressor", xgb_pc, "#ff7f0e"),
    ]

    for ax, (name, preds, color) in zip(axes, models):
        # 7-AGNR in blue-ish, 9-AGNR in orange-ish
        mask7 = (yw_te == 0)
        mask9 = (yw_te == 1)

        ax.scatter(yc_te[mask7], preds[mask7], alpha=0.15, s=8, color=color, label="7-AGNR (c ≤ 68)")
        ax.scatter(yc_te[mask9], preds[mask9], alpha=0.15, s=8, color="#d62728", label="9-AGNR (c ≤ 98)")

        min_val, max_val = 0, 100
        ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, alpha=0.7, label="Ideal")

        mae = np.mean(np.abs(preds - yc_te))
        rmse = np.sqrt(np.mean((preds - yc_te) ** 2))
        ax.set_title(f"{name}\nMAE: {mae:.2f} | RMSE: {rmse:.2f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("True Impurity Concentration $c$", fontsize=10)
        ax.set_ylabel("Predicted Concentration $\hat{c}$", fontsize=10)
        ax.set_xlim(0, 102)
        ax.set_ylim(0, 102)
        ax.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "multi_width_scatter.png"))
    plt.close()

    # Plot 2: Error Distributions
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for name, preds, color in models:
        err = preds - yc_te
        ax.hist(err, bins=80, range=(-15, 15), alpha=0.5, label=name, color=color, density=True)
    ax.axvline(0, color="k", linestyle="--", alpha=0.7)
    ax.set_xlabel("Prediction Error ($\hat{c} - c$)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title("Multi-Width Error Distributions (7-AGNR & 9-AGNR)", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "multi_width_error_dist.png"))
    plt.close()

    # Plot 3: Loss & Accuracy Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Loss
    ax1.plot(mlp_hist["train_loss"], label="MLP Train Loss", color="#1f77b4", linestyle="--")
    ax1.plot(mlp_hist["val_loss"], label="MLP Val Loss", color="#1f77b4")
    ax1.plot(tf_hist["train_loss"], label="Transformer Train Loss", color="#2ca02c", linestyle="--")
    ax1.plot(tf_hist["val_loss"], label="Transformer Val Loss", color="#2ca02c")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Total Loss ($\mathcal{L}_{\mathrm{MSE}} + 10 \cdot \mathcal{L}_{\mathrm{CE}}$)", fontsize=11)
    ax1.set_title("Multi-Task Training Loss", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True, fontsize=9)

    # Width Accuracy & Conc MAE
    ax2.plot(mlp_hist["val_acc"], label="MLP Width Acc (%)", color="#1f77b4")
    ax2.plot(tf_hist["val_acc"], label="Transformer Width Acc (%)", color="#2ca02c")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Validation Width Accuracy (%)", fontsize=11)
    ax2.set_title("Width Identification Accuracy (7-AGNR vs 9-AGNR)", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "multi_width_training_curves.png"))
    plt.close()

    print("✓ Plots saved: multi_width_scatter.png, multi_width_error_dist.png, multi_width_training_curves.png")


# ======================================================================
# 5. MAIN EXECUTION PIPELINE
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Width Joint Training Pipeline (7-AGNR & 9-AGNR).")
    parser.add_argument("--data-dir", type=str,
                        default="/run/media/shardul/storage/machine_learning/transmission_data/transmission_results/consolidated_data",
                        help="Path to consolidated data folder")
    parser.add_argument("--samples-per-conc", type=int, default=4000,
                        help="Samples per concentration (default: 4000)")
    parser.add_argument("--spectrum-len", type=int, default=150,
                        help="Number of spectral channels (default: 150)")
    parser.add_argument("--mlp-epochs", type=int, default=60,
                        help="Max epochs for MLP (default: 60)")
    parser.add_argument("--tf-epochs", type=int, default=30,
                        help="Max epochs for Transformer (default: 30)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    pristine_dir = project_root / "data" / "raw" / "transmission_results"

    mlp_save_path = script_dir / "multi_width_pinn_mlp.pt"
    tf_save_path = script_dir / "multi_width_transformer.pt"

    # 1. Load and clean multi-width data
    train_data, val_data, test_data, p7, p9 = load_multi_width_data(
        consolidated_dir=args.data_dir,
        pristine_dir=str(pristine_dir),
        samples_per_conc=args.samples_per_conc,
        spectrum_len=args.spectrum_len,
    )

    # 2. Train ConductanceMLP
    mlp_metrics, mlp_preds, mlp_hist = train_multi_mlp(
        train_data, val_data, test_data,
        save_path=str(mlp_save_path),
        epochs=args.mlp_epochs,
        batch_size=256,
        lr=1e-3,
        alpha_width=10.0,
        patience=15,
    )

    # 3. Train Patched Transformer v2
    tf_metrics, tf_preds, tf_hist = train_multi_transformer(
        train_data, val_data, test_data,
        save_path=str(tf_save_path),
        epochs=args.tf_epochs,
        batch_size=256,
        lr=5e-4,
        alpha_width=10.0,
        patience=12,
    )

    # 4. Train XGBoost Baselines
    xgb_metrics, xgb_preds = train_multi_xgb(
        train_data, test_data,
        save_dir=str(script_dir),
    )

    # 5. Generate Benchmark Plots
    plot_benchmark_results(
        test_data, mlp_preds, tf_preds, xgb_preds,
        mlp_hist, tf_hist, str(script_dir)
    )

    # 6. Save Metrics JSON
    metrics_summary = {
        "ConductanceMLP": mlp_metrics,
        "PatchedTransformerV2": tf_metrics,
        "XGBoost": xgb_metrics,
    }
    metrics_file = script_dir / "multi_width_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"{'MULTI-WIDTH BENCHMARK RESULTS (7-AGNR & 9-AGNR)':^80}")
    print("=" * 80)
    print(f"{'Model':<25} {'Width Acc':>11} {'Overall MAE':>13} {'7-AGNR MAE':>13} {'9-AGNR MAE':>13}")
    print("-" * 80)
    for name, m in metrics_summary.items():
        print(f"{name:<25} {m['Width_Acc_Overall']:>10.2f}% {m['Conc_MAE_Overall']:>13.3f} {m['Conc_MAE_7']:>13.3f} {m['Conc_MAE_9']:>13.3f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
