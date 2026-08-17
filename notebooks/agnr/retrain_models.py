#!/usr/bin/env python
"""
retrain_models.py — Retrain ConductanceMLP (PINN) and PatchedTransformerV2 models
using consolidated data from:
    /run/media/shardul/storage/machine_learning/transmission_data/transmission_results/consolidated_data

Following the exact data cleaning process from xgb.ipynb:
1. Normalization by pristine transmission: T_norm = T / T_pristine
2. Bounding & clipping: np.clip(T_norm, 0, 1)
3. Energy window cropping: [0:150] (first 150 energy channels, 0 to 1.50 eV)
4. Train/Val/Test random split (70% train, 15% val, 15% test)
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Set PyTorch threads for multi-core CPU acceleration
torch.set_num_threads(min(20, max(1, os.cpu_count() - 2)))


# ======================================================================
# 1. MODEL DEFINITIONS
# ======================================================================

class ConductanceMLP(nn.Module):
    """Deep fully-connected MLP with LayerNorm, ReLU, Dropout and Noise Augmentation."""
    def __init__(self, input_length: int = 150, hidden_dims: list = None,
                 dropout: float = 0.2, noise_std: float = 0.01):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]
        self.noise_std = noise_std

        layers = []
        in_dim = input_length
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self.regressor = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0:
            noise = 1.0 + torch.randn_like(x) * self.noise_std
            x = x * noise
        return self.regressor(self.mlp(x))


class DropPath(nn.Module):
    """Per-sample stochastic depth."""
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
    """1D ConvStem for rich local spectral feature extraction."""
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
    """1D Patching with linear projection and learned positional embeddings."""
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
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, embed_dim]
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


class PatchedTransformerV2(nn.Module):
    """Patched Transformer v2 with ConvStem and [CLS] regression head."""
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
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, L]
        feat = self.stem(x)     # [B, stem_channels, L]
        tokens = self.patch_embed(feat)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        cls_repr = tokens[:, 0]
        return self.head(cls_repr)


# ======================================================================
# 2. DATA LOADING & CLEANING (xgb.ipynb METHODOLOGY)
# ======================================================================

def load_and_clean_data(consolidated_dir: str, pristine_path: str,
                        concs_range=None, samples_per_conc=5000,
                        spectrum_len=150, seed=42):
    print("=" * 70)
    print("DATA LOADING & CLEANING (xgb.ipynb pipeline)")
    print("=" * 70)

    # 1. Load pristine reference spectrum
    pristine_full = np.load(pristine_path)
    pristine = pristine_full[:spectrum_len].astype(np.float32)
    # Prevent divide by zero
    p_safe = np.where(pristine > 1e-12, pristine, 1.0)
    print(f"✓ Loaded pristine spectrum: length={spectrum_len}, max={np.max(pristine):.4f}")

    # 2. Open consolidated file with memory mapping
    s7_path = Path(consolidated_dir) / "size_7.npy"
    if not s7_path.exists():
        raise FileNotFoundError(f"Consolidated file not found: {s7_path}")

    s7_mmap = np.load(str(s7_path), mmap_mode="r")
    total_concs, total_samples, total_len = s7_mmap.shape
    print(f"✓ Opened size_7.npy: shape={s7_mmap.shape} (34 concentrations, 10k configs each)")

    if concs_range is None:
        # All concentrations from 2 to 68 with step 2
        concs = np.arange(2, 70, 2)
    else:
        concs = np.asarray(concs_range)

    indices = concs // 2 - 1  # 0-indexed concentration index in size_7.npy
    valid_mask = (indices >= 0) & (indices < total_concs)
    indices = indices[valid_mask]
    concs = concs[valid_mask]

    print(f"✓ Selected {len(concs)} concentrations: {concs.tolist()}")
    print(f"✓ Extracting {samples_per_conc} configs per concentration (total = {len(concs) * samples_per_conc:,} samples)...")

    # 3. Clean and normalize data
    X_list = []
    y_list = []

    for idx, c in zip(indices, concs):
        # Extract raw spectra slice: [samples_per_conc, spectrum_len]
        raw_slice = np.array(s7_mmap[idx, :samples_per_conc, :spectrum_len], dtype=np.float32)
        # Normalization and clipping to [0, 1] as in xgb.ipynb
        norm_slice = np.clip(raw_slice / p_safe, 0.0, 1.0)
        X_list.append(norm_slice)
        y_list.append(np.full(samples_per_conc, c, dtype=np.float32))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    print(f"✓ Cleaned Dataset: X shape = {X.shape}, y shape = {y.shape}")
    print(f"  X min = {X.min():.4f}, max = {X.max():.4f}, mean = {X.mean():.4f}")

    # 4. Train / Val / Test split (70% train, 15% val, 15% test)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))
    n_train = int(len(X) * 0.70)
    n_val = int(len(X) * 0.15)

    idx_tr = perm[:n_train]
    idx_va = perm[n_train:n_train + n_val]
    idx_te = perm[n_train + n_val:]

    X_train, y_train = X[idx_tr], y[idx_tr]
    X_val, y_val = X[idx_va], y[idx_va]
    X_test, y_test = X[idx_te], y[idx_te]

    print(f"✓ Data Splits: Train = {len(X_train):,}, Val = {len(X_val):,}, Test = {len(X_test):,}")
    print("=" * 70 + "\n")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), pristine


# ======================================================================
# 3. TRAINING & EVALUATION FUNCTIONS
# ======================================================================

def compute_metrics(y_pred, y_true):
    err = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "Max_Error": float(np.max(np.abs(err))),
    }


def train_mlp_model(train_data, val_data, test_data, save_path,
                    epochs=100, batch_size=256, lr=1e-3, patience=15):
    print("=" * 70)
    print("TRAINING CONDUCTANCE MLP MODEL")
    print("=" * 70)
    device = torch.device("cpu")

    X_tr, y_tr = train_data
    X_va, y_val = val_data
    X_te, y_te = test_data

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                             torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1))
    val_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ConductanceMLP(input_length=X_tr.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * len(xb)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            star = " * Best"
        else:
            no_improve += 1
            star = ""

        if epoch % 5 == 0 or star or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}{star}")

        if no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch} (no improvement for {patience} epochs)")
            break

    elapsed = time.time() - start_time
    print(f"\n✓ MLP training complete in {elapsed:.1f}s (Best Val MSE: {best_val_loss:.4f})")

    # Save best checkpoint
    model.load_state_dict(best_state)
    checkpoint = {
        "model_state_dict": best_state,
        "args": {"spectrum_len": X_tr.shape[1], "hidden_dims": [256, 128, 64, 32]},
        "val_mse": best_val_loss,
    }
    torch.save(checkpoint, save_path)
    print(f"✓ Saved MLP checkpoint to: {save_path}")

    # Evaluate on held-out test set
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(X_te, dtype=torch.float32)).squeeze().numpy()
    m = compute_metrics(test_preds, y_te)
    print(f"✓ Held-out Test Metrics: MAE = {m['MAE']:.3f}, RMSE = {m['RMSE']:.3f}, Max Err = {m['Max_Error']:.3f}\n")
    return m


def train_transformer_model(train_data, val_data, test_data, save_path,
                            epochs=60, batch_size=256, lr=5e-4, patience=15):
    print("=" * 70)
    print("TRAINING PATCHED TRANSFORMER V2 MODEL")
    print("=" * 70)
    device = torch.device("cpu")

    X_tr, y_tr = train_data
    X_va, y_val = val_data
    X_te, y_te = test_data

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                             torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1))
    val_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = PatchedTransformerV2(
        seq_len=X_tr.shape[1], patch_size=10, stem_channels=32,
        embed_dim=128, depth=4, num_heads=4, mlp_ratio=4.0,
        dropout=0.1, drop_path_rate=0.1
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * len(xb)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            star = " * Best"
        else:
            no_improve += 1
            star = ""

        if epoch % 5 == 0 or star or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}{star}")

        if no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch} (no improvement for {patience} epochs)")
            break

    elapsed = time.time() - start_time
    print(f"\n✓ Transformer training complete in {elapsed:.1f}s (Best Val MSE: {best_val_loss:.4f})")

    # Save best checkpoint
    model.load_state_dict(best_state)
    checkpoint = {
        "model_state_dict": best_state,
        "args": {
            "spectrum_len": X_tr.shape[1],
            "patch_size": 10,
            "embed_dim": 128,
            "depth": 4,
            "num_heads": 4,
        },
        "val_mse": best_val_loss,
    }
    torch.save(checkpoint, save_path)
    print(f"✓ Saved Transformer checkpoint to: {save_path}")

    # Evaluate on held-out test set
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(X_te, dtype=torch.float32)).squeeze().numpy()
    m = compute_metrics(test_preds, y_te)
    print(f"✓ Held-out Test Metrics: MAE = {m['MAE']:.3f}, RMSE = {m['RMSE']:.3f}, Max Err = {m['Max_Error']:.3f}\n")
    return m


def main():
    parser = argparse.ArgumentParser(description="Retrain MLP and Transformer models on consolidated data.")
    parser.add_argument("--data-dir", type=str,
                        default="/run/media/shardul/storage/machine_learning/transmission_data/transmission_results/consolidated_data",
                        help="Path to consolidated data folder")
    parser.add_argument("--samples-per-conc", type=int, default=5000,
                        help="Samples per concentration to use (default: 5000)")
    parser.add_argument("--spectrum-len", type=int, default=150,
                        help="Number of energy channels to use (default: 150)")
    parser.add_argument("--mlp-epochs", type=int, default=80,
                        help="Max epochs for MLP (default: 80)")
    parser.add_argument("--tf-epochs", type=int, default=40,
                        help="Max epochs for Transformer (default: 40)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    pristine_path = project_root / "data" / "raw" / "transmission_results" / "pristine.npy"

    mlp_save_path = script_dir / "pinn_agnr_curvature.pt"
    tf_save_path = script_dir / "patched_transformer_v2.pt"

    # Backup existing checkpoints if present
    if mlp_save_path.exists():
        backup_mlp = script_dir / "pinn_agnr_curvature_backup.pt"
        torch.save(torch.load(mlp_save_path, weights_only=False), backup_mlp)
        print(f"[INFO] Backed up old MLP checkpoint to {backup_mlp}")

    if tf_save_path.exists():
        backup_tf = script_dir / "patched_transformer_v2_backup.pt"
        torch.save(torch.load(tf_save_path, weights_only=False), backup_tf)
        print(f"[INFO] Backed up old Transformer checkpoint to {backup_tf}")

    train_data, val_data, test_data, pristine = load_and_clean_data(
        consolidated_dir=args.data_dir,
        pristine_path=str(pristine_path),
        samples_per_conc=args.samples_per_conc,
        spectrum_len=args.spectrum_len,
    )

    mlp_metrics = train_mlp_model(
        train_data, val_data, test_data,
        save_path=mlp_save_path,
        epochs=args.mlp_epochs,
        batch_size=256,
        lr=1e-3,
        patience=15,
    )

    tf_metrics = train_transformer_model(
        train_data, val_data, test_data,
        save_path=tf_save_path,
        epochs=args.tf_epochs,
        batch_size=256,
        lr=5e-4,
        patience=12,
    )

    print("=" * 70)
    print("FINAL BENCHMARK COMPARISON ON CONSOLIDATED TEST SET")
    print("=" * 70)
    print(f"{'Model':<30} {'MAE':>10} {'RMSE':>10} {'Max Error':>12}")
    print("-" * 70)
    print(f"{'ConductanceMLP (PINN)':<30} {mlp_metrics['MAE']:>10.3f} {mlp_metrics['RMSE']:>10.3f} {mlp_metrics['Max_Error']:>12.3f}")
    print(f"{'Patched Transformer v2':<30} {tf_metrics['MAE']:>10.3f} {tf_metrics['RMSE']:>10.3f} {tf_metrics['Max_Error']:>12.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
