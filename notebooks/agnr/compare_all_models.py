"""
Four-Way Comparative Analysis: MLP vs Patched Transformer vs XGBoost vs Physical Misfit
========================================================================================

Evaluates all four approaches on the 2100 held-out test spectra and
produces:
  1. Scatter plots (pred vs true) for each model
  2. Error distribution histograms
  3. Per-concentration MAE breakdown
  4. Summary metrics table
  5. Training convergence curves

Run:  conda run -n ml python compare_all_models.py
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import xgboost as xgb
from pathlib import Path

# ======================================================================
# 1. MODEL DEFINITIONS (inlined so script is self-contained)
# ======================================================================

class ConductanceMLP(nn.Module):
    """Multi-Layer Perceptron for concentration prediction (from pinn_agnr_curvature.py)."""
    def __init__(self, input_length=200, hidden_dims=None, dropout=0.2, noise_std=0.02):
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

    def forward(self, x):
        if self.training and self.noise_std > 0:
            noise = 1.0 + torch.randn_like(x) * self.noise_std
            x = x * noise
        x = self.mlp(x)
        return self.regressor(x)


# --- Patched Transformer v2 components ---

class ConvStem(nn.Module):
    def __init__(self, in_channels=1, mid_channels=16, out_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=7, stride=1, padding=3),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class PatchEmbedding1D(nn.Module):
    def __init__(self, seq_len=200, patch_size=10, in_channels=32, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

    def forward(self, x):
        x = self.proj(x).transpose(1, 2)
        return self.norm(x) + self.pos_embed


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device))
        return x * mask / keep


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4.0, dropout=0.1, drop_path=0.0):
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

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path1(attn_out)
        return x + self.drop_path2(self.mlp(self.norm2(x)))


class PatchedTransformerV2(nn.Module):
    def __init__(self, seq_len=200, patch_size=10, stem_channels=32, embed_dim=128,
                 depth=4, num_heads=4, mlp_ratio=4.0, dropout=0.1,
                 drop_path_rate=0.1, noise_std=0.02):
        super().__init__()
        self.noise_std = noise_std
        self.stem = ConvStem(1, 16, stem_channels)
        self.patch_embed = PatchEmbedding1D(seq_len, patch_size, stem_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.training and self.noise_std > 0:
            x = x * (1.0 + torch.randn_like(x) * self.noise_std)
        x = self.stem(x)
        x = self.patch_embed(x)
        B = x.shape[0]
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])


# ======================================================================
# 2. MISFIT PREDICTION
# ======================================================================

def predict_misfit(test_spectrum_norm, ref_spectra, concentrations, pristine,
                   spectrum_start=20, spectrum_end=150):
    """Physical misfit baseline: argmin of squared difference against references."""
    test_crop = test_spectrum_norm[spectrum_start:spectrum_end]
    ref_crop = ref_spectra[:, spectrum_start:spectrum_end]
    pristine_crop = pristine[spectrum_start:spectrum_end]

    test_unnorm = np.clip(test_crop * pristine_crop, 0.0, pristine_crop)
    ref_unnorm = ref_crop * pristine_crop

    diff = test_unnorm[None, :] - ref_unnorm
    mis = np.sum(diff ** 2, axis=1) / 150.0

    return concentrations[np.argmin(mis)], mis


# ======================================================================
# 3. MAIN: LOAD, PREDICT, COMPARE
# ======================================================================

def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    data_dir = project_root / "data" / "raw" / "transmission_results"
    test_dir = project_root / "data" / "test" / "transmission_results"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Load MLP ---
    mlp_ckpt = torch.load(script_dir / 'pinn_agnr_curvature.pt',
                           map_location=device, weights_only=False)
    spectrum_len = mlp_ckpt.get('args', {}).get('spectrum_len', 200)
    mlp_model = ConductanceMLP(input_length=spectrum_len).to(device)
    mlp_model.load_state_dict(mlp_ckpt['model_state_dict'])
    mlp_model.eval()
    print("✓ MLP loaded")

    # --- Load Transformer ---
    tf_ckpt = torch.load(script_dir / 'patched_transformer_v2.pt',
                          map_location=device, weights_only=False)
    tf_args = tf_ckpt.get('args', {})
    tf_model = PatchedTransformerV2(
        seq_len=tf_args.get('spectrum_len', 200),
        patch_size=tf_args.get('patch_size', 10),
        stem_channels=32,
        embed_dim=tf_args.get('embed_dim', 128),
        depth=tf_args.get('depth', 4),
        num_heads=tf_args.get('num_heads', 4),
        mlp_ratio=4.0,
    ).to(device)
    raw_sd = tf_ckpt['model_state_dict']
    cleaned_sd = {k.replace('_orig_mod.', ''): v for k, v in raw_sd.items()}
    tf_model.load_state_dict(cleaned_sd)
    tf_model.eval()
    print("✓ Transformer loaded")

    # --- Load pristine and build reference spectra / train XGBoost ---
    pristine = np.load(str(data_dir / "pristine.npy"))[:spectrum_len].astype(np.float32)

    conc_range = np.arange(3, 45, 2)
    ref_list = []
    X_train, y_train = [], []
    print("Building reference spectra & preparing training data for XGBoost...")
    for con in conc_range:
        acc = []
        for cfg in range(100):
            fpath = data_dir / f"7_agnr_conc{int(con)}_cfg{cfg}.npy"
            if fpath.exists():
                spec = np.load(str(fpath)).astype(np.float32)[:spectrum_len]
                spec = np.clip(spec, 0, pristine)
                acc.append(spec)
                X_train.append(spec / (pristine + 1e-8))
                y_train.append(con)
        if acc:
            avg = np.mean(acc, axis=0)
            ref_list.append(avg / (pristine + 1e-8))
    ref_spectra = np.array(ref_list)
    print(f"  {len(ref_list)} reference concentrations loaded")

    # --- Train XGBoost Regressor ---
    print("Training XGBoost Regressor on training set...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(np.array(X_train), np.array(y_train))
    print("✓ XGBoost Regressor trained")

    # --- Evaluate on test set (batched for high performance) ---
    test_files = sorted(glob.glob(str(test_dir / "*.npy")))
    print(f"\nEvaluating all 4 models on {len(test_files)} test files...\n")

    true_concs = []
    all_spec_norm = []

    for fpath in test_files:
        filename = os.path.basename(fpath)
        true_c = float(filename.split('_conc')[1].split('_')[0])
        true_concs.append(true_c)

        spec = np.load(fpath).astype(np.float32)[:spectrum_len]
        spec = np.clip(spec, 0, pristine)
        all_spec_norm.append(spec / (pristine + 1e-8))

    true_concs = np.array(true_concs)
    all_spec_norm = np.array(all_spec_norm, dtype=np.float32)

    # Batch PyTorch inference
    with torch.no_grad():
        x_tensor = torch.tensor(all_spec_norm, dtype=torch.float32).to(device)
        mlp_preds = mlp_model(x_tensor).squeeze().cpu().numpy()
        tf_preds = tf_model(x_tensor).squeeze().cpu().numpy()

    # Batch XGBoost inference
    xgb_preds = xgb_model.predict(all_spec_norm)

    # Vectorized / looped Misfit inference
    misfit_preds = []
    for s_norm in all_spec_norm:
        m_pred, _ = predict_misfit(s_norm, ref_spectra, conc_range, pristine)
        misfit_preds.append(m_pred)
    misfit_preds = np.array(misfit_preds)

    # --- Compute metrics ---
    def metrics(pred, true):
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        max_err = np.max(np.abs(pred - true))
        return mae, rmse, max_err

    mlp_mae, mlp_rmse, mlp_max = metrics(mlp_preds, true_concs)
    tf_mae, tf_rmse, tf_max = metrics(tf_preds, true_concs)
    xgb_mae, xgb_rmse, xgb_max = metrics(xgb_preds, true_concs)
    mis_mae, mis_rmse, mis_max = metrics(misfit_preds, true_concs)

    print("=" * 65)
    print(f"{'Method':<28} {'MAE':>10} {'RMSE':>10} {'Max Err':>10}")
    print("-" * 65)
    print(f"{'Deep Learning (MLP)':<28} {mlp_mae:>10.3f} {mlp_rmse:>10.3f} {mlp_max:>10.3f}")
    print(f"{'Patched Transformer v2':<28} {tf_mae:>10.3f} {tf_rmse:>10.3f} {tf_max:>10.3f}")
    print(f"{'XGBoost Regressor':<28} {xgb_mae:>10.3f} {xgb_rmse:>10.3f} {xgb_max:>10.3f}")
    print(f"{'Physical Misfit Baseline':<28} {mis_mae:>10.3f} {mis_rmse:>10.3f} {mis_max:>10.3f}")
    print("=" * 65)

    # ======================================================================
    # 4. PLOTS
    # ======================================================================

    # --- 4a. Scatter plots: Predicted vs True ---
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    models = [
        ("Deep Learning (MLP)", mlp_preds, mlp_mae, mlp_rmse, "#2196F3"),
        ("Patched Transformer v2", tf_preds, tf_mae, tf_rmse, "#FF9800"),
        ("XGBoost Regressor", xgb_preds, xgb_mae, xgb_rmse, "#9C27B0"),
        ("Physical Misfit", misfit_preds, mis_mae, mis_rmse, "#4CAF50"),
    ]
    lo, hi = true_concs.min(), true_concs.max()

    for ax, (name, preds, mae, rmse, color) in zip(axes, models):
        ax.scatter(true_concs, preds, alpha=0.4, s=18, color=color, edgecolors='none')
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect')
        ax.set_title(f"{name}\nMAE: {mae:.2f}  |  RMSE: {rmse:.2f}", fontsize=12)
        ax.set_xlabel("True Concentration")
        ax.set_ylabel("Predicted Concentration")
        ax.set_xlim(lo - 2, hi + 2)
        ax.set_ylim(lo - 5, hi + 5)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper left')

    plt.suptitle("Predicted vs True Concentration (2100 Test Spectra)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(str(script_dir / "compare_scatter.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # --- 4b. Error distribution histograms ---
    fig, ax = plt.subplots(figsize=(11, 5))
    bins = np.linspace(-15, 15, 50)
    ax.hist(mlp_preds - true_concs, bins=bins, alpha=0.45, label=f'MLP (MAE={mlp_mae:.2f})',
            color='#2196F3', density=True)
    ax.hist(tf_preds - true_concs, bins=bins, alpha=0.45, label=f'Transformer (MAE={tf_mae:.2f})',
            color='#FF9800', density=True)
    ax.hist(xgb_preds - true_concs, bins=bins, alpha=0.45, label=f'XGBoost (MAE={xgb_mae:.2f})',
            color='#9C27B0', density=True)
    ax.hist(misfit_preds - true_concs, bins=bins, alpha=0.45, label=f'Misfit (MAE={mis_mae:.2f})',
            color='#4CAF50', density=True)
    ax.axvline(0, color='r', linestyle='--', lw=1.5)
    ax.set_xlabel("Prediction Error (Predicted − True)")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution Comparison Across All 4 Approaches")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(str(script_dir / "compare_error_dist.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # --- 4c. Per-concentration MAE breakdown ---
    unique_concs = np.sort(np.unique(true_concs))
    mlp_per_conc, tf_per_conc, xgb_per_conc, mis_per_conc = [], [], [], []

    for c in unique_concs:
        mask = true_concs == c
        mlp_per_conc.append(np.mean(np.abs(mlp_preds[mask] - c)))
        tf_per_conc.append(np.mean(np.abs(tf_preds[mask] - c)))
        xgb_per_conc.append(np.mean(np.abs(xgb_preds[mask] - c)))
        mis_per_conc.append(np.mean(np.abs(misfit_preds[mask] - c)))

    fig, ax = plt.subplots(figsize=(14, 5))
    w = 0.8
    x = np.arange(len(unique_concs))
    ax.bar(x - 3*w/8, mlp_per_conc, width=w/4, label='MLP', color='#2196F3', alpha=0.85)
    ax.bar(x - w/8,   tf_per_conc,  width=w/4, label='Transformer', color='#FF9800', alpha=0.85)
    ax.bar(x + w/8,   xgb_per_conc, width=w/4, label='XGBoost', color='#9C27B0', alpha=0.85)
    ax.bar(x + 3*w/8, mis_per_conc, width=w/4, label='Misfit', color='#4CAF50', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(c)}" for c in unique_concs], fontsize=8)
    ax.set_xlabel("True Concentration")
    ax.set_ylabel("MAE")
    ax.set_title("Per-Concentration MAE Breakdown Across All Models")
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(str(script_dir / "compare_per_conc.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # --- 4d. Training curves (MLP and Transformer) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mlp_train = mlp_ckpt.get('train_losses', [])
    mlp_val = mlp_ckpt.get('val_losses', [])
    if mlp_train:
        axes[0].plot(mlp_train, label='Train', color='#2196F3')
        axes[0].plot(mlp_val, label='Val', color='#2196F3', linestyle='--')
        axes[0].set_title("MLP Training Curves")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.2)

    tf_train = tf_ckpt.get('train_losses', [])
    tf_val = tf_ckpt.get('val_losses', [])
    if tf_train:
        axes[1].plot(tf_train, label='Train', color='#FF9800')
        axes[1].plot(tf_val, label='Val', color='#FF9800', linestyle='--')
        axes[1].set_title("Transformer Training Curves")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.2)

    plt.suptitle("Deep Learning Training Convergence", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(str(script_dir / "compare_training.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print("\n✓ Plots saved: compare_scatter.png, compare_error_dist.png, "
          "compare_per_conc.png, compare_training.png")


if __name__ == "__main__":
    main()
