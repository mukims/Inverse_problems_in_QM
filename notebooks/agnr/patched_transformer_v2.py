"""
Patched Transformer v2 — Concentration Prediction
===================================================

Improved 1D-Patched Transformer for inverse scattering:
    T(E) → impurity concentration (scalar regression).

Key improvements over v1:
  1. Predicts concentration (scalar) — matching the successful MLP task.
  2. Uses CurvatureMisfit physics regulariser (proven in MLP results).
  3. Convolutional stem before patching for local feature extraction.
  4. Learnable [CLS] token for clean sequence aggregation.
  5. Stochastic depth (layer drop) for regularisation.
  6. Warmup + cosine annealing LR schedule.
  7. Early stopping with patience.
  8. Gradient clipping for transformer stability.
  9. Built-in test evaluation pipeline.

Reuses NormalizedTransmissionsDataset and CurvatureMisfit from
pinn_agnr_curvature.py.
"""

import os
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from pinn_agnr_curvature import (
    NormalizedTransmissionsDataset,
    CurvatureMisfit,
)
from torch.utils.data import TensorDataset


# ======================================================================
# 0. RAM-CACHED DATASET (eliminates per-batch disk I/O)
# ======================================================================

def cache_dataset(dataset, max_samples=None, desc="Caching dataset"):
    """
    Pre-loads a NormalizedTransmissionsDataset into contiguous RAM tensors.

    Args:
        dataset:     source dataset (disk-backed)
        max_samples: if set, randomly subsample to this many items
        desc:        tqdm description
    """
    n_total = len(dataset)

    if max_samples and max_samples < n_total:
        indices = np.random.permutation(n_total)[:max_samples]
        n = max_samples
        print(f"  Subsampling {n:,} / {n_total:,} samples")
    else:
        indices = range(n_total)
        n = n_total

    x0, y0 = dataset[0]
    x_all = torch.empty(n, *x0.shape, dtype=torch.float32)
    y_all = torch.empty(n, *y0.shape, dtype=torch.float32)

    for i, idx in enumerate(tqdm(indices, desc=desc, unit="sample")):
        x_all[i], y_all[i] = dataset[idx]

    print(f"  Cached {n:,} samples  "
          f"({x_all.nelement() * 4 / 1e6:.0f} MB x + "
          f"{y_all.nelement() * 4 / 1e6:.0f} MB y)")
    return TensorDataset(x_all, y_all)


# ======================================================================
# 1. CONVOLUTIONAL STEM
# ======================================================================

class ConvStem(nn.Module):
    """
    Lightweight 1D conv stem that extracts local features before patching.
    This gives the transformer richer per-patch representations than
    raw spectrum values.  Output channels become the patch input dim.
    """

    def __init__(self, out_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, out_channels, kernel_size=7, stride=1, padding=3),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, L] → [B, out_channels, L]"""
        return self.net(x)


# ======================================================================
# 2. PATCH EMBEDDING (with conv stem input)
# ======================================================================

class PatchEmbedding1D(nn.Module):
    """
    Non-overlapping 1D patching with learned positional encoding.
    Accepts multi-channel input from the conv stem.
    """

    def __init__(self, seq_len: int = 200, patch_size: int = 10,
                 in_channels: int = 32, embed_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size

        # Non-overlapping conv IS a patched linear projection
        self.proj = nn.Conv1d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Learned absolute positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_channels, L] → [B, num_patches, embed_dim]"""
        x = self.proj(x)           # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)     # [B, num_patches, embed_dim]
        x = self.norm(x)
        x = x + self.pos_embed
        return x


# ======================================================================
# 3. STOCHASTIC DEPTH (DropPath)
# ======================================================================

class DropPath(nn.Module):
    """Per-sample stochastic depth (drops entire residual branches)."""

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


# ======================================================================
# 4. TRANSFORMER ENCODER BLOCK (Pre-Norm + DropPath)
# ======================================================================

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with stochastic depth."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
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
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path1(attn_out)

        # Pre-norm FFN
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# ======================================================================
# 5. PATCHED TRANSFORMER MODEL
# ======================================================================

class PatchedTransformerV2(nn.Module):
    """
    1D-Patched Transformer for concentration prediction.

    Architecture:
      1. ConvStem         — local feature extraction (1→32 channels)
      2. PatchEmbedding   — chop into patches, project to embed_dim
      3. [CLS] token      — prepended learnable aggregation token
      4. TransformerBlocks — global self-attention with stochastic depth
      5. MLP head          — [CLS] → concentration (scalar)
    """

    def __init__(
        self,
        seq_len: int = 200,
        patch_size: int = 10,
        stem_channels: int = 16,
        embed_dim: int = 64,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        noise_std: float = 0.02,
    ):
        super().__init__()
        self.noise_std = noise_std

        # --- 1. CONV STEM ---
        self.stem = ConvStem(out_channels=stem_channels)

        # --- 2. PATCH EMBEDDING ---
        self.patch_embed = PatchEmbedding1D(
            seq_len=seq_len,
            patch_size=patch_size,
            in_channels=stem_channels,
            embed_dim=embed_dim,
        )
        num_patches = seq_len // patch_size

        # --- 3. [CLS] TOKEN ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # --- 4. TRANSFORMER BLOCKS (linearly increasing drop-path) ---
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # --- 5. MLP HEAD (CLS → concentration) ---
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L] or [B, 1, L] → [B, 1] predicted concentration."""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, L]

        # Data augmentation (training only)
        if self.training and self.noise_std > 0:
            noise = 1.0 + torch.randn_like(x) * self.noise_std
            x = x * noise

        # 1. Conv stem
        x = self.stem(x)              # [B, stem_ch, L]

        # 2. Patch + embed
        x = self.patch_embed(x)       # [B, num_patches, embed_dim]

        # 3. Prepend [CLS]
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 1+num_patches, embed_dim]

        # 4. Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # 5. Take [CLS] output → head
        cls_out = x[:, 0]             # [B, embed_dim]
        return self.head(cls_out)     # [B, 1]


# ======================================================================
# 6. ATTENTION WEIGHT EXTRACTION (Interpretability)
# ======================================================================

def extract_attention_weights(model: PatchedTransformerV2,
                              x_input: torch.Tensor) -> list:
    """
    Manually runs through the transformer blocks with need_weights=True
    to capture per-head self-attention matrices.

    Returns a list of [B, num_heads, 1+num_patches, 1+num_patches] tensors.
    """
    attn_maps = []
    model.eval()
    with torch.no_grad():
        if x_input.dim() == 2:
            x_input = x_input.unsqueeze(1)

        x = model.stem(x_input)
        x = model.patch_embed(x)

        B = x.shape[0]
        cls = model.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        for blk in model.blocks:
            x_norm = blk.norm1(x)
            attn_out, attn_w = blk.attn(
                x_norm, x_norm, x_norm,
                need_weights=True, average_attn_weights=False,
            )
            x = x + attn_out
            x = x + blk.mlp(blk.norm2(x))
            attn_maps.append(attn_w)

    return attn_maps


# ======================================================================
# 7. WARMUP + COSINE SCHEDULER
# ======================================================================

class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.min_lr = min_lr

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            scale = self.min_lr / self.base_lrs[0] + (1 - self.min_lr / self.base_lrs[0]) * 0.5 * (1 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale


# ======================================================================
# 8. TRAINING LOOP
# ======================================================================

def train_patched_transformer(
    dataset,
    misfit_module: CurvatureMisfit,
    num_epochs: int = 200,
    val_split: float = 0.2,
    batch_size: int = 256,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    misfit_weight: float = 0.1,
    patience: int = 30,
    warmup_epochs: int = 10,
    input_length: int = 200,
    patch_size: int = 10,
    embed_dim: int = 64,
    depth: int = 3,
    num_heads: int = 4,
    grad_clip: float = 1.0,
    device_str: str = None,
):
    """Train the PatchedTransformerV2 for concentration prediction."""

    device = (torch.device(device_str) if device_str
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training on: {device}")

    # Use all available CPU cores
    if device.type == 'cpu':
        import multiprocessing
        n_threads = multiprocessing.cpu_count()
        torch.set_num_threads(n_threads)
        print(f"CPU threads: {n_threads}")

    # --- Model ---
    model = PatchedTransformerV2(
        seq_len=input_length,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
    ).to(device)
    misfit_module = misfit_module.to(device)

    # torch.compile for op fusion (significant CPU speedup)
    try:
        model = torch.compile(model)
        print("Model compiled with torch.compile()")
    except Exception:
        print("torch.compile() not available, using eager mode")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # --- Optimizer & Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, num_epochs)

    # --- Data ---
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size,
                              num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size,
                            num_workers=2, persistent_workers=True)

    # --- Training ---
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_weights = None
    epochs_no_improve = 0

    for epoch in (epoch_bar := tqdm(range(num_epochs), desc="Training", unit="epoch")):
        scheduler.step(epoch)

        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        running_mse = 0.0
        running_misfit = 0.0

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1:03d}", leave=False, unit="batch"):
            xb, yb = xb.to(device), yb.to(device)

            preds = model(xb)
            yb_flat = yb.squeeze(-1)
            preds_flat = preds.squeeze(-1)

            mse_loss = F.mse_loss(preds_flat, yb_flat)
            m_loss = misfit_module(xb, preds)
            loss = mse_loss + misfit_weight * m_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += loss.item()
            running_mse += mse_loss.item()
            running_misfit += m_loss.item()

        avg_train = running_loss / len(train_loader)
        train_losses.append(avg_train)

        # ---- VALIDATE ----
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                yb_flat = yb.squeeze(-1)
                preds_flat = preds.squeeze(-1)

                mse_loss = F.mse_loss(preds_flat, yb_flat)
                m_loss = misfit_module(xb, preds)
                loss = mse_loss + misfit_weight * m_loss
                running_val += loss.item()

        avg_val = running_val / len(val_loader)
        val_losses.append(avg_val)

        # ---- EARLY STOPPING ----
        current_lr = optimizer.param_groups[0]['lr']
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            marker = "*"
        else:
            epochs_no_improve += 1
            marker = ""

        avg_mse = running_mse / len(train_loader)
        avg_mis = running_misfit / len(train_loader)

        epoch_bar.set_postfix_str(
            f"train={avg_train:.4f} val={avg_val:.4f} "
            f"mse={avg_mse:.3f} lr={current_lr:.1e} {marker}"
        )

        if epochs_no_improve >= patience:
            tqdm.write(f"\nEarly stopping at epoch {epoch+1} "
                       f"(no improvement for {patience} epochs)")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, train_losses, val_losses


# ======================================================================
# 9. TEST EVALUATION
# ======================================================================

def test_model(model, test_dataset, device=None):
    """Evaluate on a held-out test set. Returns predictions, labels, metrics."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds = model(xb).squeeze(-1).cpu()
            all_preds.append(preds)
            all_labels.append(yb.squeeze(-1))

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    mae = np.mean(np.abs(preds - labels))
    rmse = np.sqrt(np.mean((preds - labels) ** 2))

    print(f"\n{'='*40}")
    print(f"  Test Results")
    print(f"  MAE:  {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"{'='*40}\n")

    return preds, labels, {"mae": mae, "rmse": rmse}


# ======================================================================
# 10. VISUALISATION
# ======================================================================

def visualize_prediction(model, dataset, idx=0):
    """Plot spectrum, attention heatmap, and prediction vs truth."""
    device = next(model.parameters()).device
    model.eval()

    x, y_true = dataset[idx]
    x_in = x.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x_in).item()

    true_conc = y_true.item()

    # Extract attention
    attn_maps = extract_attention_weights(model, x_in)
    # Last layer, average over heads, drop CLS row/col for patch-only view
    last_attn = attn_maps[-1][0].mean(dim=0).cpu().numpy()
    patch_attn = last_attn[1:, 1:]  # remove CLS token

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1. Spectrum
    axes[0].plot(x.numpy(), color='steelblue', lw=1.5)
    axes[0].set_title('Normalised Transmission T(E)')
    axes[0].set_xlabel('Energy Index')
    axes[0].set_ylabel('T / T_pristine')
    axes[0].grid(True, alpha=0.3)

    # 2. Attention heatmap
    im = axes[1].imshow(patch_attn, cmap='inferno', aspect='equal')
    axes[1].set_title('Attention: Energy Band Correlations')
    axes[1].set_xlabel('Key Patch')
    axes[1].set_ylabel('Query Patch')
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    # 3. CLS attention weights (which patches the CLS attends to)
    cls_attn = last_attn[0, 1:]  # CLS row, patch columns
    axes[2].bar(range(len(cls_attn)), cls_attn, color='coral')
    axes[2].set_title(f'[CLS] Attention | True: {true_conc:.0f} | Pred: {pred:.1f}')
    axes[2].set_xlabel('Patch Index (Energy Band)')
    axes[2].set_ylabel('Attention Weight')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_losses(train_losses, val_losses):
    """Plot training curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train', color='steelblue')
    plt.plot(val_losses, label='Validation', color='coral')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Patched Transformer v2 — Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ======================================================================
# 11. MISFIT MODULE BUILDER (convenience)
# ======================================================================

def build_misfit_module(data_dir, pristine, conc_range, spectrum_length=200,
                        device=None, n_sample_configs=100):
    """Build CurvatureMisfit from raw data files."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pristine_clip = pristine[:spectrum_length].astype(np.float32)
    ref_list = []

    for con in conc_range:
        acc = []
        for cfg in range(n_sample_configs):
            fpath = Path(data_dir) / f"7_agnr_conc{int(con)}_cfg{cfg}.npy"
            if fpath.exists():
                spec = np.load(str(fpath)).astype(np.float32)[:spectrum_length]
                spec = np.clip(spec, 0, pristine_clip)
                acc.append(spec)
        if not acc:
            raise FileNotFoundError(f"No data for concentration {con} in {data_dir}")
        avg_spec = np.mean(acc, axis=0)
        ref_normalised = avg_spec / (pristine_clip + 1e-8)
        ref_list.append(ref_normalised)

    ref_spectra = torch.tensor(np.array(ref_list), dtype=torch.float32)
    concentrations = torch.tensor(conc_range, dtype=torch.float32)
    pristine_tensor = torch.tensor(pristine_clip, dtype=torch.float32)

    return CurvatureMisfit(ref_spectra, concentrations, pristine_tensor).to(device)


# ======================================================================
# 12. MAIN
# ======================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Patched Transformer v2 with CurvatureMisfit"
    )
    parser.add_argument("--epochs",        type=int,   default=200)
    parser.add_argument("--batch-size",    type=int,   default=256)
    parser.add_argument("--max-samples",   type=int,   default=50000,
                        help="Max training samples (0=all). Subsamples for speed.")
    parser.add_argument("--lr",            type=float, default=5e-4)
    parser.add_argument("--weight-decay",  type=float, default=1e-4)
    parser.add_argument("--misfit-weight", type=float, default=0.001)
    parser.add_argument("--val-split",     type=float, default=0.2)
    parser.add_argument("--patience",      type=int,   default=30)
    parser.add_argument("--warmup",        type=int,   default=10)
    parser.add_argument("--spectrum-len",  type=int,   default=200)
    parser.add_argument("--patch-size",    type=int,   default=10)
    parser.add_argument("--embed-dim",     type=int,   default=64)
    parser.add_argument("--depth",         type=int,   default=3)
    parser.add_argument("--num-heads",     type=int,   default=4)
    parser.add_argument("--grad-clip",     type=float, default=1.0)
    parser.add_argument("--device",        type=str,   default=None)
    parser.add_argument("--save-path",     type=str,   default="patched_transformer_v2.pt")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    data_dir = project_root / "data" / "raw" / "transmission_results"
    manifest = script_dir / "manifest_agnr.csv"
    pristine_path = data_dir / "pristine.npy"

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    pristine = np.load(str(pristine_path))
    raw_dataset = NormalizedTransmissionsDataset(
        str(manifest), str(data_dir), pristine, args.spectrum_len,
    )
    print("Pre-loading dataset into RAM (one-time cost)...")
    max_s = args.max_samples if args.max_samples > 0 else None
    dataset = cache_dataset(raw_dataset, max_samples=max_s)

    conc_range = np.arange(1, 50, 2)
    print("Building CurvatureMisfit module...")
    misfit_module = build_misfit_module(
        data_dir, pristine, conc_range, args.spectrum_len, device,
    )

    print("Starting training...")
    model, train_losses, val_losses = train_patched_transformer(
        dataset=dataset,
        misfit_module=misfit_module,
        num_epochs=args.epochs,
        val_split=args.val_split,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        misfit_weight=args.misfit_weight,
        patience=args.patience,
        warmup_epochs=args.warmup,
        input_length=args.spectrum_len,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        grad_clip=args.grad_clip,
        device_str=str(device),
    )

    save_path = Path(args.save_path)
    torch.save({
        "model_state_dict": model.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "args": vars(args),
    }, save_path)
    print(f"✓ Model saved to: {save_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # CLI args present → run training
        main()
    else:
        # No args → smoke test
        print("Patched Transformer v2 — smoke test")
        model = PatchedTransformerV2()
        x = torch.randn(4, 200)
        out = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Input:  {x.shape}")
        print(f"  Output: {out.shape}")
        print(f"  Params: {params:,}")

        attn = extract_attention_weights(model, x[:1])
        print(f"  Attention layers: {len(attn)}")
        print(f"  Last attn shape:  {attn[-1].shape}")

        print("\nTo train, run:  python patched_transformer_v2.py --epochs 200")
