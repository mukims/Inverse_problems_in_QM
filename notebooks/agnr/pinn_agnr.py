"""
pinn_agnr.py — Improved Physics-Informed CNN for AGNR impurity concentration prediction.

Improvements over the original misfit_agnr.ipynb:
  1. DifferentiableMisfit: physics regularizer with proper gradient flow
  2. ResBlock1D + BatchNorm: deeper, more stable feature extraction
  3. Input normalization: T/T_pristine ratio for easier learning
  4. Data augmentation: multiplicative noise during training
  5. Better training: AdamW + cosine LR + early stopping
"""

import os
import copy
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# ======================================================================
# 1. DIFFERENTIABLE MISFIT — the key fix
# ======================================================================

class DifferentiableMisfit(nn.Module):
    """
    Physics-informed regularizer that is fully differentiable.

    Given the model's predicted concentration and the input spectrum,
    soft-selects the matching reference spectrum (configuration-averaged)
    and penalises the discrepancy.

    Gradients flow through `pred_conc`, so the model learns to predict
    concentrations that are physically consistent with the input.
    """

    def __init__(self, ref_spectra: torch.Tensor, concentrations: torch.Tensor,
                 temperature: float = 2.0):
        """
        Args:
            ref_spectra:    [num_conc, spectrum_length] averaged reference spectra
            concentrations: [num_conc] concentration values (e.g. [1,3,5,...,49])
            temperature:    controls softness of concentration selection
                            (lower = sharper, higher = smoother)
        """
        super().__init__()
        self.register_buffer('ref_spectra', ref_spectra)        # [C, L]
        self.register_buffer('concentrations', concentrations)  # [C]
        self.temperature = temperature

    def forward(self, x: torch.Tensor, pred_conc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:         [B, L] normalised input transmission spectra
            pred_conc: [B, 1] predicted concentrations from the model

        Returns:
            scalar misfit loss (mean over batch)
        """
        # Distance from predicted concentration to each reference concentration
        # pred_conc: [B, 1], concentrations: [C] -> diffs: [B, C]
        diffs = (pred_conc - self.concentrations.unsqueeze(0)) ** 2

        # Gaussian-kernel soft weights (differentiable soft-argmin)
        weights = F.softmax(-diffs / self.temperature, dim=1)   # [B, C]

        # Soft-select reference spectrum: weighted combination
        # weights: [B, C], ref_spectra: [C, L] -> ref_selected: [B, L]
        ref_selected = torch.matmul(weights, self.ref_spectra)  # [B, L]

        # MSE between input and physics-predicted reference
        misfit = ((x - ref_selected) ** 2).mean()

        return misfit


def build_misfit_module(ca_fn, pristine, conc_range=None, spectrum_length=200, device='cpu'):
    """
    Build a DifferentiableMisfit module from the existing `ca()` function.

    Args:
        ca_fn:            the cached `ca(conc)` function from the notebook
        pristine:         1D numpy array of pristine transmission
        conc_range:       array of concentrations (default: arange(1,50,2))
        spectrum_length:  how many energy points to use
        device:           torch device

    Returns:
        DifferentiableMisfit module
    """
    if conc_range is None:
        conc_range = np.arange(1, 50, 2)

    pristine_t = torch.tensor(pristine[:spectrum_length], dtype=torch.float32)

    ref_list = []
    for con in conc_range:
        ref_raw = ca_fn(int(con))[0][:spectrum_length]
        ref_normalised = ref_raw / (pristine[:spectrum_length] + 1e-8)
        ref_list.append(ref_normalised)

    ref_spectra = torch.tensor(np.array(ref_list), dtype=torch.float32)  # [C, L]
    concentrations = torch.tensor(conc_range, dtype=torch.float32)       # [C]

    module = DifferentiableMisfit(ref_spectra, concentrations)
    return module.to(device)


# ======================================================================
# 2. IMPROVED MODEL ARCHITECTURE
# ======================================================================

class ResBlock1D(nn.Module):
    """Residual block with BatchNorm for 1D convolutions."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.block(x), inplace=True)


class ImprovedConductanceCNN(nn.Module):
    """
    Improved 1D CNN for concentration prediction with:
      - Batch normalisation
      - Residual connections
      - Built-in data augmentation (training only)
      - Dropout for regularisation
    """

    def __init__(self, input_length: int = 200, hidden_channels: int = 32,
                 final_channels: int = 64, num_res_blocks: int = 3,
                 dropout: float = 0.2, noise_std: float = 0.02):
        super().__init__()

        self.noise_std = noise_std

        # Stem: project 1-channel input to hidden_channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        # Residual blocks
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResBlock1D(hidden_channels))
            res_blocks.append(nn.MaxPool1d(2))
            res_blocks.append(nn.Dropout(dropout))
        self.res_tower = nn.Sequential(*res_blocks)

        # Projection to final_channels
        self.project = nn.Sequential(
            nn.Conv1d(hidden_channels, final_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(final_channels),
            nn.ReLU(inplace=True),
        )

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(final_channels, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L] normalised transmission spectrum

        Returns:
            [B, 1] predicted concentration
        """
        # Data augmentation: multiplicative noise during training
        if self.training and self.noise_std > 0:
            noise = 1.0 + torch.randn_like(x) * self.noise_std
            x = x * noise

        # Reshape for Conv1d: [B, L] -> [B, 1, L]
        x = x.unsqueeze(1)

        x = self.stem(x)
        x = self.res_tower(x)
        x = self.project(x)
        x = self.pool(x)           # [B, final_channels, 1]
        x = x.squeeze(-1)          # [B, final_channels]

        return self.regressor(x)   # [B, 1]


# ======================================================================
# 3. NORMALISED DATASET
# ======================================================================

class NormalizedTransmissionsDataset(Dataset):
    """
    Dataset that loads transmission spectra and normalises by the pristine
    spectrum: x = clip(T, 0, T_pristine) / T_pristine.

    This gives values in [0, 1], making learning much easier.
    """

    def __init__(self, manifest_file: str, root_dir: str, pristine: np.ndarray,
                 spectrum_length: int = 200):
        self.manifest = pd.read_csv(manifest_file, index_col="id")
        self.root_dir = root_dir
        self.pristine = np.asarray(pristine[:spectrum_length], dtype=np.float32)
        self.spectrum_length = spectrum_length

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        sample = self.manifest.iloc[idx]
        conc = int(sample["concentration"])

        filepath = os.path.join(self.root_dir, sample["filepath"])
        trans = np.load(filepath).astype(np.float32)[:self.spectrum_length]
        trans = np.clip(trans, 0, self.pristine)

        # Normalise by pristine
        x = trans / (self.pristine + 1e-8)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor([conc], dtype=torch.float32)

        return x, y


class NormalizedTestDataset(Dataset):
    """
    Test dataset that normalises on-the-fly, for data generated
    in-memory via device_transmission().
    """

    def __init__(self, data_dict: dict, pristine: np.ndarray,
                 spectrum_length: int = 200):
        self.pristine = np.asarray(pristine[:spectrum_length], dtype=np.float32)
        self.spectrum_length = spectrum_length

        self.index_map = []
        self.data_dict = data_dict
        for conc in sorted(data_dict.keys()):
            for config_idx in range(len(data_dict[conc])):
                self.index_map.append((conc, config_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        conc, config_idx = self.index_map[idx]

        trans = np.asarray(
            self.data_dict[conc][config_idx], dtype=np.float32
        )[:self.spectrum_length]
        trans = np.clip(trans, 0, self.pristine)

        x = trans / (self.pristine + 1e-8)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor([conc], dtype=torch.float32)

        return x, y


# ======================================================================
# 4. TRAINING
# ======================================================================

def train_pinn(
    dataset: Dataset,
    misfit_module: DifferentiableMisfit,
    num_epochs: int = 200,
    val_split: float = 0.2,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    misfit_weight: float = 0.1,
    patience: int = 20,
    input_length: int = 200,
    device_str: str = None,
):
    """
    Train the improved PINN model.

    Args:
        dataset:        NormalizedTransmissionsDataset
        misfit_module:  DifferentiableMisfit (prebuilt)
        num_epochs:     maximum epochs
        val_split:      fraction for validation
        batch_size:     training batch size
        lr:             initial learning rate
        weight_decay:   AdamW weight decay
        misfit_weight:  λ coefficient for misfit term in loss
        patience:       early stopping patience (epochs without val improvement)
        input_length:   spectrum length
        device_str:     'cuda' / 'cpu' / None (auto)

    Returns:
        (model, train_losses, val_losses)
    """
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Training on: {device}")

    # Model
    model = ImprovedConductanceCNN(input_length=input_length).to(device)
    misfit_module = misfit_module.to(device)

    # Optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Data split
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size,
                            num_workers=0, pin_memory=True)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_weights = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):

        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        running_mse = 0.0
        running_misfit = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)                        # [B, 1]
            yb_flat = yb.squeeze(-1)
            preds_flat = preds.squeeze(-1)

            mse_loss = F.mse_loss(preds_flat, yb_flat)
            misfit_loss = misfit_module(xb, preds)    # differentiable!
            loss = mse_loss + misfit_weight * misfit_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_mse += mse_loss.item()
            running_misfit += misfit_loss.item()

        scheduler.step()

        avg_train = running_loss / len(train_loader)
        avg_mse_train = running_mse / len(train_loader)
        avg_misfit_train = running_misfit / len(train_loader)
        train_losses.append(avg_train)

        # ---- VALIDATE ----
        model.eval()
        running_val = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                preds = model(xb)
                yb_flat = yb.squeeze(-1)
                preds_flat = preds.squeeze(-1)

                mse_loss = F.mse_loss(preds_flat, yb_flat)
                misfit_loss = misfit_module(xb, preds)
                loss = mse_loss + misfit_weight * misfit_loss

                running_val += loss.item()

        avg_val = running_val / len(val_loader)
        val_losses.append(avg_val)

        # ---- EARLY STOPPING ----
        current_lr = optimizer.param_groups[0]['lr']
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            marker = "* Best"
        else:
            epochs_no_improve += 1
            marker = ""

        if (epoch + 1) % 5 == 0 or marker:
            print(f"Epoch {epoch+1:03d} | "
                  f"Train: {avg_train:.4f} (MSE:{avg_mse_train:.3f} Mis:{avg_misfit_train:.4f}) | "
                  f"Val: {avg_val:.4f} | LR: {current_lr:.2e} {marker}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Load best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"\nLoaded best model (Val Loss: {best_val_loss:.4f})")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label="Train Loss", alpha=0.8)
    ax1.plot(val_losses, label="Val Loss", alpha=0.8)
    best_idx = val_losses.index(best_val_loss)
    ax1.axvline(best_idx, color='green', linestyle='--', alpha=0.5, label=f'Best @ epoch {best_idx+1}')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE + λ·misfit)")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_losses, label="Train Loss", alpha=0.8)
    ax2.plot(val_losses, label="Val Loss", alpha=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss (log scale)")
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return model, train_losses, val_losses


# ======================================================================
# 5. TESTING
# ======================================================================

def test_pinn(
    model: nn.Module,
    test_dataset: Dataset,
    misfit_module: DifferentiableMisfit = None,
    misfit_weight: float = 0.1,
    batch_size: int = 64,
    device_str: str = None,
):
    """
    Test the model and return per-concentration summary.

    Returns:
        (predictions, labels, results_df)
    """
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    model = model.to(device)
    model.eval()

    if misfit_module is not None:
        misfit_module = misfit_module.to(device)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)
            yb_flat = yb.squeeze(-1)
            preds_flat = preds.squeeze(-1)

            loss = F.mse_loss(preds_flat, yb_flat)
            if misfit_module is not None:
                loss = loss + misfit_weight * misfit_module(xb, preds)

            total_loss += loss.item()
            all_preds.append(preds_flat.cpu())
            all_labels.append(yb_flat.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    avg_loss = total_loss / len(test_loader)
    print(f"\nTest Loss: {avg_loss:.4f}")

    # Build results DataFrame
    results = pd.DataFrame({
        'labels': all_labels,
        'predictions': all_preds,
        'error': np.abs(all_preds - all_labels),
    })

    summary = results.groupby('labels').agg(
        pred_mean=('predictions', 'mean'),
        pred_std=('predictions', 'std'),
        mae=('error', 'mean'),
        max_error=('error', 'max'),
        count=('error', 'count'),
    ).round(3)

    print("\nPer-concentration summary:")
    print(summary.to_string())

    return all_preds, all_labels, results
