"""
pinn_agnr_curvature.py — Physics-Informed CNN for AGNR impurity concentration prediction
with Curvature-Based Misfit regularizer.
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
from torch.utils.data import Dataset, DataLoader, random_split


# ======================================================================
# 1. CURVATURE MISFIT — Physics regularizer
# ======================================================================

class CurvatureMisfit(nn.Module):
    """
    Physics-informed regularizer that uses the curvature of the misfit
    curve as a confidence weight.
    
    Given the model's predicted concentration and the input spectrum,
    it calculates the physics misfit curve across all reference concentrations,
    finds the absolute minimum, and computes the curvature at that minimum.
    This curvature acts as a weight: if it is high, the physics signal is strong
    and the model is heavily penalized for deviating from the physical prediction.
    If it is low, the physics signal is ambiguous and the penalty is smaller.
    """

    def __init__(self, ref_spectra: torch.Tensor, concentrations: torch.Tensor, pristine: torch.Tensor,
                 temperature: float = 2.0, spectrum_start: int = 20, spectrum_end: int = 150):
        super().__init__()
        self.register_buffer('ref_spectra', ref_spectra)        # [C, L]
        self.register_buffer('concentrations', concentrations)  # [C]
        self.register_buffer('pristine', pristine)              # [L]
        self.temperature = temperature
        self.spectrum_start = spectrum_start
        self.spectrum_end = spectrum_end
        
        # Determine dx from concentrations (assumed uniform spacing)
        spacing = (concentrations[1] - concentrations[0]).item() if len(concentrations) > 1 else 2.0
        self.dx = float(spacing)

    def forward(self, x: torch.Tensor, pred_conc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:         [B, L] normalised input transmission spectra
            pred_conc: [B, 1] predicted concentrations from the model

        Returns:
            scalar representing the curvature-weighted physical misfit.
        """
        # Crop the spectra to the region of interest [20:150]
        x_crop = x[:, self.spectrum_start:self.spectrum_end]
        ref_crop = self.ref_spectra[:, self.spectrum_start:self.spectrum_end]
        pristine_crop = self.pristine[self.spectrum_start:self.spectrum_end]
        
        # Un-normalize back to original transmission values
        x_unnorm = x_crop * pristine_crop.unsqueeze(0)
        ref_unnorm = ref_crop * pristine_crop.unsqueeze(0)
        
        # Explicitly clip the input spectra between 0 and pristine values
        # (This handles any values that might exceed pristine due to data augmentation noise)
        x_unnorm = torch.clamp(x_unnorm, min=0.0)
        x_unnorm = torch.min(x_unnorm, pristine_crop.unsqueeze(0))
        
        # 1. Calculate misfits for all concentrations
        # diff shape: [B, C, L_crop]
        diff = x_unnorm.unsqueeze(1) - ref_unnorm.unsqueeze(0)
        
        # Sum squared differences and divide by 150 (matching the numpy code)
        mis_1 = torch.sum(diff ** 2, dim=2) / 150.0  # [B, C]
        
        # 2. Calculate derivatives w.r.t concentration using torch.gradient
        # dy, d2y: [B, C]
        dy = torch.gradient(mis_1, spacing=self.dx, dim=1, edge_order=1)[0]
        d2y = torch.gradient(dy, spacing=self.dx, dim=1, edge_order=1)[0]
        
        # 3. Calculate Curvature
        # dx is constant, so d2x = 0
        # curvature = |dx * d2y| / (dx**2 + dy**2)**1.5
        curvature_array = torch.abs(self.dx * d2y) / ((self.dx**2 + dy**2)**1.5)  # [B, C]
        
        # 4. Extract curvature at the minima for each sample
        min_idx = torch.argmin(mis_1, dim=1, keepdim=True) # [B, 1]
        curvature_at_minima = torch.gather(curvature_array, 1, min_idx).squeeze(1) # [B]
        
        # 5. Calculate physical misfit at the predicted concentration
        # Distance from predicted concentration to each reference concentration
        diffs = (pred_conc - self.concentrations.unsqueeze(0)) ** 2
        
        # Gaussian-kernel soft weights (differentiable soft-argmin)
        weights = F.softmax(-diffs / self.temperature, dim=1)   # [B, C]
        ref_selected = torch.matmul(weights, ref_unnorm)  # [B, L_crop]
        
        physical_misfit = ((x_unnorm - ref_selected) ** 2).mean(dim=1) # [B]
        
        # 6. Weight the physical misfit inversely by the curvature
        # We detach curvature so it acts purely as a constant weight per sample
        # If curvature is high, the penalty is small.
        # If curvature is low, the penalty is high.
        weighted_misfit = (1.0 / (curvature_at_minima.detach() + 1e-6)) * physical_misfit # [B]
        
        return weighted_misfit.mean()


# ======================================================================
# 2. MODEL ARCHITECTURE
# ======================================================================

class ConductanceMLP(nn.Module):
    """Multi-Layer Perceptron for concentration prediction."""
    def __init__(self, input_length: int = 200, hidden_dims: list = None,
                 dropout: float = 0.2, noise_std: float = 0.02):
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
        # Data augmentation
        if self.training and self.noise_std > 0:
            noise = 1.0 + torch.randn_like(x) * self.noise_std
            x = x * noise

        x = self.mlp(x)
        return self.regressor(x)


# ======================================================================
# 3. DATASET
# ======================================================================

class NormalizedTransmissionsDataset(Dataset):
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

        x = trans / (self.pristine + 1e-8)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor([conc], dtype=torch.float32)

        return x, y


# ======================================================================
# 4. TRAINING
# ======================================================================

def train_pinn(
    dataset: Dataset,
    misfit_module: CurvatureMisfit,
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
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Training on: {device}")

    model = ConductanceMLP(input_length=input_length).to(device)
    misfit_module = misfit_module.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=0)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size, num_workers=0)

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

            preds = model(xb)                        
            yb_flat = yb.squeeze(-1)
            preds_flat = preds.squeeze(-1)

            mse_loss = F.mse_loss(preds_flat, yb_flat)
            # misfit_loss here is NEGATIVE curvature
            misfit_loss = misfit_module(xb, preds)
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
                  f"Train: {avg_train:.4f} (MSE:{avg_mse_train:.3f} Curv/Mis:{avg_misfit_train:.4f}) | "
                  f"Val: {avg_val:.4f} | LR: {current_lr:.2e} {marker}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, train_losses, val_losses


# ======================================================================
# 5. MAIN
# ======================================================================

def _build_misfit_from_files(data_dir, pristine, conc_range, spectrum_length, device,
                             n_sample_configs=100):
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
        if len(acc) == 0:
            raise FileNotFoundError(f"No data files found for concentration {con} in {data_dir}")
        avg_spec = np.mean(acc, axis=0)
        ref_normalised = avg_spec / (pristine_clip + 1e-8)
        ref_list.append(ref_normalised)

    ref_spectra = torch.tensor(np.array(ref_list), dtype=torch.float32)
    concentrations = torch.tensor(conc_range, dtype=torch.float32)
    pristine_tensor = torch.tensor(pristine_clip, dtype=torch.float32)
    module = CurvatureMisfit(ref_spectra, concentrations, pristine_tensor)
    return module.to(device)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train PINN with Curvature Misfit")
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--misfit-weight",type=float, default=0.001, help="λ for curvature term")
    parser.add_argument("--val-split",    type=float, default=0.2)
    parser.add_argument("--patience",     type=int,   default=50)
    parser.add_argument("--spectrum-len", type=int,   default=200)
    parser.add_argument("--device",       type=str,   default=None)
    parser.add_argument("--save-path",    type=str,   default="pinn_agnr_curvature.pt")
    args = parser.parse_args()

    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]
    data_dir     = project_root / "data" / "raw" / "transmission_results"
    manifest     = script_dir.parent / "manifest_agnr.csv"
    pristine_path = data_dir / "pristine.npy"

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    pristine = np.load(str(pristine_path))
    dataset = NormalizedTransmissionsDataset(str(manifest), str(data_dir), pristine, args.spectrum_len)

    conc_range = np.arange(1, 50, 2)
    print("Building CurvatureMisfit module...")
    misfit_module = _build_misfit_from_files(data_dir, pristine, conc_range, args.spectrum_len, device)
    
    print("Starting training...")
    model, train_losses, val_losses = train_pinn(
        dataset=dataset,
        misfit_module=misfit_module,
        num_epochs=args.epochs,
        val_split=args.val_split,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        misfit_weight=args.misfit_weight,
        patience=args.patience,
        input_length=args.spectrum_len,
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
    main()
