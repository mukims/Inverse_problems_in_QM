"""
Sweep over different misfit_weight values to find the best trade-off
between Focal Loss and the physics-informed Misfit Loss constraint.

Usage:
    python sweep_misfit_weight.py

Results (models + loss curves) are saved to ../../models/trained/sweep/
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from inverse_model import InverseGNRDataset, InverseModel, FocalLoss, MisfitLoss, get_reference_spectra
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Configuration ──────────────────────────────────────────────────────────────
MISFIT_WEIGHTS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
NUM_EPOCHS     = 500
BATCH_SIZE     = 16
LR             = 7e-4
MAX_CONC       = 10
SPECTRUM_LEN   = 200
DATA_DIR       = os.path.expanduser('~/transmission_results')
OUTPUT_DIR     = '../../models/trained/sweep'
# ───────────────────────────────────────────────────────────────────────────────


def train_with_misfit_weight(dataset, misfit_weight, ref_spectra, concs,
                             num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    """Train the inverse model with a specific misfit_weight and return loss curves."""
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size

    # Use a fixed random seed per weight so the split is always the same
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"  Training with misfit_weight = {misfit_weight}  (device: {device})")
    print(f"{'='*70}")

    model     = InverseModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    focal_criterion  = FocalLoss(alpha=0.95, gamma=2.0)
    misfit_criterion = MisfitLoss(ref_spectra, concs, tau=2.0).to(device)

    best_val_loss    = float('inf')
    train_losses     = []
    val_losses       = []

    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits     = model(inputs)
            f_loss     = focal_criterion(logits, targets)
            m_loss     = misfit_criterion(inputs, logits)
            total_loss = f_loss + misfit_weight * m_loss
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        # ── Validate ──
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits     = model(inputs)
                f_loss     = focal_criterion(logits, targets)
                m_loss     = misfit_criterion(inputs, logits)
                total_loss = f_loss + misfit_weight * m_loss
                val_running += total_loss.item() * inputs.size(0)

        epoch_val_loss = val_running / len(val_dataset)
        val_losses.append(epoch_val_loss)
        scheduler.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:03d}/{num_epochs} | "
                  f"Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(OUTPUT_DIR, f'inverse_model_mw{misfit_weight}.pth'))

    return train_losses, val_losses, best_val_loss


def plot_comparison(results):
    """Plot training & validation loss curves for all misfit_weight values."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    cmap = plt.cm.viridis
    n = len(results)

    # ── Training loss ──
    ax = axes[0]
    for i, (mw, data) in enumerate(results.items()):
        color = cmap(i / max(n - 1, 1))
        ax.plot(data['train'], label=f'mw={mw}', color=color, alpha=0.8)
    ax.set_title('Training Loss vs. Epoch', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.legend()
    ax.grid(True)

    # ── Validation loss ──
    ax = axes[1]
    for i, (mw, data) in enumerate(results.items()):
        color = cmap(i / max(n - 1, 1))
        ax.plot(data['val'], label=f'mw={mw}', color=color, alpha=0.8)
    ax.set_title('Validation Loss vs. Epoch', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.legend()
    ax.grid(True)

    plt.suptitle('Misfit Weight Sweep: Loss Curves', fontsize=16, y=1.02)
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'misfit_weight_sweep.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved to {os.path.join(OUTPUT_DIR, 'misfit_weight_sweep.png')}")


def plot_best_val_loss(results):
    """Bar chart of best validation loss for each misfit_weight."""
    weights = list(results.keys())
    best_vals = [results[w]['best_val'] for w in weights]

    plt.figure(figsize=(10, 5))
    bars = plt.bar([str(w) for w in weights], best_vals, color='steelblue')
    plt.xlabel('misfit_weight', fontsize=12)
    plt.ylabel('Best Validation Loss', fontsize=12)
    plt.title('Best Validation Loss by misfit_weight', fontsize=14)
    plt.grid(axis='y', alpha=0.5)

    # Label each bar with its value
    for bar, val in zip(bars, best_vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'best_val_loss_by_weight.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {os.path.join(OUTPUT_DIR, 'best_val_loss_by_weight.png')}")


def main():
    # ── Load dataset ──
    try:
        pristine = np.load('../../data/raw/transmission_results/pristine.npy')[:SPECTRUM_LEN]
    except Exception:
        print("Warning: Pristine file not found. Using array of ones for test.")
        pristine = np.ones(SPECTRUM_LEN)

    dataset = InverseGNRDataset(
        manifest_file='manifest_agnr.csv',
        root_dir=DATA_DIR,
        pristine=pristine,
        max_conc=MAX_CONC,
        spectrum_length=SPECTRUM_LEN,
    )

    if len(dataset) == 0:
        print("Dataset is empty. Check your data path and CSV.")
        return

    # Pre-compute reference spectra ONCE (shared across all runs)
    ref_spectra, concs = get_reference_spectra(dataset)

    # ── Run sweep ──
    results = {}   # {misfit_weight: {train: [...], val: [...], best_val: float}}

    for mw in MISFIT_WEIGHTS:
        train_losses, val_losses, best_val = train_with_misfit_weight(
            dataset, mw, ref_spectra, concs
        )
        results[mw] = {
            'train':    train_losses,
            'val':      val_losses,
            'best_val': best_val,
        }

    # ── Save raw results as JSON ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, 'sweep_results.json')
    with open(json_path, 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nRaw results saved to {json_path}")

    # ── Print summary ──
    print(f"\n{'='*70}")
    print(f"  SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"  {'misfit_weight':>14s}  {'Best Val Loss':>14s}")
    print(f"  {'-'*14}  {'-'*14}")
    for mw in MISFIT_WEIGHTS:
        print(f"  {mw:>14.1f}  {results[mw]['best_val']:>14.4f}")
    print()

    # ── Plot comparison ──
    plot_comparison(results)
    plot_best_val_loss(results)


if __name__ == '__main__':
    main()
