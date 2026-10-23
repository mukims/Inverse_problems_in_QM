"""
1D-Patched Transformer for Inverse Scattering
==============================================

Maps a 200-point transmission spectrum T(E) to a 10×10 impurity distance matrix
using non-overlapping 1D patching + Transformer encoder.

Key ideas:
  - Conv1d with kernel_size == stride == patch_size implements the patch
    projection as a single fused CUDA kernel.
  - The Transformer sees 20 "energy band" tokens instead of 200 raw points,
    reducing the attention matrix from 200×200 to 20×20.
  - extract_attention_weights() lets you visualise which energy bands the
    network couples, verifying it learns real scattering physics.

Reuses InverseGNRDataset and MisfitLoss from inverse_model.py.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Reuse dataset and losses from the existing module
import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402 - puts sibling topic folders on sys.path
from inverse_model import InverseGNRDataset, MisfitLoss, get_reference_spectra


# =============================================================================
# 1. Patch Embedding
# =============================================================================
class PatchEmbedding1D(nn.Module):
    """
    Splits a 1D spectrum into non-overlapping patches, projects each patch
    into a high-dimensional embedding, and adds learned positional encoding.

    The Conv1d with kernel_size == stride == patch_size is mathematically
    equivalent to chopping the sequence into windows and applying a linear
    projection to each — but runs as a single fused CUDA kernel.
    """
    def __init__(self, seq_len=200, patch_size=10, in_channels=1, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size

        # Non-overlapping convolution IS a patched linear projection
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Absolute positional encoding for the N energy bands
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )

    def forward(self, x):
        # x: [B, 1, seq_len]
        x = self.proj(x)          # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)     # [B, num_patches, embed_dim]
        x = x + self.pos_embed    # inject positional information
        return x


# =============================================================================
# 2. Patched Transformer Model
# =============================================================================
class PatchedInverseModel(nn.Module):
    """
    1D-Patched Transformer for inverse scattering: T(E) → distance matrix.

    Architecture:
      1. PatchEmbedding1D   — chops 200-pt spectrum into 20 patches of 10
      2. TransformerEncoder  — global self-attention over energy bands
      3. 1×1 Conv bottleneck — channel mixing / compression
      4. FC decoder          — maps to flattened max_conc × max_conc matrix
    """
    def __init__(self, seq_len=200, patch_size=10, embed_dim=64,
                 num_heads=4, depth=3, max_conc=10):
        super().__init__()
        self.max_conc = max_conc
        self.num_patches = seq_len // patch_size

        # --- 1. PATCH EMBEDDING ---
        self.patch_embed = PatchEmbedding1D(seq_len, patch_size, 1, embed_dim)

        # --- 2. TRANSFORMER ENCODER ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True   # Pre-norm architecture: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )

        # --- 3. BOTTLENECK (1×1 conv channel mixing) ---
        self.bottleneck = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 32, kernel_size=1),
            nn.ReLU()
        )

        # --- 4. DECODER (to flattened distance matrix) ---
        self.flatten = nn.Flatten()
        flattened_dim = 32 * self.num_patches  # e.g. 32 * 20 = 640

        self.fc_dec1 = nn.Linear(flattened_dim, 256)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc_out = nn.Linear(256, max_conc * max_conc)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)        # [B, 1, 200]

        # 1. Patch and embed
        x = self.patch_embed(x)       # [B, num_patches, embed_dim]

        # 2. Global self-attention
        x = self.transformer(x)       # [B, num_patches, embed_dim]

        # 3. Bottleneck (needs channels-first)
        x = x.transpose(1, 2)         # [B, embed_dim, num_patches]
        x = self.bottleneck(x)         # [B, 32, num_patches]

        # 4. Decode to max_conc × max_conc matrix
        x = self.flatten(x)
        x = F.relu(self.fc_bn1(self.fc_dec1(x)))
        x = self.fc_out(x)

        return x


# =============================================================================
# 3. Attention Weight Extraction (Interpretability)
# =============================================================================
def extract_attention_weights(model, x_input):
    """
    Manually runs through the Transformer encoder layers with
    need_weights=True to capture per-head self-attention weight matrices.

    Returns a list of [B, num_heads, num_patches, num_patches] tensors
    (one per encoder layer).

    Usage:
        attn_weights = extract_attention_weights(model, x_sample)
        # Average over heads → [num_patches, num_patches]
        avg_attn = attn_weights[-1][0].mean(dim=0).cpu().numpy()
        plt.imshow(avg_attn, cmap='inferno'); plt.colorbar()
    """
    attn_maps = []

    model.eval()
    with torch.no_grad():
        if x_input.dim() == 2:
            x_input = x_input.unsqueeze(1)

        x = model.patch_embed(x_input)

        # Walk through each encoder layer, calling self-attn explicitly
        for layer in model.transformer.layers:
            # Pre-norm: norm1 → self-attn → residual → norm2 → ffn → residual
            x_normed = layer.norm1(x)
            attn_out, attn_w = layer.self_attn(
                x_normed, x_normed, x_normed,
                need_weights=True, average_attn_weights=False
            )
            x = x + attn_out
            x = x + layer._ff_block(layer.norm2(x))
            attn_maps.append(attn_w)

    return attn_maps  # list of [B, num_heads, num_patches, num_patches]


# =============================================================================
# 4. Training Loop: MSE Loss + Misfit Constraint
# =============================================================================
def train_patched_model(dataset, num_epochs=100, batch_size=64, lr=7e-4,
                        misfit_weight=0.1, patch_size=10, embed_dim=64,
                        num_heads=4, depth=3):
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    model = PatchedInverseModel(
        seq_len=dataset.spectrum_length,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        max_conc=dataset.max_conc
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Misfit Loss (physics constraint via concentration estimation from row norms)
    ref_spectra, concs = get_reference_spectra(dataset)
    misfit_criterion = MisfitLoss(
        ref_spectra, concs, max_conc=dataset.max_conc, tau=2.0
    ).to(device)

    best_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets_flat = targets.view(targets.size(0), -1)  # [B, max_conc²]

            optimizer.zero_grad()
            preds = model(inputs)

            mse_loss = F.mse_loss(preds, targets_flat)
            m_loss = misfit_criterion(inputs, preds)
            loss = mse_loss + (misfit_weight * m_loss)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            #if (i + 1) % 20 == 0:
            #    print(f"   [Epoch {epoch+1}] Batch {i+1}/{len(train_loader)}")

        epoch_train_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets_flat = targets.view(targets.size(0), -1)
                preds = model(inputs)

                mse_loss = F.mse_loss(preds, targets_flat)
                m_loss = misfit_criterion(inputs, preds)
                loss = mse_loss + (misfit_weight * m_loss)

                val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)
        scheduler.step()

        print(f"Epoch {epoch+1:03d}/{num_epochs} | "
              f"Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            os.makedirs('../../../models/trained', exist_ok=True)
            torch.save(model.state_dict(),
                       '../../../models/trained/patched_transformer.pth')

    return model, train_losses, val_losses


# =============================================================================
# 5. Visualization: Prediction + Attention Heatmap
# =============================================================================
def visualize_prediction(model, dataset, idx=0, max_conc=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    x, y_true = dataset[idx]
    x_in = x.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(x_in).squeeze().cpu().numpy()
        y_pred_mat = preds.reshape(max_conc, max_conc)

    y_true_mat = y_true.numpy()

    # Extract attention weights for interpretability
    attn_maps = extract_attention_weights(model, x_in)
    # Average the last layer's attention over heads → [num_patches, num_patches]
    last_attn = attn_maps[-1][0].mean(dim=0).cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(26, 6))

    # 1. Transmission Spectrum
    axes[0].plot(x.numpy(), color='blue')
    axes[0].set_title('Normalized Transmission T(E)')
    axes[0].set_xlabel('Energy Index')
    axes[0].set_ylabel('T / T_pristine')
    axes[0].grid(True)

    # 2. True Impurity Distance Matrix
    im1 = axes[1].imshow(y_true_mat, cmap='viridis', aspect='equal')
    axes[1].set_title('True Distance Matrix')
    axes[1].set_xlabel('Impurity Index')
    axes[1].set_ylabel('Impurity Index')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # 3. Predicted Impurity Distance Matrix
    im2 = axes[2].imshow(y_pred_mat, cmap='viridis', aspect='equal')
    axes[2].set_title('Predicted Distance Matrix')
    axes[2].set_xlabel('Impurity Index')
    axes[2].set_ylabel('Impurity Index')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    # 4. Self-Attention Heatmap (Energy Band Correlations)
    im3 = axes[3].imshow(last_attn, cmap='inferno', aspect='equal')
    axes[3].set_title('Attention: Energy Band Correlations')
    axes[3].set_xlabel('Key Patch (Energy Band)')
    axes[3].set_ylabel('Query Patch (Energy Band)')
    plt.colorbar(im3, ax=axes[3], shrink=0.8)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("Patched Transformer model ready.")
    # Quick smoke test
    model = PatchedInverseModel()
    x = torch.randn(4, 200)
    out = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {params:,}")

    attn = extract_attention_weights(model, x[:1])
    print(f"  Attention layers: {len(attn)}")
    print(f"  Last attn shape:  {attn[-1].shape}")
