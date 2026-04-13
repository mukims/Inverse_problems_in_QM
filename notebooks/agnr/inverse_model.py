import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.spatial.distance import pdist, squareform

try:
    import agnr  # Needs to run from notebooks/agnr
except ImportError:
    pass

# =============================================================================
# 1. Dataset: Generates [T(E)] -> [10x10 Impurity Distance Matrix] on the fly
# =============================================================================
class InverseGNRDataset(Dataset):
    def __init__(self, manifest_file, root_dir, pristine, max_conc=10, width=7, spectrum_length=200):
        self.root_dir = root_dir
        self.pristine = pristine[:spectrum_length]
        self.spectrum_length = spectrum_length
        self.max_conc = max_conc
        self.width = width
        
        # Load and filter manifest
        df = pd.read_csv(manifest_file)
        self.df = df[df['concentration'] <= max_conc].reset_index(drop=True)
        print(f"Loaded InverseGNRDataset with {len(self.df)} samples (conc <= {max_conc})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        conc = int(row['concentration'])
        config_id = int(row['config_id'])
        filename = row['filepath']

        # Load Transmission Spectrum T(E)
        file_path = os.path.join(self.root_dir, filename)
        try:
            raw_t = np.load(file_path)[:self.spectrum_length]
        except Exception:
            raw_t = np.zeros(self.spectrum_length)

        # Normalise T(E)
        t_clipped = np.clip(raw_t, 0, self.pristine)
        norm_t = np.divide(t_clipped, self.pristine, out=np.zeros_like(t_clipped), where=self.pristine!=0)
        x = torch.tensor(norm_t, dtype=torch.float32)

        # Build impurity distance matrix (max_conc x max_conc)
        # Diagonal[i,i] = site position of impurity i within the unit cell (0 to 2*width-1)
        # Off-diagonal[i,j] = Euclidean distance between impurity i and j
        # Zero-padded when conc < max_conc
        mat = np.zeros((self.max_conc, self.max_conc), dtype=np.float32)
        if conc > 0:
            imps = agnr.chosen_for_config(n=conc, width=self.width, config=config_id)
            # imps[:,0] = unit cell index [0..99]
            # imps[:,1] = site index [0..2*width-1]

            # Off-diagonal: pairwise Euclidean distances
            if conc > 1:
                dists = squareform(pdist(imps.astype(np.float64), metric='sqeuclidean'))
                mat[:conc, :conc] = dists.astype(np.float32)

            # Diagonal: site position within the unit cell
            for i in range(conc):
                mat[i, i] = float(imps[i, 1])

        y = torch.tensor(mat, dtype=torch.float32)  # [max_conc, max_conc]

        return x, y

def get_reference_spectra(dataset):
    """ Pre-computes the average T(E) for each concentration to use in Misfit Loss """
    print("Pre-computing reference spectra for Misfit Loss...")
    ref_spectra = []
    concs = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # We just average the first ~500 items per concentration to save time
    for c in range(1, dataset.max_conc + 1):
        idx = dataset.df.index[dataset.df['concentration'] == c].tolist()[:500]
        if not idx: continue
        
        spectra = [dataset[i][0] for i in idx]
        avg_spectrum = torch.stack(spectra).mean(dim=0)
        ref_spectra.append(avg_spectrum)
        concs.append(c)
        
    return torch.stack(ref_spectra).to(device), torch.tensor(concs, dtype=torch.float32).to(device)

# =============================================================================
# 2. Physics-Informed & Focal Loss Functions
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0):
        """
        alpha weights the minority class (1). 
        gamma severely discounts loss for easily predicted '0' backgrounds.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt is the probability of the *correct* class
        
        # Apply alpha weighting: alpha for targets=1, (1-alpha) for targets=0
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class MisfitLoss(nn.Module):
    def __init__(self, reference_spectra, conc_range, max_conc=10, tau=2.0, scale=5.0):
        """
        Adapted for distance-matrix targets.
        Estimates concentration from the predicted matrix by computing
        row-wise L2 norms: rows with impurities have large norms,
        empty (zero-padded) rows have norms near 0.
        A sigmoid soft-threshold converts norms to [0,1] probabilities,
        and their sum gives a differentiable concentration estimate.
        """
        super().__init__()
        self.register_buffer('reference_spectra', reference_spectra)
        self.register_buffer('conc_range', conc_range)
        self.max_conc = max_conc
        self.tau = tau
        self.scale = scale   # sharpness of the soft-threshold
        
    def forward(self, x_input, predicted_output):
        B = predicted_output.size(0)
        pred_mat = predicted_output.view(B, self.max_conc, self.max_conc)
        
        # Row-wise L2 norm → large for impurity rows, ~0 for empty rows
        row_norms = torch.norm(pred_mat, dim=2)        # [B, max_conc]
        
        # Soft count: sigmoid maps large norms → ~1, small norms → ~0
        c_pred = torch.sigmoid(self.scale * (row_norms - 0.5)).sum(dim=1, keepdim=True)  # [B, 1]
        
        # Soft index into reference spectra
        dists = (c_pred - self.conc_range.unsqueeze(0)) ** 2
        weights = F.softmax(-dists / self.tau, dim=1)   # [B, num_concs]
        
        # Expected spectrum given predicted global concentration
        R_hat = torch.matmul(weights, self.reference_spectra)  # [B, 200]
        
        # Penalise if input spectrum doesn't match expected R_hat
        x_input_squeeze = x_input.squeeze(1) if x_input.dim() == 3 else x_input
        return F.mse_loss(x_input_squeeze, R_hat)

# =============================================================================
# 3. Model: Spatial 1D Encoder-Decoder
# =============================================================================
class LayerNorm1D(nn.Module):
    """Applies LayerNorm over the channel dimension for 1D convolution outputs."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)

class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = LayerNorm1D(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = LayerNorm1D(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class InverseModel(nn.Module):
    def __init__(self):
        super().__init__()
        # --- ENCODER --- (Compresses T(E) [200] -> Latent [12])
        self.enc_conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)  # -> [32, 100]
        self.enc_bn1 = LayerNorm1D(32)
        
        self.enc_res1 = ResBlock1D(32)
        self.enc_pool1 = nn.MaxPool1d(2) # -> [32, 50]
        
        self.enc_res2 = ResBlock1D(32)
        self.enc_pool2 = nn.MaxPool1d(2) # -> [32, 25]
        
        self.enc_res3 = ResBlock1D(32)
        self.enc_pool3 = nn.MaxPool1d(2) # -> [32, 12]
        
        # --- BOTTLENECK ---
        # 32 channels * 12 length = 384 exactly. 
        self.bottleneck = nn.Sequential(
            # Input: [Batch, 32, 12]
            nn.Conv1d(32, 64, kernel_size=3, padding=1), # Expands channels: [Batch, 64, 12]
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 32, kernel_size=3, padding=1), # Compresses channels: [Batch, 32, 12]
            nn.ReLU()
            )
        
        # --- DECODER --- (Expands Latent [12] -> Spatial Map [100])
        self.dec_conv1 = nn.ConvTranspose1d(32, 32, kernel_size=5, stride=2, padding=1, output_padding=0) # -> [32, 25]
        self.dec_bn1 = LayerNorm1D(32)
        
        self.dec_conv2 = nn.ConvTranspose1d(32, 32, kernel_size=4, stride=2, padding=1, output_padding=0) # -> [32, 50]
        self.dec_bn2 = LayerNorm1D(32)
        
        self.dec_conv3 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=0) # -> [16, 100]
        self.dec_bn3 = LayerNorm1D(16)
        
        # POSITIONAL ENCODINGS
        # Adds spatial awareness to the 100 unit cells
        self.pos_emb = nn.Parameter(torch.randn(1, 16, 100) * 0.01)
        
        # Final map down to 1 channel probability map
        self.dec_final = nn.Conv1d(16, 1, kernel_size=3, padding=1) # -> [1, 100]

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Encoder
        x = F.relu(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_pool1(self.enc_res1(x))
        x = self.enc_pool2(self.enc_res2(x))
        x = self.enc_pool3(self.enc_res3(x))
        
        # Bottleneck: [B, 32, 12] -> [B, 32, 12]
        x = self.bottleneck(x)
        
        # Decoder
        x = F.relu(self.dec_bn1(self.dec_conv1(x)))
        x = F.relu(self.dec_bn2(self.dec_conv2(x)))
        x = F.relu(self.dec_bn3(self.dec_conv3(x)))
        
        # Inject Spatial Awareness
        x = x + self.pos_emb
        
        # Output [B, 100] Raw Logits
        return self.dec_final(x).squeeze(1)

# =============================================================================
# 4. Training Loop: MSE Loss for Distance Matrix Regression
# =============================================================================
def train_inverse_model(dataset, num_epochs=100, batch_size=64, lr=7e-4, misfit_weight=0.1):
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = InverseModel().to(device)
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
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets_flat = targets.view(targets.size(0), -1)  # [B, max_conc*max_conc]
            
            optimizer.zero_grad()
            preds = model(inputs)
            
            mse_loss = F.mse_loss(preds, targets_flat)
            m_loss = misfit_criterion(inputs, preds)
            loss = mse_loss + (misfit_weight * m_loss)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
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
        
        print(f"Epoch {epoch+1:03d}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            os.makedirs('../../models/trained', exist_ok=True)
            torch.save(model.state_dict(), '../../models/trained/inverse_model.pth')
            
    return model, train_losses, val_losses

# =============================================================================
# 5. Visualization Helper
# =============================================================================
def visualize_prediction(model, dataset, idx=0, max_conc=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    x, y_true = dataset[idx]
    with torch.no_grad():
        x_in = x.unsqueeze(0).to(device)
        preds = model(x_in).squeeze().cpu().numpy()
        y_pred_mat = preds.reshape(max_conc, max_conc)
        
    y_true_mat = y_true.numpy()  # [max_conc, max_conc]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
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
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("Inverse Model with Distance Matrix regression ready.")
