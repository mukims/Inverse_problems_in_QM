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

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402 - puts sibling topic folders on sys.path
try:
    import agnr  # Needs to run from notebooks/agnr
except ImportError:
    pass

# =============================================================================
# 1. Dataset: Generates [10x10 Impurity Distance Matrix] -> [T(E)] on the fly
# =============================================================================
class ForwardGNRDataset(Dataset):
    def __init__(self, manifest_file, root_dir, pristine, max_conc=10, width=7, spectrum_length=200):
        self.root_dir = root_dir
        self.pristine = pristine[:spectrum_length]
        self.spectrum_length = spectrum_length
        self.max_conc = max_conc
        self.width = width
        
        # Load and filter manifest
        df = pd.read_csv(manifest_file)
        self.df = df[df['concentration'] <= max_conc].reset_index(drop=True)
        print(f"Loaded ForwardGNRDataset with {len(self.df)} samples (conc <= {max_conc})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        conc = int(row['concentration'])
        config_id = int(row['config_id'])
        filename = row['filepath']

        # Load Transmission Spectrum T(E) (Target)
        file_path = os.path.join(self.root_dir, filename)
        try:
            raw_t = np.load(file_path)[:self.spectrum_length]
        except Exception:
            raw_t = np.zeros(self.spectrum_length)

        # Normalise T(E) between 0 and 1
        t_clipped = np.clip(raw_t, 0, self.pristine)
        norm_t = np.divide(t_clipped, self.pristine, out=np.zeros_like(t_clipped), where=self.pristine!=0)
        y = torch.tensor(norm_t, dtype=torch.float32)

        # Build impurity distance matrix (Input)
        mat = np.zeros((self.max_conc, self.max_conc), dtype=np.float32)
        if conc > 0:
            imps = agnr.chosen_for_config(n=conc, width=self.width, config=config_id)

            # Off-diagonal: pairwise Euclidean distances
            if conc > 1:
                dists = squareform(pdist(imps.astype(np.float64), metric='sqeuclidean'))
                # Distance scale could be relatively large, let's keep it as is since NN can learn it.
                mat[:conc, :conc] = dists.astype(np.float32)

            # Diagonal: site position within the unit cell
            for i in range(conc):
                mat[i, i] = float(imps[i, 1])

        # Flatten the matrix into a vector
        x = torch.tensor(mat, dtype=torch.float32).view(-1)  # [100]

        return x, y

# =============================================================================
# 2. Model: Simple Mapping (Multilayer Perceptron)
# =============================================================================
class ForwardModel(nn.Module):
    def __init__(self, input_dim=100, output_dim=200):
        super().__init__()
        # --- SIMPLE MLP ---
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, output_dim),
            nn.Sigmoid()  # Sigmoid is used because normalized transmission is mostly in [0, 1]
        )

    def forward(self, x):
        # x shape should be [Batch, 100]
        # output shape will be [Batch, 200]
        return self.net(x)

# =============================================================================
# 3. Training Loop: MSE Loss for Transmission Spectrum Regression
# =============================================================================
def train_forward_model(dataset, num_epochs=100, batch_size=64, lr=1e-3):
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = ForwardModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_loss = float('inf')
    train_losses, val_losses = [], []
    
    # We mainly use standard MSE loss; physical bounds are loosely enforced by Sigmoid.
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            preds = model(inputs)
            
            loss = criterion(preds, targets)
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
                preds = model(inputs)
                
                loss = criterion(preds, targets)
                val_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)
        scheduler.step()
        
        print(f"Epoch {epoch+1:03d}/{num_epochs} | Train MSE: {epoch_train_loss:.4f} | Val MSE: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            os.makedirs('../../../models/trained', exist_ok=True)
            torch.save(model.state_dict(), '../../../models/trained/forward_model.pth')
            
    return model, train_losses, val_losses

# =============================================================================
# 4. Visualization Helper
# =============================================================================
def visualize_forward_prediction(model, dataset, idx=0, max_conc=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # x: dist_matrix [100], y: true_spectrum [200]
    x, y_true = dataset[idx]
    with torch.no_grad():
        x_in = x.unsqueeze(0).to(device)
        preds = model(x_in).squeeze().cpu().numpy()
        
    x_mat = x.numpy().reshape(max_conc, max_conc)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. True Impurity Distance Matrix
    im1 = axes[0].imshow(x_mat, cmap='viridis', aspect='equal')
    axes[0].set_title('Input: True Impurity Distance Matrix')
    axes[0].set_xlabel('Impurity Index')
    axes[0].set_ylabel('Impurity Index')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # 2. Transmission Spectra
    axes[1].plot(y_true.numpy(), color='blue', label='True T(E)', alpha=0.7, linestyle='--')
    axes[1].plot(preds, color='red', label='Predicted T(E)', alpha=0.8)
    axes[1].set_title('Normalized Transmission T(E)')
    axes[1].set_xlabel('Energy Index')
    axes[1].set_ylabel('T / T_pristine')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("Forward MLP Model ready. Setup allows mapping 1D flattened 10x10 matrix to 200 dimensional sequence.")
    
    # Simple tensor test
    try:
        model = ForwardModel()
        dummy_input = torch.randn(2, 100) # Batch size 2, 100 features
        output = model(dummy_input)
        print(f"Test Pass: Input shape {dummy_input.shape} mapped to Output shape {output.shape}")
    except Exception as e:
        print(f"Test Failed: {e}")
