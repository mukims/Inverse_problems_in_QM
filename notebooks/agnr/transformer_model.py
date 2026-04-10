import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import agnr  # Needs to run from notebooks/agnr
except ImportError:
    pass

# =============================================================================
# 1. Dataset: Generates [T(E)] -> [Unit Cell Map] on the fly
# =============================================================================
class InverseGNRDataset(Dataset):
    def __init__(self, manifest_file, root_dir, pristine, max_conc=10, spectrum_length=200):
        self.root_dir = root_dir
        self.pristine = pristine[:spectrum_length]
        self.spectrum_length = spectrum_length
        self.max_conc = max_conc
        
        df = pd.read_csv(manifest_file)
        self.df = df[df['concentration'] <= max_conc].reset_index(drop=True)
        print(f"Loaded manifest with {len(self.df)} samples.")
        
        # --- CACHING IN RAM ---
        print("Pre-loading dataset into RAM (this takes a minute but saves hours)...")
        self.x_data = []
        self.y_data = []
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            conc = int(row['concentration'])
            config_id = int(row['config_id'])
            filename = row['filepath']

            # Load T(E)
            file_path = os.path.join(self.root_dir, filename)
            try:
                raw_t = np.load(file_path)[:self.spectrum_length]
            except Exception:
                raw_t = np.zeros(self.spectrum_length)

            # Normalise
            t_clipped = np.clip(raw_t, 0, self.pristine)
            norm_t = np.divide(t_clipped, self.pristine, out=np.zeros_like(t_clipped), where=self.pristine!=0)
            
            # Create M
            imps = agnr.chosen_for_config(n=conc, width=7, config=config_id)
            x_coords = imps[:, 0]
            M = np.zeros(100, dtype=np.float32)
            M[x_coords] = 1.0  
            
            self.x_data.append(torch.tensor(norm_t, dtype=torch.float32))
            self.y_data.append(torch.tensor(M, dtype=torch.float32))
            
        self.x_data = torch.stack(self.x_data)
        self.y_data = torch.stack(self.y_data)
        print("Dataset successfully cached in memory!")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Now getitem is instantaneous!
        return self.x_data[idx], self.y_data[idx]

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
    def __init__(self, reference_spectra, conc_range, tau=2.0):
        super().__init__()
        self.register_buffer('reference_spectra', reference_spectra)
        self.register_buffer('conc_range', conc_range)
        self.tau = tau
        
    def forward(self, x_input, predicted_logits):
        # predicted_M: [B, 100] spatial probability map
        predicted_M = torch.sigmoid(predicted_logits)
        
        # The true physics: The sum of probabilities = total concentration
        c_pred = predicted_M.sum(dim=1, keepdim=True) # [B, 1]
        
        # Soft index into reference spectra
        dists = (c_pred - self.conc_range.unsqueeze(0)) ** 2
        weights = F.softmax(-dists / self.tau, dim=1) # [B, num_concs]
        
        # Expected spectrum given predicted global concentration
        R_hat = torch.matmul(weights, self.reference_spectra) # [B, 200]
        
        # Add loss if original input spectrum doesn't match expected R_hat
        x_input_squeeze = x_input.squeeze(1) if x_input.dim() == 3 else x_input
        return F.mse_loss(x_input_squeeze, R_hat)

# =============================================================================
# 3. Model: Pure Spectrum Transformer (Sequence-to-Sequence)
# =============================================================================
class SpectrumTransformer(nn.Module):
    def __init__(self, spectrum_length=200, num_impurities=100, d_model=128, nhead=4, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.spectrum_length = spectrum_length
        self.num_impurities = num_impurities
        self.d_model = d_model
        
        # 1. Project input 1D spectrum to d_model
        #self.input_proj = nn.Linear(1, d_model)
        self.tokenizer = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=5, padding=2)
        # 2. Positional embeddings for the spectrum
        self.encoder_pos_embed = nn.Parameter(torch.randn(spectrum_length, d_model))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="relu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 4. Decoder Queries (the 100 spatial unit cells)
        # These act as the initial target tokens that cross-attend to the memory
        self.target_queries = nn.Parameter(torch.zeros(num_impurities, d_model))
        self.spatial_pos_embed = nn.Parameter(torch.randn(num_impurities, d_model))
        # 5. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="relu", batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 6. Final Logit Projection
        self.output_proj = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x is expected to be [B, 200]
        if x.dim() == 2:
            x = x.unsqueeze(1) # [B, 200, 1]
        elif x.dim() == 3 and x.size(1) == 1:
            x = x.transpose(1, 2) # [B, 1, 200] -> [B, 200, 1]
            
        B = x.size(0)
        
        # --- Encoder ---
        src = self.tokenizer(x)
        #src = self.input_proj(x) # [B, 200, d_model]
        src = src.transpose(1, 2)

        src = src + self.encoder_pos_embed.unsqueeze(0) # [B, 200, d_model]

        memory = self.transformer_encoder(src) # [B, 200, d_model]
        
        # --- Decoder ---
        tgt = self.target_queries.unsqueeze(0).expand(B, -1, -1) # [B, 100, d_model]

        tgt = tgt + self.spatial_pos_embed.unsqueeze(0)
        outs = self.transformer_decoder(tgt=tgt, memory=memory) # [B, 100, d_model]
        
        # --- Prediction ---
        logits = self.output_proj(outs).squeeze(-1) # [B, 100, 1] -> [B, 100]
        
        return logits

# =============================================================================
# 4. Training Loop: Focal Loss + Misfit Constraint
# =============================================================================
def train_inverse_model(dataset, num_epochs=100, batch_size=64, lr=7e-4, misfit_weight=0.1):
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = SpectrumTransformer().to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 1. Focal Loss (Solves heavy 95% zero imbalance)
    focal_criterion = FocalLoss(alpha=0.95, gamma=2.0)
    
    # 2. Misfit Loss (Injects physics constraint via total spatial probability sum)
    ref_spectra, concs = get_reference_spectra(dataset)
    misfit_criterion = MisfitLoss(ref_spectra, concs, tau=2.0).to(device)
    
    best_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            
            f_loss = focal_criterion(logits, targets)
            m_loss = misfit_criterion(inputs, logits)
            
            current_misfit_weight = misfit_weight * (epoch / num_epochs)
            
            total_loss = f_loss + (current_misfit_weight * m_loss)


            #total_loss = f_loss + (misfit_weight * m_loss)
            
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * inputs.size(0)
            
            # Print batch progress (transformers are much slower on CPU)
            if (i + 1) % 10 == 0:
                print(f"   [Epoch {epoch+1}] Processed batch {i+1} of {len(train_loader)}...")
            
        epoch_train_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                
                f_loss = focal_criterion(logits, targets)
                m_loss = misfit_criterion(inputs, logits)
                total_loss = f_loss + (misfit_weight * m_loss)
                
                val_loss += total_loss.item() * inputs.size(0)
                
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
def visualize_prediction(model, dataset, idx=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    x, y_true = dataset[idx]
    with torch.no_grad():
        x_in = x.unsqueeze(0).to(device)
        logits = model(x_in)
        y_pred = torch.sigmoid(logits).squeeze().cpu().numpy()
        
    y_true = y_true.numpy()
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(x.numpy(), label='Normalized Transmission', color='blue')
    plt.title('Target Spectrum T(E)')
    plt.xlabel('Energy Index')
    plt.ylabel('T / T_pristine')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(100), y_true, alpha=0.5, label='Actual Impurities', color='black')
    plt.bar(np.arange(100), y_pred, alpha=0.5, label='Predicted Probability', color='red')
    plt.title('Predicted Defect Unit Cells')
    plt.xlabel('Unit Cell Index [0-99]')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("Spectrum Transformer with Focal+Misfit Loss module ready.")
