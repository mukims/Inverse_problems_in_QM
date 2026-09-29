"""1D-conv autoencoder (same design as the 9-AGNR/square studies) and helpers."""
import time

import numpy as np
import torch
import torch.nn as nn


class Conv1dAE(nn.Module):
    def __init__(self, latent: int, seq_len: int):
        super().__init__()
        self.seq_len, self.enc_len = seq_len, seq_len // 8
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, 2, 3), nn.GroupNorm(1, 32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, 2, 2), nn.GroupNorm(1, 64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, 2, 1), nn.GroupNorm(1, 128), nn.ReLU())
        self.to_latent = nn.Linear(128 * self.enc_len, latent)
        self.from_latent = nn.Linear(latent, 128 * self.enc_len)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 3, 2, 1, 1), nn.GroupNorm(1, 64), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 5, 2, 2, 1), nn.GroupNorm(1, 32), nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 7, 2, 3, 1))

    def forward(self, x):
        z = self.to_latent(self.encoder(x).flatten(1))
        return self.decoder(self.from_latent(z).view(-1, 128, self.enc_len)), z


def _t(X):
    return torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).unsqueeze(1)


def train_autoencoder(X_tr, X_va, latent=32, epochs=60, patience=8, lr=1e-3, batch_size=256, threads=4, seed=0):
    """X length must be a multiple of 8 (400 channels satisfies this)."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = Conv1dAE(latent, X_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(_t(X_tr)), batch_size=batch_size, shuffle=True)
    xv = _t(X_va)
    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        for (xb,) in loader:
            loss = nn.functional.mse_loss(model(xb)[0], xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(nn.functional.mse_loss(model(xv)[0], xv))
        hist.append(val)
        sched.step(val)
        print(f"[atlas] epoch {ep + 1:3d} val mse {val:.6f} {time.time() - t0:.0f}s", flush=True)
        if val < best - 1e-7:
            best, best_state, wait = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, hist


@torch.no_grad()
def embed(model, X):
    if len(X) == 0:
        return np.empty((0, model.to_latent.out_features), dtype=np.float32), np.empty(0, dtype=np.float32)
    Z, err = [], []
    for i in range(0, len(X), 4096):
        xb = _t(X[i:i + 4096])
        r, z = model(xb)
        Z.append(z.cpu().numpy() if hasattr(z, "cpu") else z.numpy())
        err.append(((r - xb) ** 2).mean((1, 2)).cpu().numpy())
    return np.concatenate(Z), np.concatenate(err)
