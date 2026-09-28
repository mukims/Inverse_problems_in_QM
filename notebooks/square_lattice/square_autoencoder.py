#!/usr/bin/env python
"""
square_autoencoder.py — Standalone 1D-conv autoencoder for Square-10 spectra
============================================================================

Trains an unsupervised autoencoder on square-lattice (width 10) transmission
spectra, then asks: can impurity concentration be read off the embedding alone?

Pipeline
  1. Load stacked spectra from combine_sq.py output (conc_{c}.npy, (N, 400)).
     Normalise like the AGNR loaders (LOGBOOK Bug #6):
        X = clip(round(T, 3) / round(T_pris, 3), 0, 1)   over the first 150 channels
  2. Split by CONFIG ID, not by row: a config seed gives nested impurity sets
     across concentrations (cfg k at c=40 contains cfg k at c=20), so configs
     0..(n_train-1) train, the rest are held out at every concentration.
  3. Train Conv1dAutoencoder (same architecture as the 9-AGNR study) with MSE.
  4. Probe concentration from features fit on train configs, scored on held-out
     configs: AE latent, PCA of the spectrum (same dimension), raw spectrum.

Outputs (in --out-dir): sq_autoencoder.pt, sq_ae_metrics.json and plots.

Usage
    python square_autoencoder.py --concs 10 20 30 40 50 --per-conc 2000 --epochs 5   # smoke test
    python square_autoencoder.py                                                        # full grid
"""

import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONCS = list(range(5, 91, 5))


# =============================================================================
# 1. Data
# =============================================================================
def load_square(combined_dir, pristine_path, concs, per_conc, spectrum_len, n_train_cfg):
    p = np.round(np.load(pristine_path)[:spectrum_len], 3)
    p_safe = np.where(p > 0, p, 1.0)

    X, c_lab, cfg = [], [], []
    for c in concs:
        rows = np.load(os.path.join(combined_dir, f"conc_{c}.npy"), mmap_mode="r")
        meta = np.loadtxt(os.path.join(combined_dir, f"conc_{c}_meta.csv"), delimiter=",", skiprows=1, dtype=int)
        cfg_ids = meta[:, 1]
        # take per_conc rows spread over train and held-out configs in the same ratio
        tr = np.where(cfg_ids < n_train_cfg)[0]
        te = np.where(cfg_ids >= n_train_cfg)[0]
        n_tr = int(round(per_conc * len(tr) / len(cfg_ids)))
        pick = np.concatenate([tr[:n_tr], te[:per_conc - n_tr]])
        raw = np.round(np.asarray(rows[pick, :spectrum_len], dtype=np.float32), 3)
        X.append(np.clip(raw / p_safe, 0.0, 1.0))
        c_lab.append(np.full(len(pick), c, dtype=np.float32))
        cfg.append(cfg_ids[pick])
    X, c_lab, cfg = np.concatenate(X), np.concatenate(c_lab), np.concatenate(cfg)
    train = cfg < n_train_cfg
    return X.astype(np.float32), c_lab, train


# =============================================================================
# 2. Model (identical to the 9-AGNR Conv1dAutoencoder checkpoint architecture)
# =============================================================================
class Conv1dAutoencoder(nn.Module):
    def __init__(self, latent_dim=64, seq_len=152):
        super().__init__()
        self.enc_len = seq_len // 8
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3), nn.GroupNorm(1, 32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GroupNorm(1, 64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(1, 128), nn.ReLU(),
        )
        self.to_latent = nn.Linear(128 * self.enc_len, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 128 * self.enc_len)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.GroupNorm(1, 64), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 5, stride=2, padding=2, output_padding=1), nn.GroupNorm(1, 32), nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 7, stride=2, padding=3, output_padding=1),
        )

    def encode(self, x):
        return self.to_latent(self.encoder(x).flatten(1))

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(self.from_latent(z).view(-1, 128, self.enc_len)), z


def pad(X):
    """[N, 150] numpy -> [N, 1, 152] tensor (one zero channel each side, as in the AGNR study)."""
    return nn.functional.pad(torch.from_numpy(X).unsqueeze(1), (1, 1))


def unpad(t):
    return t[:, 0, 1:-1]


def train_ae(X_tr, X_va, args):
    model = Conv1dAutoencoder(latent_dim=args.latent)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pad(X_tr)),
                                         batch_size=args.batch_size, shuffle=True)
    xv = pad(X_va)
    hist, best, best_state, wait = {"train": [], "val": []}, float("inf"), None, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); model.train(); tot = 0.0
        for (xb,) in loader:
            recon, _ = model(xb)
            loss = nn.functional.mse_loss(recon, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val = sum(nn.functional.mse_loss(model(xv[i:i + 4096])[0], xv[i:i + 4096], reduction="sum").item()
                      for i in range(0, len(xv), 4096)) / xv.numel()
        hist["train"].append(tot / len(X_tr)); hist["val"].append(val)
        sched.step(val)
        mark = ""
        if best - val > 1e-6:
            best, best_state, wait, mark = val, {k: v.clone() for k, v in model.state_dict().items()}, 0, " *"
        else:
            wait += 1
        print(f"epoch {ep:3d}  train mse {hist['train'][-1]:.5f}  held-out mse {val:.5f}  "
              f"lr {opt.param_groups[0]['lr']:.1e}  {time.time() - t0:.1f}s{mark}", flush=True)
        if wait >= args.patience:
            print(f"early stop at epoch {ep}"); break
    model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def embed_and_reconstruct(model, X):
    model.eval(); Z, R = [], []
    xt = pad(X)
    for i in range(0, len(xt), 4096):
        r, z = model(xt[i:i + 4096]); Z.append(z.numpy()); R.append(unpad(r).numpy())
    return np.concatenate(Z), np.concatenate(R)


# =============================================================================
# 3. Probes
# =============================================================================
def score(pred, y):
    err = pred - y
    return {"R2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
            "MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2)))}


def run_probes(feats, y, train):
    results, preds = {}, {}
    for fname, F in feats.items():
        for pname, make in (("Ridge", lambda: Ridge(alpha=1.0)),
                            ("HistGB", lambda: HistGradientBoostingRegressor(max_iter=400, random_state=0))):
            m = make().fit(F[train], y[train])
            p = m.predict(F[~train])
            results[f"{fname} | {pname}"] = score(p, y[~train]); preds[f"{fname} | {pname}"] = p
            print(f"  {fname:<26s} {pname:<7s} R2 {results[f'{fname} | {pname}']['R2']:.3f}  "
                  f"MAE {results[f'{fname} | {pname}']['MAE']:.2f}", flush=True)
    return results, preds


# =============================================================================
# 4. Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combined-dir", default="~/transmissions_sq/size_10_combined")
    ap.add_argument("--pristine", default="~/transmissions_sq/pristine_10.npy")
    ap.add_argument("--concs", type=int, nargs="+", default=DEFAULT_CONCS)
    ap.add_argument("--per-conc", type=int, default=10000)
    ap.add_argument("--n-train-cfg", type=int, default=8000, help="configs [0, n) train; the rest are held out")
    ap.add_argument("--spectrum-len", type=int, default=150)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out-dir", default=str(SCRIPT_DIR / "sq_ae_results"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads); torch.manual_seed(0)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    X, y, train = load_square(os.path.expanduser(args.combined_dir), os.path.expanduser(args.pristine),
                              args.concs, args.per_conc, args.spectrum_len, args.n_train_cfg)
    print(f"Loaded {len(X):,} spectra | {len(args.concs)} concentrations | "
          f"train configs {train.sum():,} / held-out configs {(~train).sum():,}", flush=True)

    t0 = time.time()
    model, hist = train_ae(X[train], X[~train], args)
    train_time = time.time() - t0
    Z, R = embed_and_reconstruct(model, X)

    mse = float(np.mean((R[~train] - X[~train]) ** 2)); var = float(X[~train].var())
    recon = {"heldout_MSE": mse, "heldout_R2": 1 - mse / var,
             "MSE_by_conc": {int(c): float(np.mean((R[~train & (y == c)] - X[~train & (y == c)]) ** 2)) for c in args.concs}}
    print(f"\nReconstruction (held-out configs): MSE {mse:.5f} | R2 {recon['heldout_R2']:.3f}")

    pca = PCA(n_components=args.latent, random_state=0).fit(X[train])
    feats = {f"AE latent ({args.latent}-d)": Z, f"PCA ({args.latent}-d)": pca.transform(X),
             f"Spectrum ({args.spectrum_len} ch)": X}
    print("\nConcentration probes (fit on train configs, scored on held-out configs):")
    probes, preds = run_probes(feats, y, train)

    yt = y[~train]
    per_conc = {k: {int(c): float(np.mean(np.abs(p[yt == c] - c))) for c in args.concs} for k, p in preds.items()}

    torch.save({"model_state": model.state_dict(), "config": {"latent_dim": args.latent, "seq_len": 152},
                "args": vars(args)}, out / "sq_autoencoder.pt")
    json.dump({"args": vars(args), "n_spectra": int(len(X)), "train_time_s": round(train_time, 1),
               "epochs_run": len(hist["val"]), "reconstruction": recon, "probes": probes,
               "probe_MAE_by_conc": per_conc, "pca_explained_variance": float(pca.explained_variance_ratio_.sum())},
              open(out / "sq_ae_metrics.json", "w"), indent=2)

    # ---- plots ----
    E = np.arange(args.spectrum_len) * 0.01
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    ax[0].plot(hist["train"], label="train"); ax[0].plot(hist["val"], "--", label="held-out configs")
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE"); ax[0].legend(); ax[0].set_title("Autoencoder training")
    rng = np.random.default_rng(0); te_idx = np.where(~train)[0]
    for k, i in enumerate(rng.choice(te_idx, 3, replace=False)):
        ax[1].plot(E, X[i], color=f"C{k}", lw=1, label=f"c={int(y[i])}"); ax[1].plot(E, R[i], "--", color=f"C{k}", lw=1)
    ax[1].set_xlabel("E (eV)"); ax[1].set_ylabel("X"); ax[1].legend(); ax[1].set_title("Held-out spectra (solid) vs reconstruction (dashed)")
    pc = PCA(2).fit_transform(Z[~train])
    s = ax[2].scatter(pc[:, 0], pc[:, 1], c=yt, s=2, cmap="viridis"); plt.colorbar(s, ax=ax[2], label="c")
    ax[2].set_title("Latent space (PCA-2 of held-out embeddings)")
    plt.tight_layout(); plt.savefig(out / "sq_ae_training.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    best = f"AE latent ({args.latent}-d) | HistGB"
    ax[0].scatter(yt + rng.uniform(-1, 1, len(yt)), preds[best], s=2, alpha=0.3)
    ax[0].plot([0, max(args.concs)], [0, max(args.concs)], "r--")
    ax[0].set_xlabel("true c"); ax[0].set_ylabel("predicted c")
    ax[0].set_title(f"Concentration from embedding only ({best})\nMAE {probes[best]['MAE']:.2f} | R2 {probes[best]['R2']:.3f}")
    for k in preds:
        if k.endswith("HistGB"):
            ax[1].plot(args.concs, [per_conc[k][c] for c in args.concs], "o-", label=k.replace(" | HistGB", ""))
    ax[1].set_xlabel("true c"); ax[1].set_ylabel("MAE (impurities)"); ax[1].legend(); ax[1].set_title("Probe error by concentration (HistGB)")
    plt.tight_layout(); plt.savefig(out / "sq_ae_probes.png", dpi=130); plt.close()
    print(f"\nSaved checkpoint, metrics and plots to {out}")


if __name__ == "__main__":
    main()
