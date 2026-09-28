#!/usr/bin/env python
"""
joint_autoencoder.py — One autoencoder over 7-AGNR, 9-AGNR and Square-10 spectra
================================================================================

Trains a single unsupervised Conv1dAutoencoder (same architecture as the
9-AGNR and Square-10 studies) on all three systems at once, then probes the
shared embedding for:
  1. system identity (7-AGNR / 9-AGNR / Square-10), with a linear classifier
  2. impurity concentration, per system, with and without the system label

Inputs are normalised the same way for every system (LOGBOOK Bug #6):
    X = clip(round(T, 3) / round(T_pris, 3), 0, 1)   over the first 150 channels (E < 1.5 eV)

Split by CONFIG ID in every system: seeds give nested impurity sets across
concentrations, so configs [0, 0.8K) train and [0.8K, K) are held out.

Note: c is an impurity COUNT on ~1,400 (7-AGNR), ~1,800 (9-AGNR) and 1,000
(Square-10) sites, so equal c means different densities across systems.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "notebooks" / "square_lattice"))
from square_autoencoder import Conv1dAutoencoder, train_ae, embed_and_reconstruct  # noqa: E402

AGNR_DIR = "/run/media/shardul/storage/machine_learning/transmission_data/transmission_results/consolidated_data"
SYSTEMS = ["7-AGNR", "9-AGNR", "Square-10"]


def normalise(raw, pris):
    p = np.round(pris, 3)
    return np.clip(np.round(raw, 3) / np.where(p > 0, p, 1.0), 0.0, 1.0).astype(np.float32)


def load_all(args):
    L, K = args.spectrum_len, args.per_conc
    n_tr = int(0.8 * K)
    X, sys_id, conc, train = [], [], [], []

    # AGNR: row index within a concentration == config seed
    for s, (fname, pname) in enumerate((("size_7.npy", "7_agnr_pris.npy"), ("size_9.npy", "9_agnr_pris.npy"))):
        mm = np.load(os.path.join(args.agnr_dir, fname), mmap_mode="r")
        pris = np.load(REPO_ROOT / pname)[:L]
        concs = np.arange(2, 2 * mm.shape[0] + 1, 2)
        for i, c in enumerate(concs):
            X.append(normalise(np.asarray(mm[i, :K, :L], dtype=np.float32), pris))
            sys_id.append(np.full(K, s)); conc.append(np.full(K, c, float)); train.append(np.arange(K) < n_tr)

    # Square-10: stacked by combine_sq.py, config seed in the meta file
    sq_dir = os.path.expanduser(args.square_dir)
    pris = np.load(os.path.expanduser(args.square_pristine))[:L]
    for c in args.square_concs:
        rows = np.load(os.path.join(sq_dir, f"conc_{c}.npy"), mmap_mode="r")
        cfg = np.loadtxt(os.path.join(sq_dir, f"conc_{c}_meta.csv"), delimiter=",", skiprows=1, dtype=int)[:, 1]
        pick = np.where(cfg < K)[0]
        X.append(normalise(np.asarray(rows[pick, :L], dtype=np.float32), pris))
        sys_id.append(np.full(len(pick), 2)); conc.append(np.full(len(pick), c, float)); train.append(cfg[pick] < n_tr)

    return (np.concatenate(X), np.concatenate(sys_id), np.concatenate(conc), np.concatenate(train))


def probe_system(F, s, train):
    sc = StandardScaler().fit(F[train])
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(F[train]), s[train])
    return float((clf.predict(sc.transform(F[~train])) == s[~train]).mean() * 100)


def probe_conc(F, s, y, train, with_system):
    if with_system:
        F = np.hstack([F, np.eye(3)[s]])
    m = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.1, max_leaf_nodes=63,
                                      early_stopping=False, random_state=0).fit(F[train], y[train])
    p = m.predict(F[~train]); st, yt = s[~train], y[~train]
    out = {"all": float(np.abs(p - yt).mean())}
    for k, name in enumerate(SYSTEMS):
        out[name] = float(np.abs(p[st == k] - yt[st == k]).mean())
    return out, p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agnr-dir", default=AGNR_DIR)
    ap.add_argument("--square-dir", default="~/transmissions_sq/size_10_combined")
    ap.add_argument("--square-pristine", default="~/transmissions_sq/pristine_10.npy")
    ap.add_argument("--square-concs", type=int, nargs="+", default=list(range(5, 91, 5)))
    ap.add_argument("--per-conc", type=int, default=3000, help="configs per concentration per system")
    ap.add_argument("--spectrum-len", type=int, default=150)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out-dir", default=str(SCRIPT_DIR / "results"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads); torch.manual_seed(0)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    X, s, y, train = load_all(args)
    counts = {n: int((s == k).sum()) for k, n in enumerate(SYSTEMS)}
    print(f"Loaded {len(X):,} spectra {counts} | train {train.sum():,} / held-out {(~train).sum():,}", flush=True)

    t0 = time.time()
    model, hist = train_ae(X[train], X[~train], args)
    train_time = time.time() - t0
    Z, R = embed_and_reconstruct(model, X)

    recon = {}
    for k, n in enumerate(SYSTEMS):
        m = ~train & (s == k)
        mse = float(np.mean((R[m] - X[m]) ** 2)); recon[n] = {"MSE": mse, "R2": 1 - mse / float(X[m].var())}
    print("\nReconstruction (held-out):", {n: round(v["R2"], 3) for n, v in recon.items()})

    pca = PCA(n_components=args.latent, random_state=0).fit(X[train])
    feats = {f"AE latent ({args.latent}-d)": Z, f"PCA ({args.latent}-d)": pca.transform(X),
             f"Spectrum ({args.spectrum_len} ch)": X}

    results, preds = {}, {}
    print("\nProbes (fit on train configs, scored on held-out configs)")
    for fname, F in feats.items():
        acc = probe_system(F, s, train)
        c_blind, p_blind = probe_conc(F, s, y, train, with_system=False)
        c_told, _ = probe_conc(F, s, y, train, with_system=True)
        results[fname] = {"system_acc": acc, "conc_MAE_no_system_label": c_blind, "conc_MAE_with_system_label": c_told}
        preds[fname] = p_blind
        print(f"  {fname:<20s} system acc {acc:6.2f}% | conc MAE (no label) " +
              " ".join(f"{k} {v:.2f}" for k, v in c_blind.items()) +
              " | (with label) " + " ".join(f"{k} {v:.2f}" for k, v in c_told.items()), flush=True)

    torch.save({"model_state": model.state_dict(), "config": {"latent_dim": args.latent, "seq_len": 152},
                "args": vars(args)}, out / "joint_autoencoder.pt")
    json.dump({"args": vars(args), "counts": counts, "train_time_s": round(train_time, 1),
               "epochs_run": len(hist["val"]), "reconstruction": recon, "probes": results},
              open(out / "joint_ae_metrics.json", "w"), indent=2)

    # ---- plots ----
    te = ~train
    pc = PCA(2, random_state=0).fit_transform(Z[te])
    rng = np.random.default_rng(0); sub = rng.choice(te.sum(), min(30000, te.sum()), replace=False)
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
    for k, n in enumerate(SYSTEMS):
        m = s[te][sub] == k
        ax[0].scatter(pc[sub][m, 0], pc[sub][m, 1], s=2, alpha=0.5, label=n)
    ax[0].legend(markerscale=6); ax[0].set_title("Shared latent space (PCA-2), coloured by system")
    sc = ax[1].scatter(pc[sub, 0], pc[sub, 1], c=y[te][sub], s=2, cmap="viridis")
    plt.colorbar(sc, ax=ax[1], label="c (impurity count)"); ax[1].set_title("Same space, coloured by concentration")
    best = f"AE latent ({args.latent}-d)"
    for k, n in enumerate(SYSTEMS):
        m = s[te] == k; cs = np.unique(y[te][m])
        ax[2].plot(cs, [np.abs(preds[best][m][y[te][m] == c] - c).mean() for c in cs], "o-", ms=3, label=n)
    ax[2].set_xlabel("true c"); ax[2].set_ylabel("MAE (impurities)"); ax[2].legend()
    ax[2].set_title("Concentration from the joint embedding (no system label)")
    plt.tight_layout(); plt.savefig(out / "joint_ae_latent_and_probes.png", dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist["train"], label="train"); ax.plot(hist["val"], "--", label="held-out configs")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.legend(); ax.set_title("Joint autoencoder training")
    plt.tight_layout(); plt.savefig(out / "joint_ae_training.png", dpi=130); plt.close()
    print(f"\nSaved checkpoint, metrics and plots to {out}")


if __name__ == "__main__":
    main()
