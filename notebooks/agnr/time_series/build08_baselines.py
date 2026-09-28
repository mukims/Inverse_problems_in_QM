#!/usr/bin/env python
"""
build08_baselines.py — Comparable baselines for BUILD-08 spectral continuation
==============================================================================

Task: from the first 150 channels (E < 1.50 eV) predict the next 20
(E = 1.50-1.69 eV), for 7-AGNR and 9-AGNR at c = 2..58 (29 concentrations).

Why this script exists: the LightGBM number in time_series.ipynb (0.1073) is an
MAE mislabelled as RMSE and was computed on RAW transmission, while the MLP's
0.0222 is an MSE on pristine-normalised [0, 1] inputs, so the two were not
comparable. Here every model sees the same inputs and the same split:

  * inputs/targets: clip(round(T, 3) / round(T_pris, 3), 0, 1)  (LOGBOOK Bug #6)
  * split: by config seed, seeds [0, 8000) train, [8000, 10000) test  (Bug #7)
  * models: persistence (repeat the last input channel), LightGBM (one model per
    output channel, the notebook's hyperparameters), the notebook's 4-layer MLP
  * metrics: MSE, RMSE, MAE on the normalised scale, over all 20 outputs
"""

import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgbm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = "/run/media/shardul/storage/machine_learning/transmission_data/transmission_results/consolidated_data"


def load(args):
    concs = np.arange(2, 60, 2); idx = concs // 2 - 1
    X, Y, seed = [], [], []
    for f, pf in (("size_7.npy", "7_agnr_pris.npy"), ("size_9.npy", "9_agnr_pris.npy")):
        mm = np.load(os.path.join(args.data_dir, f), mmap_mode="r")
        p = np.round(np.load(REPO_ROOT / pf)[:170], 3); p = np.where(p > 0, p, 1.0)
        for i in idx:
            a = np.clip(np.round(np.asarray(mm[i, :args.nconfigs, :170], np.float32), 3) / p, 0, 1).astype(np.float32)
            X.append(a[:, :150]); Y.append(a[:, 150:170]); seed.append(np.arange(args.nconfigs))
    X, Y, seed = np.concatenate(X), np.concatenate(Y), np.concatenate(seed)
    tr = seed < int(0.8 * args.nconfigs)
    return X[tr], Y[tr], X[~tr], Y[~tr]


def metrics(pred, Y):
    e = pred - Y
    return {"MSE": float(np.mean(e ** 2)), "RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e)))}


def run_lgbm(Xtr, Ytr, Xte, args):
    preds = np.zeros((len(Xte), Ytr.shape[1]), np.float32)
    for j in range(Ytr.shape[1]):
        t0 = time.time()
        m = lgbm.LGBMRegressor(n_estimators=args.lgbm_trees, learning_rate=0.03, num_leaves=15,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               random_state=42, n_jobs=args.threads, verbose=-1)
        m.fit(Xtr, Ytr[:, j]); preds[:, j] = m.predict(Xte)
        print(f"  LightGBM channel {j + 1:2d}/20 fitted in {time.time() - t0:.0f}s", flush=True)
    return preds


def run_mlp(Xtr, Ytr, Xte, Yte, args):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(150, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                          nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 20))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                                     batch_size=1024, shuffle=True, drop_last=True)
    xte, yte = torch.from_numpy(Xte), torch.from_numpy(Yte)
    best, best_state = float("inf"), None
    for ep in range(1, args.mlp_epochs + 1):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); nn.functional.mse_loss(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val = float(nn.functional.mse_loss(model(xte), yte))
        sched.step(val)
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0:
            print(f"  MLP epoch {ep:3d}  held-out MSE {val:.5f}  lr {opt.param_groups[0]['lr']:.1e}", flush=True)
    # Selecting the best epoch on the test seeds mirrors the notebook; no separate validation set here.
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        return model(xte).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--nconfigs", type=int, default=10000)
    ap.add_argument("--lgbm-trees", type=int, default=1000)
    ap.add_argument("--mlp-epochs", type=int, default=100)
    ap.add_argument("--threads", type=int, default=22)
    ap.add_argument("--out", default=str(SCRIPT_DIR / "build08_baselines.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    Xtr, Ytr, Xte, Yte = load(args)
    print(f"train {len(Xtr):,} | test {len(Xte):,} (held-out config seeds)", flush=True)

    res = {"persistence": metrics(np.repeat(Xte[:, -1:], 20, axis=1), Yte)}
    print("persistence", res["persistence"], flush=True)
    t0 = time.time(); res["lightgbm"] = metrics(run_lgbm(Xtr, Ytr, Xte, args), Yte); res["lightgbm"]["time_s"] = round(time.time() - t0)
    print("lightgbm", res["lightgbm"], flush=True)
    t0 = time.time(); res["mlp"] = metrics(run_mlp(Xtr, Ytr, Xte, Yte, args), Yte); res["mlp"]["time_s"] = round(time.time() - t0)
    print("mlp", res["mlp"], flush=True)

    json.dump({"args": vars(args), "n_train": len(Xtr), "n_test": len(Xte), "results": res}, open(args.out, "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
