#!/usr/bin/env python
"""
train_conc_models.py -- step 2 of the AGNR inference pipeline.

Once the ribbon WIDTH is known (step 1), predict the impurity CONCENTRATION
from the transmission signature. Trains and benchmarks three model families on
the same split so the agent can pick the best-performing one at inference time:

    xgb          XGBoost regressor on the normalised spectrum
    mlp          ConductanceMLP-style fully-connected net (LayerNorm + dropout)
    transformer  patch-embedding encoder over the spectrum

Inputs are the combined per-concentration stacks produced for the 7-AGNR system
(``transmission_results_combined/conc_{c}.npy``, each (N, 300)).

Preprocessing (shared by all models, so the comparison is fair):
    x = clip(T, 0, T_pristine) / T_pristine        -> in [0, 1]
    crop to [crop_lo:crop_hi]                       (region of real variation)

Artifacts (written to --out):
    xgb_model.json / mlp_model.pt / transformer_model.pt
    conc_model_metrics.json      MAE/RMSE per model on the held-out split
    conc_model_compare.png       error distributions + predicted-vs-true

Example
-------
    python train_conc_models.py --data-dir ../../transmission_results_combined \\
        --pristine ~/agnr_infer/pristine_7.npy --models xgb mlp transformer
"""

import os
import sys
import json
import glob
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_dataset(data_dir, pristine, concs=None, max_per_conc=None,
                 crop_lo=20, crop_hi=150):
    data_dir = os.path.expanduser(data_dir)
    files = sorted(glob.glob(os.path.join(data_dir, "conc_*.npy")))
    X, y = [], []
    for f in files:
        base = os.path.basename(f)
        if base.endswith("_meta.csv"):
            continue
        try:
            c = int(base.replace("conc_", "").replace(".npy", ""))
        except ValueError:
            continue
        if concs is not None and c not in concs:
            continue
        arr = np.load(f)
        if arr.ndim != 2 or arr.shape[1] < crop_hi:
            continue
        if max_per_conc:
            arr = arr[:max_per_conc]
        # normalise by pristine, then crop
        p = np.where(pristine > 1e-12, pristine, 1.0)
        xn = np.clip(arr, 0, pristine) / p
        X.append(xn[:, crop_lo:crop_hi].astype(np.float32))
        y.append(np.full(len(arr), c, dtype=np.float32))
    if not X:
        raise SystemExit(f"no data found in {data_dir}")
    return np.concatenate(X), np.concatenate(y)


def split(X, y, test_frac=0.2, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], y[tr], X[te], y[te]


def metrics(pred, true):
    err = np.asarray(pred, float) - np.asarray(true, float)
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "max_err": float(np.max(np.abs(err)))}


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def train_xgb(Xtr, ytr, Xte, yte, out, seed=0):
    import xgboost as xgb
    t0 = time.time()
    model = xgb.XGBRegressor(
        n_estimators=600, max_depth=8, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", random_state=seed, n_jobs=4)
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    pred = model.predict(Xte)
    model.save_model(os.path.join(out, "xgb_model.json"))
    m = metrics(pred, yte); m["train_seconds"] = round(time.time() - t0, 1)
    return m, pred


def _torch_common():
    import torch
    torch.manual_seed(0)
    return torch, torch.device("cpu")


def train_mlp(Xtr, ytr, Xte, yte, out, epochs=60, bs=256, lr=1e-3):
    torch, dev = _torch_common()
    import torch.nn as nn
    t0 = time.time()
    D = Xtr.shape[1]

    model = nn.Sequential(
        nn.Linear(D, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(32, 1),
    ).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr).view(-1, 1)
    Xte_t = torch.tensor(Xte)
    n = len(Xtr_t)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[b]), ytr_t[b])
            loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).view(-1).numpy()
    torch.save(model.state_dict(), os.path.join(out, "mlp_model.pt"))
    m = metrics(pred, yte); m["train_seconds"] = round(time.time() - t0, 1)
    return m, pred


def train_transformer(Xtr, ytr, Xte, yte, out, epochs=40, bs=256, lr=1e-3,
                      patch=10, dmodel=64, nhead=4, layers=2):
    """Patch-embedding transformer encoder over the spectrum."""
    torch, dev = _torch_common()
    import torch.nn as nn
    t0 = time.time()
    D = Xtr.shape[1]
    pad = (-D) % patch
    npatch = (D + pad) // patch

    class PatchTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(patch, dmodel)
            self.pos = nn.Parameter(torch.zeros(1, npatch, dmodel))
            enc = nn.TransformerEncoderLayer(dmodel, nhead, dim_feedforward=4 * dmodel,
                                             dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(enc, layers)
            self.head = nn.Sequential(nn.LayerNorm(dmodel), nn.Linear(dmodel, 1))

        def forward(self, x):
            if pad:
                x = torch.nn.functional.pad(x, (0, pad))
            x = x.view(x.shape[0], npatch, patch)
            h = self.proj(x) + self.pos
            h = self.enc(h)
            return self.head(h.mean(1))

    model = PatchTransformer().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr).view(-1, 1)
    Xte_t = torch.tensor(Xte)
    n = len(Xtr_t)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[b]), ytr_t[b])
            loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).view(-1).numpy()
    torch.save(model.state_dict(), os.path.join(out, "transformer_model.pt"))
    m = metrics(pred, yte); m["train_seconds"] = round(time.time() - t0, 1)
    m["arch"] = {"patch": patch, "d_model": dmodel, "nhead": nhead, "layers": layers}
    return m, pred


TRAINERS = {"xgb": train_xgb, "mlp": train_mlp, "transformer": train_transformer}


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../../transmission_results_combined")
    ap.add_argument("--pristine", default="~/agnr_infer/pristine_7.npy")
    ap.add_argument("--out", default="~/agnr_infer/models")
    ap.add_argument("--models", nargs="+", default=["xgb", "mlp", "transformer"],
                    choices=list(TRAINERS))
    ap.add_argument("--concs", type=int, nargs="+", default=None)
    ap.add_argument("--max-per-conc", type=int, default=1500,
                    help="cap configs per concentration (keeps training tractable)")
    ap.add_argument("--crop-lo", type=int, default=20)
    ap.add_argument("--crop-hi", type=int, default=150)
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    pristine = np.load(os.path.expanduser(args.pristine))
    X, y = load_dataset(args.data_dir, pristine, concs=args.concs,
                        max_per_conc=args.max_per_conc,
                        crop_lo=args.crop_lo, crop_hi=args.crop_hi)
    Xtr, ytr, Xte, yte = split(X, y, test_frac=args.test_frac)
    print(f"[data] X={X.shape}  train={len(Xtr)}  test={len(Xte)}  "
          f"concs={int(y.min())}..{int(y.max())}")

    results, preds = {}, {}
    for name in args.models:
        print(f"[train] {name} ...")
        m, p = TRAINERS[name](Xtr, ytr, Xte, yte, out)
        results[name] = m
        preds[name] = p
        print(f"[train] {name}: MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} "
              f"({m['train_seconds']}s)")

    meta = {"crop": [args.crop_lo, args.crop_hi], "n_train": len(Xtr),
            "n_test": len(Xte), "results": results,
            "best": min(results, key=lambda k: results[k]["MAE"])}
    with open(os.path.join(out, "conc_model_metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)
    np.savez(os.path.join(out, "conc_model_preds.npz"), y_true=yte, **preds)
    print(f"\n[best] {meta['best']} (MAE {results[meta['best']]['MAE']:.3f})")
    print(f"[done] artifacts -> {out}")

    try:
        _plot(out, yte, preds, results)
    except Exception as ex:
        print(f"[warn] plot failed: {ex}")


def _plot(out, yte, preds, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(preds)
    fig, ax = plt.subplots(2, n, figsize=(5 * n, 8), squeeze=False)
    for i, (k, p) in enumerate(preds.items()):
        ax[0][i].scatter(yte, p, s=4, alpha=0.2)
        lim = [yte.min() - 2, yte.max() + 2]
        ax[0][i].plot(lim, lim, "k--", lw=0.8)
        ax[0][i].set_title(f"{k}: MAE={results[k]['MAE']:.2f}")
        ax[0][i].set_xlabel("true conc"); ax[0][i].set_ylabel("pred")
        ax[1][i].hist(p - yte, bins=60)
        ax[1][i].set_title(f"{k} error dist"); ax[1][i].set_xlabel("pred - true")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "conc_model_compare.png"), dpi=130)
    print(f"[plot] {os.path.join(out, 'conc_model_compare.png')}")


if __name__ == "__main__":
    main()
