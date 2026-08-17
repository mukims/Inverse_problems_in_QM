#!/usr/bin/env python
"""
run_sequential_pipeline.py — Sequential Multi-Width Modeling & Benchmark Pipeline.

Executes the four modeling tasks in strict sequence:
  1. Physical Misfit Function Analysis (Width & Concentration analytical baseline)
  2. XGBoost Baseline (Width Classifier + Concentration Regressor)
  3. Multi-Task ConductanceMLP (deep MLP with dual heads)
  4. Multi-Task Patched Transformer v2 (ConvStem + Self-Attention + [CLS] head)
  5. Unified Comparison & Visualization Generation

Usage:
------
    python run_sequential_pipeline.py --stage all
    python run_sequential_pipeline.py --stage misfit
    python run_sequential_pipeline.py --stage xgb
    python run_sequential_pipeline.py --stage mlp
    python run_sequential_pipeline.py --stage transformer
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import xgboost as xgb
from tqdm.auto import tqdm

# Set unbuffered stdout and thread count
torch.set_num_threads(min(20, max(1, os.cpu_count() - 2)))
sys.stdout.reconfigure(line_buffering=True)


# ======================================================================
# LOGGING & PROGRESS HELPERS
# ======================================================================

LOG_PATH = None        # set from --log-file in main()
SHOW_PROGRESS = True   # disabled by --no-progress
_T_START = time.time()


def log(msg="", stamp=True):
    """Print to stdout and mirror to the log file so progress survives the terminal.

    Progress bars go to stderr (tqdm default), so redirecting stdout to a file
    keeps the log clean:  python run_sequential_pipeline.py > run.log
    """
    body = msg.strip()
    is_rule = body and set(body) <= set("=-")   # separator lines stay unstamped
    if stamp and body and not is_rule:
        elapsed = time.time() - _T_START
        prefix = f"[{datetime.now():%H:%M:%S} | +{elapsed/60:6.1f}m] "
    else:
        prefix = ""
    line = prefix + msg
    print(line, flush=True)
    if LOG_PATH is not None:
        with open(LOG_PATH, "a") as fh:
            fh.write(line + "\n")


def pbar(iterable, desc, unit="it", total=None, leave=False):
    """tqdm wrapper honouring --no-progress; writes to stderr."""
    return tqdm(
        iterable, desc=desc, unit=unit, total=total, leave=leave,
        disable=not SHOW_PROGRESS, dynamic_ncols=True, file=sys.stderr,
    )


def fmt_eta(done, total, elapsed):
    """Human-readable ETA from fraction complete."""
    if done <= 0:
        return "??"
    rate = elapsed / done
    remain = rate * (total - done)
    return f"{remain/60:.1f}m left ({rate:.1f}s/epoch)"


# ======================================================================
# MODEL ARCHITECTURES
# ======================================================================

class MultiTaskConductanceMLP(nn.Module):
    def __init__(self, input_length=150, hidden_dims=None, dropout=0.2, noise_std=0.01):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        self.noise_std = noise_std
        layers = []
        in_dim = input_length
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.width_head = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 2)
        )
        self.conc_head = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        if self.training and self.noise_std > 0:
            x = x * (1.0 + torch.randn_like(x) * self.noise_std)
        feat = self.backbone(x)
        return self.width_head(feat), self.conc_head(feat)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device))
        return x * mask / keep


class ConvStem(nn.Module):
    def __init__(self, out_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3),
            nn.GELU(),
            nn.Conv1d(16, out_channels, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class PatchEmbedding1D(nn.Module):
    def __init__(self, seq_len=150, patch_size=10, in_channels=32, embed_dim=128):
        super().__init__()
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return self.norm(x) + self.pos_embed


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4.0, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class MultiTaskPatchedTransformerV2(nn.Module):
    def __init__(self, seq_len=150, patch_size=10, stem_channels=32,
                 embed_dim=128, depth=4, num_heads=4, mlp_ratio=4.0,
                 dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        self.stem = ConvStem(out_channels=stem_channels)
        self.patch_embed = PatchEmbedding1D(
            seq_len=seq_len, patch_size=patch_size,
            in_channels=stem_channels, embed_dim=embed_dim
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                             dropout=dropout, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.width_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )
        self.conc_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.stem(x)
        tokens = self.patch_embed(feat)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        cls_repr = tokens[:, 0]
        return self.width_head(cls_repr), self.conc_head(cls_repr)


# ======================================================================
# DATA LOADING & EVALUATION HELPERS
# ======================================================================

def load_data(consolidated_dir, samples_per_conc=3000, spectrum_len=150, seed=42):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]

    # Load corrected base pristine files
    p7_path = project_root / "7_agnr_pris.npy"
    p9_path = project_root / "9_agnr_pris.npy"

    p7 = np.load(str(p7_path))[:spectrum_len].astype(np.float32)
    p9 = np.load(str(p9_path))[:spectrum_len].astype(np.float32)
    p7_safe = np.where(p7 > 1e-12, p7, 1.0)
    p9_safe = np.where(p9 > 1e-12, p9, 1.0)

    log(f"  Pristine refs loaded: 7-AGNR max={p7.max():.2f} G0 | 9-AGNR max={p9.max():.2f} G0")

    s7_mmap = np.load(os.path.join(consolidated_dir, "size_7.npy"), mmap_mode="r")
    s9_mmap = np.load(os.path.join(consolidated_dir, "size_9.npy"), mmap_mode="r")
    log(f"  Memory-mapped size_7.npy {s7_mmap.shape} | size_9.npy {s9_mmap.shape}")

    X_all, yw_all, yc_all, raw_all = [], [], [], []

    # 7-AGNR (34 concs: 2..68)
    concs_7 = np.arange(2, 70, 2)
    for idx, c in enumerate(pbar(concs_7, "Loading 7-AGNR", unit="conc")):
        if idx >= s7_mmap.shape[0]:
            break
        raw = np.array(s7_mmap[idx, :samples_per_conc, :spectrum_len], dtype=np.float32)
        norm = np.clip(raw / p7_safe, 0.0, 1.0)
        X_all.append(norm)
        raw_all.append(raw)
        yw_all.append(np.zeros(samples_per_conc, dtype=np.int64))
        yc_all.append(np.full(samples_per_conc, c, dtype=np.float32))

    # 9-AGNR (49 concs: 2..98)
    concs_9 = np.arange(2, 100, 2)
    for idx, c in enumerate(pbar(concs_9, "Loading 9-AGNR", unit="conc")):
        if idx >= s9_mmap.shape[0]:
            break
        raw = np.array(s9_mmap[idx, :samples_per_conc, :spectrum_len], dtype=np.float32)
        norm = np.clip(raw / p9_safe, 0.0, 1.0)
        X_all.append(norm)
        raw_all.append(raw)
        yw_all.append(np.ones(samples_per_conc, dtype=np.int64))
        yc_all.append(np.full(samples_per_conc, c, dtype=np.float32))

    X = np.concatenate(X_all, axis=0)
    raw_arr = np.concatenate(raw_all, axis=0)
    y_width = np.concatenate(yw_all, axis=0)
    y_conc = np.concatenate(yc_all, axis=0)

    # 70/15/15 split
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))
    n_tr = int(len(X) * 0.70)
    n_va = int(len(X) * 0.15)

    tr_idx = perm[:n_tr]
    va_idx = perm[n_tr:n_tr + n_va]
    te_idx = perm[n_tr + n_va:]

    train_data = (X[tr_idx], y_width[tr_idx], y_conc[tr_idx], raw_arr[tr_idx])
    val_data = (X[va_idx], y_width[va_idx], y_conc[va_idx], raw_arr[va_idx])
    test_data = (X[te_idx], y_width[te_idx], y_conc[te_idx], raw_arr[te_idx])

    log(f"  Split: train {len(tr_idx):,} | val {len(va_idx):,} | test {len(te_idx):,}")
    log(f"  Tensor footprint: X={X.nbytes/1e9:.2f} GB, raw={raw_arr.nbytes/1e9:.2f} GB "
        f"| conc range [{y_conc.min():.0f}, {y_conc.max():.0f}] | 7-AGNR {(y_width==0).sum():,} / 9-AGNR {(y_width==1).sum():,}")

    return train_data, val_data, test_data, p7, p9, concs_7, concs_9


def compute_multi_metrics(w_preds, w_true, c_preds, c_true):
    w_acc = float(np.mean(w_preds == w_true)) * 100.0
    err = np.asarray(c_preds, float) - np.asarray(c_true, float)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    max_err = float(np.max(np.abs(err)))

    m7 = (w_true == 0)
    m9 = (w_true == 1)
    acc7 = float(np.mean(w_preds[m7] == 0)) * 100.0 if np.any(m7) else 0.0
    acc9 = float(np.mean(w_preds[m9] == 1)) * 100.0 if np.any(m9) else 0.0
    mae7 = float(np.mean(np.abs(err[m7]))) if np.any(m7) else 0.0
    rmse7 = float(np.sqrt(np.mean(err[m7] ** 2))) if np.any(m7) else 0.0
    mae9 = float(np.mean(np.abs(err[m9]))) if np.any(m9) else 0.0
    rmse9 = float(np.sqrt(np.mean(err[m9] ** 2))) if np.any(m9) else 0.0

    return {
        "Width_Acc_Overall": w_acc,
        "Width_Acc_7": acc7,
        "Width_Acc_9": acc9,
        "Conc_MAE_Overall": mae,
        "Conc_RMSE_Overall": rmse,
        "Conc_Max_Error": max_err,
        "Conc_MAE_7": mae7,
        "Conc_RMSE_7": rmse7,
        "Conc_MAE_9": mae9,
        "Conc_RMSE_9": rmse9,
    }


# ======================================================================
# STAGE 1: PHYSICAL MISFIT FUNCTION ANALYSIS
# ======================================================================

def run_stage_1_misfit(train_data, test_data, p7, p9, concs_7, concs_9, crop_start=20, crop_end=150):
    log("\n" + "=" * 80)
    log("STAGE 1: PHYSICAL MISFIT FUNCTION ANALYSIS (ANALYTICAL BASELINE)")
    log("=" * 80)
    t0 = time.time()

    X_tr, yw_tr, yc_tr, _ = train_data
    X_te, yw_te, yc_te, raw_te = test_data

    # Build reference spectra by averaging training configurations for each concentration
    p7_crop = p7[crop_start:crop_end]
    p9_crop = p9[crop_start:crop_end]

    ref7_list = []
    for c in concs_7:
        mask = (yw_tr == 0) & (yc_tr == c)
        if np.any(mask):
            avg_norm = np.mean(X_tr[mask, crop_start:crop_end], axis=0)
            ref7_list.append(avg_norm * p7_crop)
        else:
            ref7_list.append(p7_crop)
    ref7_mat = np.array(ref7_list, dtype=np.float32)  # [C7, L]

    ref9_list = []
    for c in concs_9:
        mask = (yw_tr == 1) & (yc_tr == c)
        if np.any(mask):
            avg_norm = np.mean(X_tr[mask, crop_start:crop_end], axis=0)
            ref9_list.append(avg_norm * p9_crop)
        else:
            ref9_list.append(p9_crop)
    ref9_mat = np.array(ref9_list, dtype=np.float32)  # [C9, L]

    log(f"  Reference libraries built: 7-AGNR ({len(concs_7)} concs) | 9-AGNR ({len(concs_9)} concs)")
    log(f"  Evaluating Physical Misfit across {len(X_te):,} test samples (chunked)...")

    raw_crop = raw_te[:, crop_start:crop_end]
    L = crop_end - crop_start
    N = raw_crop.shape[0]

    # Chunked scan: the full [N, C, L] difference tensor would be ~1 GB at
    # N=37k, C=49, L=130. Chunking bounds peak memory to a few tens of MB.
    CHUNK = 2048
    min7_val = np.empty(N, dtype=np.float32)
    best7_c = np.empty(N, dtype=np.float32)
    min9_val = np.empty(N, dtype=np.float32)
    best9_c = np.empty(N, dtype=np.float32)

    n_chunks = (N + CHUNK - 1) // CHUNK
    for s in pbar(range(0, N, CHUNK), "Misfit scan", unit="chunk", total=n_chunks):
        e = min(s + CHUNK, N)
        blk = raw_crop[s:e]
        rows = np.arange(e - s)

        mis7 = np.sum((blk[:, None, :] - ref7_mat[None, :, :]) ** 2, axis=2) / float(L)
        i7 = np.argmin(mis7, axis=1)
        min7_val[s:e] = mis7[rows, i7]
        best7_c[s:e] = concs_7[i7]

        mis9 = np.sum((blk[:, None, :] - ref9_mat[None, :, :]) ** 2, axis=2) / float(L)
        i9 = np.argmin(mis9, axis=1)
        min9_val[s:e] = mis9[rows, i9]
        best9_c[s:e] = concs_9[i9]

    # Width Decision: Choose width with minimum envelope misfit
    pred_w = np.where(min7_val <= min9_val, 0, 1)
    pred_c = np.where(pred_w == 0, best7_c, best9_c)

    metrics = compute_multi_metrics(pred_w, yw_te, pred_c, yc_te)
    metrics["eval_time_sec"] = round(time.time() - t0, 1)

    log("-" * 80)
    log(f"Physical Misfit Results (Elapsed: {metrics['eval_time_sec']}s):")
    log(f"  Width Classification Accuracy: {metrics['Width_Acc_Overall']:.2f}% (7-AGNR: {metrics['Width_Acc_7']:.1f}%, 9-AGNR: {metrics['Width_Acc_9']:.1f}%)")
    log(f"  Overall Conc MAE: {metrics['Conc_MAE_Overall']:.3f} | RMSE: {metrics['Conc_RMSE_Overall']:.3f} | Max Err: {metrics['Conc_Max_Error']:.1f}")
    log(f"  7-AGNR MAE: {metrics['Conc_MAE_7']:.3f} | 9-AGNR MAE: {metrics['Conc_MAE_9']:.3f}")
    log("=" * 80 + "\n")

    return metrics, (pred_w, pred_c)


# ======================================================================
# STAGE 2: XGBOOST BASELINES
# ======================================================================

def run_stage_2_xgboost(train_data, test_data, save_dir):
    log("\n" + "=" * 80)
    log("STAGE 2: XGBOOST MODEL TRAINING & EVALUATION")
    log("=" * 80)
    t0 = time.time()
    X_tr, yw_tr, yc_tr, _ = train_data
    X_te, yw_te, yc_te, _ = test_data

    # 1. Width Classifier
    log(f"[1/2] Training XGBoost Width Classifier (300 trees, depth 6) on {len(X_tr):,} samples...")
    t_clf = time.time()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        tree_method="hist", random_state=42, n_jobs=-1
    )
    xgb_clf.fit(X_tr, yw_tr)
    log(f"      classifier fit in {time.time() - t_clf:.1f}s")
    pred_w = xgb_clf.predict(X_te)
    clf_path = os.path.join(save_dir, "multi_width_xgb_width.json")
    xgb_clf.save_model(clf_path)
    log(f"✓ Saved XGBoost classifier to {clf_path}")

    # 2. Concentration Regressor
    log(f"[2/2] Training XGBoost Concentration Regressor (500 trees, depth 8) on {len(X_tr):,} samples...")
    t_reg = time.time()
    xgb_reg = xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", random_state=42, n_jobs=-1
    )
    prob_w_tr = xgb_clf.predict_proba(X_tr)[:, 1:2]
    prob_w_te = xgb_clf.predict_proba(X_te)[:, 1:2]
    X_tr_aug = np.hstack([X_tr, prob_w_tr])
    X_te_aug = np.hstack([X_te, prob_w_te])

    log(f"      width-probability feature appended -> {X_tr_aug.shape[1]} features")
    xgb_reg.fit(X_tr_aug, yc_tr)
    log(f"      regressor fit in {time.time() - t_reg:.1f}s")
    pred_c = xgb_reg.predict(X_te_aug)
    reg_path = os.path.join(save_dir, "multi_width_xgb_conc.json")
    xgb_reg.save_model(reg_path)
    log(f"✓ Saved XGBoost regressor to {reg_path}")

    metrics = compute_multi_metrics(pred_w, yw_te, pred_c, yc_te)
    metrics["train_time_sec"] = round(time.time() - t0, 1)

    log("-" * 80)
    log(f"XGBoost Results (Elapsed: {metrics['train_time_sec']}s):")
    log(f"  Width Classification Accuracy: {metrics['Width_Acc_Overall']:.2f}% (7: {metrics['Width_Acc_7']:.1f}%, 9: {metrics['Width_Acc_9']:.1f}%)")
    log(f"  Overall Conc MAE: {metrics['Conc_MAE_Overall']:.3f} | RMSE: {metrics['Conc_RMSE_Overall']:.3f} | Max Err: {metrics['Conc_Max_Error']:.1f}")
    log(f"  7-AGNR MAE: {metrics['Conc_MAE_7']:.3f} | 9-AGNR MAE: {metrics['Conc_MAE_9']:.3f}")
    log("=" * 80 + "\n")

    return metrics, (pred_w, pred_c)


# ======================================================================
# STAGE 3: MULTI-TASK CONDUCTANCE MLP
# ======================================================================

def run_stage_3_mlp(train_data, val_data, test_data, save_path,
                    epochs=50, batch_size=256, lr=1e-3, alpha_width=10.0, patience=15):
    log("\n" + "=" * 80)
    log("STAGE 3: MULTI-TASK CONDUCTANCE MLP TRAINING")
    log("=" * 80)
    t0 = time.time()
    device = torch.device("cpu")

    X_tr, yw_tr, yc_tr, _ = train_data
    X_va, yw_va, yc_va, _ = val_data
    X_te, yw_te, yc_te, _ = test_data

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(yw_tr, dtype=torch.long),
        torch.tensor(yc_tr, dtype=torch.float32).unsqueeze(1)
    )
    val_ds = TensorDataset(
        torch.tensor(X_va, dtype=torch.float32),
        torch.tensor(yw_va, dtype=torch.long),
        torch.tensor(yc_va, dtype=torch.float32).unsqueeze(1)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MultiTaskConductanceMLP(input_length=X_tr.shape[1]).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"  Model: MultiTaskConductanceMLP | {n_par:,} params | device={device} | threads={torch.get_num_threads()}")
    log(f"  Train {len(train_ds):,} samples / {len(train_loader):,} batches per epoch (bs={batch_size}) | "
        f"val {len(val_ds):,} | max {epochs} epochs, patience {patience}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_mae": []}

    epoch_bar = pbar(range(1, epochs + 1), "MLP epochs", unit="ep", total=epochs, leave=True)
    for epoch in epoch_bar:
        ep_t0 = time.time()
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_ce = 0.0

        batch_bar = pbar(train_loader, f"  ep {epoch:03d}/{epochs:03d} train",
                         unit="b", total=len(train_loader))
        for xb, ywb, ycb in batch_bar:
            optimizer.zero_grad()
            w_logits, c_pred = model(xb)
            l_ce = ce_loss(w_logits, ywb)
            l_mse = mse_loss(c_pred, ycb)
            loss = l_mse + alpha_width * l_ce
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(xb)
            train_mse += l_mse.item() * len(xb)
            train_ce += l_ce.item() * len(xb)
            batch_bar.set_postfix(loss=f"{loss.item():.2f}", mse=f"{l_mse.item():.2f}")

        train_loss /= len(train_ds)
        train_mse /= len(train_ds)
        train_ce /= len(train_ds)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        correct_w = 0
        total_c_err = 0.0

        with torch.no_grad():
            for xb, ywb, ycb in pbar(val_loader, f"  ep {epoch:03d}/{epochs:03d} val",
                                     unit="b", total=len(val_loader)):
                w_logits, c_pred = model(xb)
                l_ce = ce_loss(w_logits, ywb)
                l_mse = mse_loss(c_pred, ycb)
                v_loss = l_mse + alpha_width * l_ce
                val_loss += v_loss.item() * len(xb)
                pred_w = torch.argmax(w_logits, dim=1)
                correct_w += (pred_w == ywb).sum().item()
                total_c_err += torch.abs(c_pred - ycb).sum().item()

        val_loss /= len(val_ds)
        val_acc = (correct_w / len(val_ds)) * 100.0
        val_mae = total_c_err / len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_mae"].append(val_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            star = " * Best"
        else:
            no_improve += 1
            star = ""

        ep_dt = time.time() - ep_t0
        epoch_bar.set_postfix(mae=f"{val_mae:.3f}", acc=f"{val_acc:.2f}%", loss=f"{val_loss:.2f}")
        log(f"Epoch {epoch:03d}/{epochs:03d} | Train: {train_loss:.3f} (MSE:{train_mse:.2f}, CE:{train_ce:.3f}) | "
            f"Val: {val_loss:.3f} (Width Acc: {val_acc:.2f}%, Conc MAE: {val_mae:.3f}) | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | {ep_dt:.1f}s/ep | "
            f"ETA {fmt_eta(epoch, epochs, time.time() - t0)}{star}")

        if no_improve >= patience:
            log(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    checkpoint = {
        "model_state_dict": best_state,
        "args": {"input_length": X_tr.shape[1], "hidden_dims": [256, 128, 64]},
        "val_loss": best_val_loss,
    }
    torch.save(checkpoint, save_path)
    log(f"✓ Saved ConductanceMLP checkpoint to: {save_path}")

    # Evaluate on held-out test set
    model.eval()
    with torch.no_grad():
        w_logits, c_pred = model(torch.tensor(X_te, dtype=torch.float32))
        pred_w = torch.argmax(w_logits, dim=1).numpy()
        pred_c = c_pred.squeeze().numpy()

    metrics = compute_multi_metrics(pred_w, yw_te, pred_c, yc_te)
    metrics["train_time_sec"] = round(time.time() - t0, 1)

    log("-" * 80)
    log(f"ConductanceMLP Results (Elapsed: {metrics['train_time_sec']}s):")
    log(f"  Width Classification Accuracy: {metrics['Width_Acc_Overall']:.2f}% (7: {metrics['Width_Acc_7']:.1f}%, 9: {metrics['Width_Acc_9']:.1f}%)")
    log(f"  Overall Conc MAE: {metrics['Conc_MAE_Overall']:.3f} | RMSE: {metrics['Conc_RMSE_Overall']:.3f} | Max Err: {metrics['Conc_Max_Error']:.1f}")
    log(f"  7-AGNR MAE: {metrics['Conc_MAE_7']:.3f} | 9-AGNR MAE: {metrics['Conc_MAE_9']:.3f}")
    log("=" * 80 + "\n")

    return metrics, (pred_w, pred_c), history


# ======================================================================
# STAGE 4: MULTI-TASK PATCHED TRANSFORMER V2
# ======================================================================

def run_stage_4_transformer(train_data, val_data, test_data, save_path,
                            epochs=25, batch_size=256, lr=5e-4, alpha_width=10.0, patience=12):
    log("\n" + "=" * 80)
    log("STAGE 4: MULTI-TASK PATCHED TRANSFORMER V2 TRAINING")
    log("=" * 80)
    t0 = time.time()
    device = torch.device("cpu")

    X_tr, yw_tr, yc_tr, _ = train_data
    X_va, yw_va, yc_va, _ = val_data
    X_te, yw_te, yc_te, _ = test_data

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(yw_tr, dtype=torch.long),
        torch.tensor(yc_tr, dtype=torch.float32).unsqueeze(1)
    )
    val_ds = TensorDataset(
        torch.tensor(X_va, dtype=torch.float32),
        torch.tensor(yw_va, dtype=torch.long),
        torch.tensor(yc_va, dtype=torch.float32).unsqueeze(1)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MultiTaskPatchedTransformerV2(
        seq_len=X_tr.shape[1], patch_size=10, stem_channels=32,
        embed_dim=128, depth=4, num_heads=4, mlp_ratio=4.0,
        dropout=0.1, drop_path_rate=0.1
    ).to(device)

    n_par = sum(p.numel() for p in model.parameters())
    log(f"  Model: MultiTaskPatchedTransformerV2 | {n_par:,} params | device={device} | threads={torch.get_num_threads()}")
    log(f"  Tokens: {model.patch_embed.num_patches} patches + 1 [CLS] | embed_dim=128, depth=4, heads=4")
    log(f"  Train {len(train_ds):,} samples / {len(train_loader):,} batches per epoch (bs={batch_size}) | "
        f"val {len(val_ds):,} | max {epochs} epochs, patience {patience}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_mae": []}

    epoch_bar = pbar(range(1, epochs + 1), "Transformer epochs", unit="ep", total=epochs, leave=True)
    for epoch in epoch_bar:
        ep_t0 = time.time()
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_ce = 0.0

        batch_bar = pbar(train_loader, f"  ep {epoch:03d}/{epochs:03d} train",
                         unit="b", total=len(train_loader))
        for xb, ywb, ycb in batch_bar:
            optimizer.zero_grad()
            w_logits, c_pred = model(xb)
            l_ce = ce_loss(w_logits, ywb)
            l_mse = mse_loss(c_pred, ycb)
            loss = l_mse + alpha_width * l_ce
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(xb)
            train_mse += l_mse.item() * len(xb)
            train_ce += l_ce.item() * len(xb)
            batch_bar.set_postfix(loss=f"{loss.item():.2f}", mse=f"{l_mse.item():.2f}")

        train_loss /= len(train_ds)
        train_mse /= len(train_ds)
        train_ce /= len(train_ds)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        correct_w = 0
        total_c_err = 0.0

        with torch.no_grad():
            for xb, ywb, ycb in pbar(val_loader, f"  ep {epoch:03d}/{epochs:03d} val",
                                     unit="b", total=len(val_loader)):
                w_logits, c_pred = model(xb)
                l_ce = ce_loss(w_logits, ywb)
                l_mse = mse_loss(c_pred, ycb)
                v_loss = l_mse + alpha_width * l_ce
                val_loss += v_loss.item() * len(xb)
                pred_w = torch.argmax(w_logits, dim=1)
                correct_w += (pred_w == ywb).sum().item()
                total_c_err += torch.abs(c_pred - ycb).sum().item()

        val_loss /= len(val_ds)
        val_acc = (correct_w / len(val_ds)) * 100.0
        val_mae = total_c_err / len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_mae"].append(val_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            star = " * Best"
        else:
            no_improve += 1
            star = ""

        ep_dt = time.time() - ep_t0
        epoch_bar.set_postfix(mae=f"{val_mae:.3f}", acc=f"{val_acc:.2f}%", loss=f"{val_loss:.2f}")
        log(f"Epoch {epoch:03d}/{epochs:03d} | Train: {train_loss:.3f} (MSE:{train_mse:.2f}, CE:{train_ce:.3f}) | "
            f"Val: {val_loss:.3f} (Width Acc: {val_acc:.2f}%, Conc MAE: {val_mae:.3f}) | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | {ep_dt:.1f}s/ep | "
            f"ETA {fmt_eta(epoch, epochs, time.time() - t0)}{star}")

        if no_improve >= patience:
            log(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    checkpoint = {
        "model_state_dict": best_state,
        "args": {"seq_len": X_tr.shape[1], "patch_size": 10, "embed_dim": 128, "depth": 4, "num_heads": 4},
        "val_loss": best_val_loss,
    }
    torch.save(checkpoint, save_path)
    log(f"✓ Saved Transformer checkpoint to: {save_path}")

    # Evaluate on held-out test set
    model.eval()
    with torch.no_grad():
        w_logits, c_pred = model(torch.tensor(X_te, dtype=torch.float32))
        pred_w = torch.argmax(w_logits, dim=1).numpy()
        pred_c = c_pred.squeeze().numpy()

    metrics = compute_multi_metrics(pred_w, yw_te, pred_c, yc_te)
    metrics["train_time_sec"] = round(time.time() - t0, 1)

    log("-" * 80)
    log(f"Patched Transformer v2 Results (Elapsed: {metrics['train_time_sec']}s):")
    log(f"  Width Classification Accuracy: {metrics['Width_Acc_Overall']:.2f}% (7: {metrics['Width_Acc_7']:.1f}%, 9: {metrics['Width_Acc_9']:.1f}%)")
    log(f"  Overall Conc MAE: {metrics['Conc_MAE_Overall']:.3f} | RMSE: {metrics['Conc_RMSE_Overall']:.3f} | Max Err: {metrics['Conc_Max_Error']:.1f}")
    log(f"  7-AGNR MAE: {metrics['Conc_MAE_7']:.3f} | 9-AGNR MAE: {metrics['Conc_MAE_9']:.3f}")
    log("=" * 80 + "\n")

    return metrics, (pred_w, pred_c), history


# ======================================================================
# STAGE 5: UNIFIED COMPARISON & PLOTTING
# ======================================================================

def run_stage_5_summary(test_data, all_preds, all_metrics, mlp_hist, tf_hist, out_dir):
    log("\n" + "=" * 80)
    log("STAGE 5: UNIFIED 4-WAY COMPARATIVE BENCHMARK & PLOTTING")
    log("=" * 80)

    _, yw_te, yc_te, _ = test_data

    # Save metrics JSON
    metrics_file = Path(out_dir) / "multi_width_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # 1. Comparison Summary Table
    log(f"{'Method / Architecture':<28} {'Width Acc (%)':>14} {'Overall MAE':>12} {'Overall RMSE':>13} {'Max Error':>11}")
    log("-" * 80)
    for name, m in all_metrics.items():
        log(f"{name:<28} {m['Width_Acc_Overall']:>13.2f}% {m['Conc_MAE_Overall']:>12.3f} {m['Conc_RMSE_Overall']:>13.3f} {m['Conc_Max_Error']:>11.2f}")
    log("=" * 80)

    log(f"✓ Metrics JSON written to {metrics_file}")

    # 2. Scatter Plot: Predicted vs True
    log("  [1/3] Rendering predicted-vs-true scatter grid...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    plot_items = [
        ("Physical Misfit Baseline", all_preds["Physical Misfit"][1], axes[0, 0], "#7f7f7f"),
        ("XGBoost Regressor", all_preds["XGBoost"][1], axes[0, 1], "#ff7f0e"),
        ("ConductanceMLP", all_preds["ConductanceMLP"][1], axes[1, 0], "#1f77b4"),
        ("Patched Transformer v2", all_preds["Patched Transformer v2"][1], axes[1, 1], "#2ca02c"),
    ]

    mask7 = (yw_te == 0)
    mask9 = (yw_te == 1)

    for name, preds, ax, color in plot_items:
        ax.scatter(yc_te[mask7], preds[mask7], alpha=0.15, s=6, color=color, label="7-AGNR (c ≤ 68)")
        ax.scatter(yc_te[mask9], preds[mask9], alpha=0.15, s=6, color="#d62728", label="9-AGNR (c ≤ 98)")
        ax.plot([0, 100], [0, 100], "k--", lw=1.5, alpha=0.7, label="Ideal")
        mae = np.mean(np.abs(preds - yc_te))
        rmse = np.sqrt(np.mean((preds - yc_te) ** 2))
        ax.set_title(f"{name}\nMAE: {mae:.2f} | RMSE: {rmse:.2f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("True Impurity Concentration $c$", fontsize=10)
        ax.set_ylabel(r"Predicted Concentration $\hat{c}$", fontsize=10)
        ax.set_xlim(0, 102)
        ax.set_ylim(0, 102)
        ax.legend(loc="upper left", frameon=True, fontsize=8)

    plt.tight_layout()
    scatter_path = Path(out_dir) / "multi_width_scatter.png"
    plt.savefig(scatter_path)
    plt.close()

    # 3. Error Distribution Histogram
    log("  [2/3] Rendering error-distribution histogram...")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for name, preds, _, color in plot_items:
        err = preds - yc_te
        ax.hist(err, bins=80, range=(-15, 15), alpha=0.5, label=name, color=color, density=True)
    ax.axvline(0, color="k", linestyle="--", alpha=0.7)
    ax.set_xlabel(r"Prediction Error ($\hat{c} - c$)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title("Multi-Width Error Distributions (7-AGNR & 9-AGNR)", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    dist_path = Path(out_dir) / "multi_width_error_dist.png"
    plt.savefig(dist_path)
    plt.close()

    # 4. Training Loss & Accuracy Curves (if histories available)
    log("  [3/3] Rendering training curves...")
    if mlp_hist and tf_hist:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        ax1.plot(mlp_hist["train_loss"], label="MLP Train Loss", color="#1f77b4", linestyle="--")
        ax1.plot(mlp_hist["val_loss"], label="MLP Val Loss", color="#1f77b4")
        ax1.plot(tf_hist["train_loss"], label="Transformer Train Loss", color="#2ca02c", linestyle="--")
        ax1.plot(tf_hist["val_loss"], label="Transformer Val Loss", color="#2ca02c")
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel(r"Multi-Task Loss ($\mathcal{L}_{\mathrm{MSE}} + 10\cdot\mathcal{L}_{\mathrm{CE}}$)", fontsize=11)
        ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
        ax1.legend(frameon=True, fontsize=9)

        ax2.plot(mlp_hist["val_acc"], label="MLP Width Acc (%)", color="#1f77b4")
        ax2.plot(tf_hist["val_acc"], label="Transformer Width Acc (%)", color="#2ca02c")
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Validation Accuracy (%)", fontsize=11)
        ax2.set_title("Width Identification Accuracy (7 vs 9 AGNR)", fontsize=12, fontweight="bold")
        ax2.legend(frameon=True, fontsize=9)
        plt.tight_layout()
        curve_path = Path(out_dir) / "multi_width_training_curves.png"
        plt.savefig(curve_path)
        plt.close()

    log(f"✓ Saved all visual benchmark plots to {out_dir}\n")


# ======================================================================
# MAIN ORCHESTRATION
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Sequential Multi-Width Pipeline.")
    parser.add_argument("--data-dir", type=str,
                        default="/run/media/shardul/storage/machine_learning/transmission_data/transmission_results/consolidated_data",
                        help="Path to consolidated data folder")
    parser.add_argument("--samples-per-conc", type=int, default=3000,
                        help="Samples per concentration (default: 3000)")
    parser.add_argument("--spectrum-len", type=int, default=150,
                        help="Number of energy channels (default: 150)")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "misfit", "xgb", "mlp", "transformer"],
                        help="Pipeline stage to execute (default: all)")
    parser.add_argument("--mlp-epochs", type=int, default=50,
                        help="Max epochs for MLP (default: 50)")
    parser.add_argument("--tf-epochs", type=int, default=25,
                        help="Max epochs for Transformer (default: 25)")
    parser.add_argument("--log-file", type=str, default="run_sequential_pipeline.log",
                        help="Mirror all stdout logging here ('' to disable). "
                             "Tail it with: tail -f <file>")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable tqdm progress bars (for non-TTY / cron runs)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Where checkpoints/metrics/plots are written (default: script dir). "
                             "Point smoke tests at a scratch dir so they cannot overwrite real results.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve() if args.out_dir else script_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    mlp_save_path = out_dir / "multi_width_mlp.pt"
    tf_save_path = out_dir / "multi_width_transformer.pt"

    # Wire up logging/progress globals before anything prints
    global LOG_PATH, SHOW_PROGRESS
    SHOW_PROGRESS = not args.no_progress
    if args.log_file:
        LOG_PATH = str(out_dir / args.log_file) if not os.path.isabs(args.log_file) else args.log_file
        with open(LOG_PATH, "w") as fh:   # fresh log per run
            fh.write(f"# run_sequential_pipeline.py — started {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    log("=" * 80)
    log(f"{'SEQUENTIAL MULTI-WIDTH EXPERIMENTAL PIPELINE':^80}", stamp=False)
    log("=" * 80)
    log(f"Consolidated Data Directory: {args.data_dir}")
    log(f"Samples Per Concentration:  {args.samples_per_conc} (Total = {83 * args.samples_per_conc:,})")
    log(f"Selected Execution Stage:   {args.stage.upper()}")
    log(f"Epoch budget:               MLP {args.mlp_epochs} | Transformer {args.tf_epochs}")
    log(f"Torch threads / CPU cores:  {torch.get_num_threads()} / {os.cpu_count()}")
    log(f"Log file:                   {LOG_PATH or '(disabled)'}")
    log("=" * 80)

    # 0. Load Data
    log("STAGE 0: Loading consolidated spectra...")
    t_load = time.time()
    train_data, val_data, test_data, p7, p9, concs_7, concs_9 = load_data(
        args.data_dir, samples_per_conc=args.samples_per_conc, spectrum_len=args.spectrum_len
    )
    log(f"✓ Data ready in {time.time() - t_load:.1f}s")

    all_metrics = {}
    all_preds = {}
    mlp_hist = None
    tf_hist = None

    # STAGE 1: Physical Misfit Analysis
    if args.stage in ["all", "misfit"]:
        misfit_metrics, misfit_preds = run_stage_1_misfit(
            train_data, test_data, p7, p9, concs_7, concs_9
        )
        all_metrics["Physical Misfit"] = misfit_metrics
        all_preds["Physical Misfit"] = misfit_preds

    # STAGE 2: XGBoost Baseline
    if args.stage in ["all", "xgb"]:
        xgb_metrics, xgb_preds = run_stage_2_xgboost(
            train_data, test_data, str(out_dir)
        )
        all_metrics["XGBoost"] = xgb_metrics
        all_preds["XGBoost"] = xgb_preds

    # STAGE 3: Multi-Task ConductanceMLP
    if args.stage in ["all", "mlp"]:
        mlp_metrics, mlp_preds, mlp_hist = run_stage_3_mlp(
            train_data, val_data, test_data,
            save_path=str(mlp_save_path),
            epochs=args.mlp_epochs,
            batch_size=256,
            lr=1e-3,
        )
        all_metrics["ConductanceMLP"] = mlp_metrics
        all_preds["ConductanceMLP"] = mlp_preds

    # STAGE 4: Multi-Task Patched Transformer v2
    if args.stage in ["all", "transformer"]:
        tf_metrics, tf_preds, tf_hist = run_stage_4_transformer(
            train_data, val_data, test_data,
            save_path=str(tf_save_path),
            epochs=args.tf_epochs,
            batch_size=256,
            lr=5e-4,
        )
        all_metrics["Patched Transformer v2"] = tf_metrics
        all_preds["Patched Transformer v2"] = tf_preds

    # STAGE 5: Summary & Plots (if all stages ran)
    if args.stage == "all":
        run_stage_5_summary(
            test_data, all_preds, all_metrics, mlp_hist, tf_hist, str(out_dir)
        )

    total = time.time() - _T_START
    log("=" * 80)
    log(f"[DONE] Pipeline finished in {total/60:.1f} min ({total:.0f}s)")
    if LOG_PATH:
        log(f"       Full log written to: {LOG_PATH}")
    log("=" * 80)


if __name__ == "__main__":
    main()
