#!/usr/bin/env python
"""
mw_common.py — Shared infrastructure for the multi-width AGNR model suite.

The four technique scripts (mw_misfit / mw_xgboost / mw_mlp / mw_transformer)
each run standalone, but all import this module so they share:

  * an identical, deterministic train/val/test split (seed=42),
  * identical metric definitions,
  * identical logging and progress conventions.

That shared split is what makes the four result sets directly comparable.
NOTE: the split is a function of --samples-per-conc and --spectrum-len. If you
change either, re-run *every* technique or the comparison is apples-to-oranges.

Each script writes  <out-dir>/mw_results/<tag>_metrics.json  and
<out-dir>/mw_results/<tag>_preds.npz ; mw_compare.py assembles whatever it finds.
"""

import os
import sys
import json
import math
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

DEFAULT_DATA_DIR = ("/run/media/shardul/storage/machine_learning/transmission_data"
                    "/transmission_results/consolidated_data")

# 7-AGNR: 34 concentrations c=2..68 ; 9-AGNR: 49 concentrations c=2..98
CONCS_7 = np.arange(2, 70, 2)
CONCS_9 = np.arange(2, 100, 2)

LOG_PATH = None
SHOW_PROGRESS = True
_T_START = time.time()


# ======================================================================
# LOGGING & PROGRESS
# ======================================================================

def log(msg="", stamp=True):
    """Print to stdout and mirror to the log file so progress outlives the terminal."""
    body = msg.strip()
    is_rule = body and set(body) <= set("=-")
    if stamp and body and not is_rule:
        prefix = f"[{datetime.now():%H:%M:%S} | +{(time.time() - _T_START)/60:6.1f}m] "
    else:
        prefix = ""
    line = prefix + msg
    print(line, flush=True)
    if LOG_PATH is not None:
        with open(LOG_PATH, "a") as fh:
            fh.write(line + "\n")


def pbar(iterable, desc, unit="it", total=None, leave=False):
    """tqdm wrapper honouring --no-progress. Writes to stderr so stdout logs stay clean."""
    return tqdm(iterable, desc=desc, unit=unit, total=total, leave=leave,
                disable=not SHOW_PROGRESS, dynamic_ncols=True, file=sys.stderr)


def fmt_eta(done, total, elapsed):
    if done <= 0:
        return "??"
    rate = elapsed / done
    return f"{rate*(total-done)/60:.1f}m left ({rate:.1f}s/ep)"


def banner(title):
    log("")
    log("=" * 80)
    log(title)
    log("=" * 80)


# ======================================================================
# CLI PLUMBING
# ======================================================================

def add_common_args(parser):
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Consolidated data folder holding size_7.npy / size_9.npy")
    parser.add_argument("--samples-per-conc", type=int, default=3000,
                        help="Samples per concentration (default 3000). Changing this changes the split.")
    parser.add_argument("--spectrum-len", type=int, default=150,
                        help="Energy channels (default 150). Changing this changes the split.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Where checkpoints/metrics land (default: this script's dir). "
                             "Point test runs at a scratch dir so they cannot clobber real results.")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Mirror stdout here (default <tag>.log inside out-dir; '' disables)")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable tqdm bars (non-TTY / cron runs)")
    parser.add_argument("--snap-grid", action="store_true",
                        help="Snap predicted concentrations onto the physical even-integer grid. "
                             "Off by default so the raw regression quality stays visible.")
    return parser


def setup_run(args, tag):
    """Resolve out-dir, wire logging globals, return (out_dir, results_dir)."""
    global LOG_PATH, SHOW_PROGRESS
    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve() if args.out_dir else script_dir
    results_dir = out_dir / "mw_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    SHOW_PROGRESS = not args.no_progress
    log_file = args.log_file if args.log_file is not None else f"{tag}.log"
    if log_file:
        LOG_PATH = log_file if os.path.isabs(log_file) else str(out_dir / log_file)
        with open(LOG_PATH, "w") as fh:
            fh.write(f"# {tag} — started {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    return out_dir, results_dir


# ======================================================================
# DATA
# ======================================================================

def load_data(consolidated_dir, samples_per_conc=3000, spectrum_len=150, seed=42):
    """Load both widths, normalise by pristine, and split 70/15/15.

    Deterministic given (samples_per_conc, spectrum_len, seed) so every technique
    script sees exactly the same test set.
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]

    p7 = np.load(str(project_root / "7_agnr_pris.npy"))[:spectrum_len].astype(np.float32)
    p9 = np.load(str(project_root / "9_agnr_pris.npy"))[:spectrum_len].astype(np.float32)
    p7_safe = np.where(p7 > 1e-12, p7, 1.0)
    p9_safe = np.where(p9 > 1e-12, p9, 1.0)
    log(f"  Pristine refs: 7-AGNR max={p7.max():.2f} G0 | 9-AGNR max={p9.max():.2f} G0")

    s7 = np.load(os.path.join(consolidated_dir, "size_7.npy"), mmap_mode="r")
    s9 = np.load(os.path.join(consolidated_dir, "size_9.npy"), mmap_mode="r")
    log(f"  Memory-mapped size_7 {s7.shape} | size_9 {s9.shape}")

    X_all, yw_all, yc_all, raw_all = [], [], [], []
    for mmap, concs, pris_safe, wlabel, name in (
        (s7, CONCS_7, p7_safe, 0, "7-AGNR"),
        (s9, CONCS_9, p9_safe, 1, "9-AGNR"),
    ):
        for idx, c in enumerate(pbar(concs, f"Loading {name}", unit="conc")):
            if idx >= mmap.shape[0]:
                break
            raw = np.array(mmap[idx, :samples_per_conc, :spectrum_len], dtype=np.float32)
            X_all.append(np.clip(raw / pris_safe, 0.0, 1.0))
            raw_all.append(raw)
            yw_all.append(np.full(len(raw), wlabel, dtype=np.int64))
            yc_all.append(np.full(len(raw), c, dtype=np.float32))

    X = np.concatenate(X_all, axis=0)
    raw_arr = np.concatenate(raw_all, axis=0)
    y_width = np.concatenate(yw_all, axis=0)
    y_conc = np.concatenate(yc_all, axis=0)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))
    n_tr, n_va = int(len(X) * 0.70), int(len(X) * 0.15)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    log(f"  Split: train {len(tr):,} | val {len(va):,} | test {len(te):,} "
        f"(seed={seed}, identical across all technique scripts)")
    log(f"  Footprint X={X.nbytes/1e9:.2f} GB | conc range [{y_conc.min():.0f}, {y_conc.max():.0f}] "
        f"| 7-AGNR {(y_width==0).sum():,} / 9-AGNR {(y_width==1).sum():,}")

    pack = lambda i: (X[i], y_width[i], y_conc[i], raw_arr[i])
    return pack(tr), pack(va), pack(te), p7, p9


class TargetScaler:
    """IMPROVEMENT 1 — standardise the regression target.

    Concentrations span 2..98, so raw-scale MSE starts near E[c^2] ~ 2500 and the
    output layer has to emit ~98 from LayerNorm'd unit-scale features. That forces
    large final-layer weights which weight decay then fights. Trees are invariant
    to target scale; networks are not. Fit on train only, invert for reporting.
    """

    def __init__(self, y):
        self.mean = float(np.mean(y))
        self.std = float(np.std(y)) or 1.0

    def transform(self, y):
        return (np.asarray(y, np.float32) - self.mean) / self.std

    def inverse(self, y):
        if torch.is_tensor(y):
            return y * self.std + self.mean
        return np.asarray(y, np.float32) * self.std + self.mean

    def as_dict(self):
        return {"mean": self.mean, "std": self.std}


# ======================================================================
# METRICS
# ======================================================================

def compute_multi_metrics(w_preds, w_true, c_preds, c_true):
    w_preds = np.asarray(w_preds)
    err = np.asarray(c_preds, float) - np.asarray(c_true, float)
    m7, m9 = (w_true == 0), (w_true == 1)
    f = lambda cond, arr: float(arr[cond].mean()) if np.any(cond) else 0.0
    return {
        "Width_Acc_Overall": float(np.mean(w_preds == w_true)) * 100.0,
        "Width_Acc_7": f(m7, (w_preds == w_true).astype(float)) * 100.0,
        "Width_Acc_9": f(m9, (w_preds == w_true).astype(float)) * 100.0,
        "Conc_MAE_Overall": float(np.mean(np.abs(err))),
        "Conc_RMSE_Overall": float(np.sqrt(np.mean(err ** 2))),
        "Conc_Max_Error": float(np.max(np.abs(err))),
        "Conc_MAE_7": f(m7, np.abs(err)),
        "Conc_RMSE_7": float(np.sqrt(np.mean(err[m7] ** 2))) if np.any(m7) else 0.0,
        "Conc_MAE_9": f(m9, np.abs(err)),
        "Conc_RMSE_9": float(np.sqrt(np.mean(err[m9] ** 2))) if np.any(m9) else 0.0,
    }


def snap_to_grid(pred_c, pred_w):
    """Snap predictions onto the physical even-integer concentration grid.

    Targets only ever take values in {2,4,...,68} (7-AGNR) or {2,4,...,98} (9-AGNR);
    plain regression ignores that. Gain is modest when MAE exceeds the grid spacing
    of 2, so this stays opt-in via --snap-grid.
    """
    out = np.asarray(pred_c, np.float32).copy()
    for wlabel, grid in ((0, CONCS_7), (1, CONCS_9)):
        m = (np.asarray(pred_w) == wlabel)
        if np.any(m):
            g = grid.astype(np.float32)
            out[m] = g[np.abs(out[m][:, None] - g[None, :]).argmin(axis=1)]
    return out


def report_metrics(name, metrics, elapsed_key=None):
    log("-" * 80)
    tail = f" (Elapsed: {metrics[elapsed_key]}s)" if elapsed_key and elapsed_key in metrics else ""
    log(f"{name} Results{tail}:")
    log(f"  Width Accuracy:   {metrics['Width_Acc_Overall']:.2f}% "
        f"(7: {metrics['Width_Acc_7']:.1f}%, 9: {metrics['Width_Acc_9']:.1f}%)")
    log(f"  Conc MAE:         {metrics['Conc_MAE_Overall']:.3f} | "
        f"RMSE: {metrics['Conc_RMSE_Overall']:.3f} | Max: {metrics['Conc_Max_Error']:.2f}")
    log(f"  Per width MAE:    7-AGNR {metrics['Conc_MAE_7']:.3f} | 9-AGNR {metrics['Conc_MAE_9']:.3f}")
    log("=" * 80)


def save_result(results_dir, tag, display_name, metrics, pred_w, pred_c,
                y_width, y_conc, history=None, extra=None):
    """Persist metrics + predictions so mw_compare.py can assemble the benchmark."""
    payload = {"display_name": display_name, "metrics": metrics,
               "written": datetime.now().isoformat(timespec="seconds")}
    if extra:
        payload.update(extra)
    mpath = Path(results_dir) / f"{tag}_metrics.json"
    with open(mpath, "w") as fh:
        json.dump(payload, fh, indent=2)

    ppath = Path(results_dir) / f"{tag}_preds.npz"
    arrays = {"pred_w": np.asarray(pred_w), "pred_c": np.asarray(pred_c),
              "true_w": np.asarray(y_width), "true_c": np.asarray(y_conc)}
    if history:
        for k, v in history.items():
            arrays[f"hist_{k}"] = np.asarray(v, dtype=np.float32)
    np.savez_compressed(ppath, **arrays)
    log(f"✓ Wrote {mpath.name} and {ppath.name} to {results_dir}")


# ======================================================================
# SHARED MULTI-TASK TRAINER  (used by both mw_mlp.py and mw_transformer.py)
# ======================================================================

def _make_loader(X, yw, yc_norm, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(yw, dtype=torch.long),
                       torch.tensor(yc_norm, dtype=torch.float32).unsqueeze(1))
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def fit_multitask(model, train_data, val_data, scaler, *, tag, epochs, batch_size,
                  lr, alpha_width, patience, warmup_epochs=0, weight_decay=1e-4,
                  grad_clip=None, huber_beta=1.0, device="cpu"):
    """Train a dual-head model. Returns (best_state, history, best_val_mae).

    Embeds several of the improvements:
      * targets are trained in normalised space (TargetScaler)
      * IMPROVEMENT 5 — Huber/SmoothL1 regression loss, and the best checkpoint is
        selected on validation MAE *in original units*, i.e. the metric actually
        reported, rather than on the MSE-dominated composite loss.
      * IMPROVEMENT 7 — optional linear LR warmup ahead of the cosine decay.
    """
    X_tr, yw_tr, yc_tr, _ = train_data
    X_va, yw_va, yc_va, _ = val_data

    train_ds, train_loader = _make_loader(X_tr, yw_tr, scaler.transform(yc_tr), batch_size, True)
    val_ds, val_loader = _make_loader(X_va, yw_va, scaler.transform(yc_va), batch_size, False)

    n_par = sum(p.numel() for p in model.parameters())
    log(f"  Params: {n_par:,} | device={device} | torch threads={torch.get_num_threads()}")
    log(f"  Train {len(train_ds):,} ({len(train_loader):,} batches, bs={batch_size}) | val {len(val_ds):,}")
    log(f"  epochs={epochs}, lr={lr:g}, warmup={warmup_epochs}, alpha_width={alpha_width:g}, "
        f"patience={patience}, wd={weight_decay:g}")
    log(f"  Target scaler: mean={scaler.mean:.3f}, std={scaler.std:.3f} (training in normalised space)")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(ep):
        if warmup_epochs and ep < warmup_epochs:
            return (ep + 1) / float(warmup_epochs)
        prog = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ce_loss = nn.CrossEntropyLoss()
    reg_loss = nn.SmoothL1Loss(beta=huber_beta)

    best_mae, best_state, no_improve = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_mae": []}
    t0 = time.time()

    epoch_bar = pbar(range(1, epochs + 1), f"{tag} epochs", unit="ep", total=epochs, leave=True)
    for epoch in epoch_bar:
        ep_t0 = time.time()
        model.train()
        tr_loss = tr_reg = tr_ce = 0.0

        batch_bar = pbar(train_loader, f"  ep {epoch:03d}/{epochs:03d} train",
                         unit="b", total=len(train_loader))
        for xb, ywb, ycb in batch_bar:
            optimizer.zero_grad()
            w_logits, c_pred = model(xb)
            l_ce = ce_loss(w_logits, ywb)
            l_reg = reg_loss(c_pred, ycb)
            loss = l_reg + alpha_width * l_ce
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            tr_loss += loss.item() * len(xb)
            tr_reg += l_reg.item() * len(xb)
            tr_ce += l_ce.item() * len(xb)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}", reg=f"{l_reg.item():.4f}")

        tr_loss /= len(train_ds); tr_reg /= len(train_ds); tr_ce /= len(train_ds)
        scheduler.step()

        model.eval()
        va_loss, correct, abs_err_sum = 0.0, 0, 0.0
        with torch.no_grad():
            for xb, ywb, ycb in pbar(val_loader, f"  ep {epoch:03d}/{epochs:03d} val",
                                     unit="b", total=len(val_loader)):
                w_logits, c_pred = model(xb)
                loss = reg_loss(c_pred, ycb) + alpha_width * ce_loss(w_logits, ywb)
                va_loss += loss.item() * len(xb)
                correct += (torch.argmax(w_logits, dim=1) == ywb).sum().item()
                # MAE reported in ORIGINAL units, not normalised ones
                abs_err_sum += torch.abs(scaler.inverse(c_pred) - scaler.inverse(ycb)).sum().item()

        va_loss /= len(val_ds)
        va_acc = correct / len(val_ds) * 100.0
        va_mae = abs_err_sum / len(val_ds)
        for k, v in zip(history, (tr_loss, va_loss, va_acc, va_mae)):
            history[k].append(v)

        if va_mae < best_mae:                      # selection on the reported metric
            best_mae = va_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve, star = 0, " * Best"
        else:
            no_improve += 1
            star = ""

        epoch_bar.set_postfix(mae=f"{va_mae:.3f}", acc=f"{va_acc:.2f}%", best=f"{best_mae:.3f}")
        log(f"Epoch {epoch:03d}/{epochs:03d} | Train {tr_loss:.4f} (reg {tr_reg:.4f}, ce {tr_ce:.4f}) | "
            f"Val {va_loss:.4f} (Acc {va_acc:.2f}%, MAE {va_mae:.3f}) | "
            f"LR {scheduler.get_last_lr()[0]:.2e} | {time.time()-ep_t0:.1f}s/ep | "
            f"ETA {fmt_eta(epoch, epochs, time.time()-t0)}{star}")

        if no_improve >= patience:
            log(f"Early stopping at epoch {epoch} (no val-MAE improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"  Best validation MAE: {best_mae:.4f}")
    return best_state, history, best_mae


@torch.no_grad()
def predict_torch(model, X, scaler, batch_size=1024, desc="predict"):
    """Batched inference — avoids the multi-GB spike of one 37k-row forward pass."""
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    pw, pc = [], []
    for s in pbar(range(0, len(Xt), batch_size), desc, unit="b",
                  total=(len(Xt) + batch_size - 1) // batch_size):
        w_logits, c_pred = model(Xt[s:s + batch_size])
        pw.append(torch.argmax(w_logits, dim=1).numpy())
        pc.append(scaler.inverse(c_pred.squeeze(1)).numpy())
    return np.concatenate(pw), np.concatenate(pc)
