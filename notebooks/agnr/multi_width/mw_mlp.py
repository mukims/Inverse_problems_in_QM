#!/usr/bin/env python
"""
mw_mlp.py — Technique 3/4: Multi-task ConductanceMLP (dual-head, purely data-driven).

No physics-informed loss term: the premise is that a sufficiently expressive network
learns scattering behaviour directly from T(E). The shared backbone feeds a width
classifier head and a concentration regression head.

Improvements carried over from the BUILD-06 post-mortem
-------------------------------------------------------
1. Target normalisation      — handled by mw_common.TargetScaler.
2. alpha_width rebalanced    — with normalised targets both losses are O(1), so the
                               old alpha=10 (chosen against raw-scale MSE ~2500, where
                               10*CE contributed ~0.3% of the loss) is now alpha=1.
3. Width-conditioned head    — the concentration head receives the width-head softmax,
                               mirroring the extra feature XGBoost gets. 7- and 9-AGNR
                               cover different ranges (<=68 vs <=98), so this matters.
4. Less regularisation       — BUILD-06 showed train loss ABOVE val loss and both still
                               falling at epoch 50: underfitting, not overfitting. So
                               dropout 0.2 -> 0.1, input noise off, more epochs.
5. Huber + val-MAE selection — see mw_common.fit_multitask.

Usage
-----
    python mw_mlp.py
    python mw_mlp.py --epochs 120 --hidden-dims 512 256 128 --dropout 0.05
    python mw_mlp.py --no-width-condition        # ablation
"""

import time
import argparse

import torch
import torch.nn as nn

import mw_common as mw
from mw_common import log, banner

TAG = "mlp"
DISPLAY = "ConductanceMLP"


class MultiTaskConductanceMLP(nn.Module):
    """Fully-connected dual-head network over the normalised spectrum."""

    def __init__(self, input_length=150, hidden_dims=(256, 128, 64), dropout=0.1,
                 noise_std=0.0, width_condition=True):
        super().__init__()
        self.noise_std = noise_std
        self.width_condition = width_condition

        layers, in_dim = [], input_length
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h),
                       nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        self.width_head = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2), nn.Linear(32, 2),
        )
        # IMPROVEMENT 3: concentration head also sees the width posterior
        conc_in = in_dim + (2 if width_condition else 0)
        self.conc_head = nn.Sequential(
            nn.Linear(conc_in, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2), nn.Linear(64, 1),
        )

    def forward(self, x):
        if self.training and self.noise_std > 0:
            x = x * (1.0 + torch.randn_like(x) * self.noise_std)
        feat = self.backbone(x)
        w_logits = self.width_head(feat)
        if self.width_condition:
            feat = torch.cat([feat, torch.softmax(w_logits, dim=1)], dim=1)
        return w_logits, self.conc_head(feat)


def run(args, train_data, val_data, test_data, out_dir, results_dir):
    banner("TECHNIQUE 3/4: MULTI-TASK CONDUCTANCE MLP")
    t0 = time.time()

    X_tr, _, yc_tr, _ = train_data
    _, yw_te, yc_te, _ = test_data
    scaler = mw.TargetScaler(yc_tr)          # IMPROVEMENT 1 (fit on train only)

    model = MultiTaskConductanceMLP(
        input_length=X_tr.shape[1], hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout, noise_std=args.noise_std,
        width_condition=not args.no_width_condition,
    )
    log(f"  Architecture: {list(args.hidden_dims)} | dropout={args.dropout} | "
        f"noise_std={args.noise_std} | width_condition={not args.no_width_condition}")

    best_state, history, best_mae = mw.fit_multitask(
        model, train_data, val_data, scaler, tag="MLP",
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        alpha_width=args.alpha_width, patience=args.patience,
        warmup_epochs=args.warmup_epochs, weight_decay=args.weight_decay,
        huber_beta=args.huber_beta,
        lr_schedule=args.lr_schedule, sched_epochs=args.sched_epochs,
        plateau_factor=args.plateau_factor, plateau_patience=args.plateau_patience,
        swa=args.swa, swa_start=args.swa_start, swa_lr=args.swa_lr,
        swa_anneal=args.swa_anneal,
    )

    ckpt_path = out_dir / "mw_mlp.pt"
    torch.save({
        "model_state_dict": best_state,
        "args": {"input_length": X_tr.shape[1], "hidden_dims": list(args.hidden_dims),
                 "dropout": args.dropout, "noise_std": args.noise_std,
                 "width_condition": not args.no_width_condition},
        "target_scaler": scaler.as_dict(),
        "best_val_mae": best_mae,
    }, ckpt_path)
    log(f"✓ Saved checkpoint -> {ckpt_path.name}")

    pred_w, pred_c = mw.predict_torch(model, test_data[0], scaler, desc="MLP test")
    if args.snap_grid:
        pred_c = mw.snap_to_grid(pred_c, pred_w)
        log("  Applied even-integer grid snapping")

    metrics = mw.compute_multi_metrics(pred_w, yw_te, pred_c, yc_te)
    metrics["train_time_sec"] = round(time.time() - t0, 1)
    metrics["best_val_mae"] = round(best_mae, 4)
    mw.report_metrics(DISPLAY, metrics, "train_time_sec")

    mw.save_result(results_dir, TAG, DISPLAY, metrics, pred_w, pred_c, yw_te, yc_te,
                   history=history,
                   extra={"snap_grid": bool(args.snap_grid),
                          "width_condition": not args.no_width_condition})
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Multi-task ConductanceMLP.")
    mw.add_common_args(parser)
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Max epochs (default 100; BUILD-06 was still improving at 50)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="MLPs rarely need warmup; kept for parity with the transformer")
    parser.add_argument("--alpha-width", type=float, default=1.0,
                        help="Weight on the width CE loss (default 1.0; see IMPROVEMENT 2)")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--lr-schedule", choices=["cosine", "plateau", "none"], default="cosine",
                        help="cosine: decay over --sched-epochs. plateau: ReduceLROnPlateau on val MAE "
                             "(horizon-independent - use this when pairing a big --epochs with early stopping). "
                             "none: constant LR.")
    parser.add_argument("--sched-epochs", type=int, default=None,
                        help="Cosine horizon (T_max). Defaults to --epochs. Set this when --epochs is only "
                             "a safety cap, otherwise the LR barely decays before early stopping fires.")
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=8)
    parser.add_argument("--swa", action="store_true",
                        help="Stochastic Weight Averaging: average weights over the tail of "
                             "training instead of taking one epoch's snapshot. Targets the "
                             "epoch-to-epoch val-MAE oscillation this model shows. The averaged "
                             "model is kept only if it actually validates better.")
    parser.add_argument("--swa-start", type=int, default=None,
                        help="Epoch at which averaging begins (default: 75%% of --epochs). "
                             "Must be reached before early stopping fires.")
    parser.add_argument("--swa-lr", type=float, default=None,
                        help="Constant LR during the SWA phase (default: --lr / 10)")
    parser.add_argument("--swa-anneal", type=int, default=5,
                        help="Epochs to anneal into swa_lr (default 5)")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Multiplicative input noise; 0 disables (was 0.01)")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64],
                        help="Backbone widths. No overfitting was observed, so widening is safe.")
    parser.add_argument("--no-width-condition", action="store_true",
                        help="Ablation: hide the width posterior from the concentration head")
    args = parser.parse_args()

    out_dir, results_dir = mw.setup_run(args, TAG)
    banner("MULTI-WIDTH SUITE — CONDUCTANCE MLP")
    log(f"Data dir: {args.data_dir}")
    log(f"Out dir:  {out_dir}")

    train_data, val_data, test_data, _, _ = mw.load_data(
        args.data_dir, args.samples_per_conc, args.spectrum_len)

    run(args, train_data, val_data, test_data, out_dir, results_dir)
    log(f"[DONE] MLP finished in {(time.time() - mw._T_START)/60:.1f} min")


if __name__ == "__main__":
    main()
