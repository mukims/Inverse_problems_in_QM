#!/usr/bin/env python
"""
mw_transformer.py — Technique 4/4: Multi-task 1D-Patched Transformer.

ConvStem -> patch embedding -> pre-norm self-attention blocks -> [CLS] dual heads.
Purely data-driven: long-range attention is expected to correlate distant spectral
features (band-edge shifts against Fano dips) without any hand-built physics loss.

Improvements carried over from the BUILD-06 post-mortem
-------------------------------------------------------
1. Target normalisation      — mw_common.TargetScaler.
2. alpha_width 10 -> 1       — rebalanced for normalised targets.
3. Width-conditioned head    — [CLS] features are concatenated with the width softmax.
4. Less regularisation       — BUILD-06 ran all 25 epochs without early stopping and
                               val loss was still falling, so: more epochs, dropout
                               0.1 -> 0.05, drop-path 0.1 -> 0.05.
5. Huber + val-MAE selection — mw_common.fit_multitask.
6. Positional embedding fix  — was `norm(x) + pos` with pos init std 0.02 added to
                               unit-variance LayerNorm'd tokens, i.e. ~2% of token
                               magnitude. Energy position is critical physics here
                               (band edges sit at specific E), so pos is now added
                               BEFORE the norm and initialised at std 0.10.
7. LR warmup                 — linear warmup ahead of cosine decay; the old schedule
                               ran cosine from step 0, a classic early-instability source.

This is by far the slowest technique on CPU (BUILD-06: 64 min for 25 epochs).
Budget accordingly, or trim with --epochs / --depth.

Usage
-----
    python mw_transformer.py
    python mw_transformer.py --epochs 40 --warmup-epochs 3
    python mw_transformer.py --pos-embed-std 0.02 --pos-after-norm   # old behaviour
"""

import time
import argparse

import torch
import torch.nn as nn

import mw_common as mw
from mw_common import log, banner

TAG = "transformer"
DISPLAY = "Patched Transformer v2"


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        mask = torch.bernoulli(torch.full((x.shape[0],) + (1,) * (x.ndim - 1), keep,
                                          device=x.device))
        return x * mask / keep


class ConvStem(nn.Module):
    def __init__(self, out_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(16, out_channels, kernel_size=5, padding=2), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class PatchEmbedding1D(nn.Module):
    """IMPROVEMENT 6 — positional information must survive the LayerNorm.

    Old: `return self.norm(x) + self.pos_embed`, with pos_embed ~ N(0, 0.02) added to
    unit-variance normalised tokens, leaving position at roughly 2% of token magnitude.
    New default adds position BEFORE the norm at std 0.10, so the norm rescales content
    and position jointly. `pos_after_norm=True` restores the old behaviour for ablation.
    """

    def __init__(self, seq_len=150, patch_size=10, in_channels=32, embed_dim=128,
                 pos_embed_std=0.10, pos_after_norm=False):
        super().__init__()
        self.num_patches = seq_len // patch_size
        self.pos_after_norm = pos_after_norm
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * pos_embed_std)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        x = self.proj(x).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.pos_after_norm:
            return self.norm(x) + self.pos_embed
        return self.norm(x + self.pos_embed)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4.0, dropout=0.05, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim), nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.drop_path1(attn_out)
        return x + self.drop_path2(self.mlp(self.norm2(x)))


class MultiTaskPatchedTransformerV2(nn.Module):
    def __init__(self, seq_len=150, patch_size=10, stem_channels=32, embed_dim=128,
                 depth=4, num_heads=4, mlp_ratio=4.0, dropout=0.05, drop_path_rate=0.05,
                 pos_embed_std=0.10, pos_after_norm=False, width_condition=True):
        super().__init__()
        self.width_condition = width_condition
        self.stem = ConvStem(out_channels=stem_channels)
        self.patch_embed = PatchEmbedding1D(
            seq_len=seq_len, patch_size=patch_size, in_channels=stem_channels,
            embed_dim=embed_dim, pos_embed_std=pos_embed_std, pos_after_norm=pos_after_norm)
        dpr = [v.item() for v in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.width_head = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 2))
        conc_in = embed_dim + (2 if width_condition else 0)
        self.conc_head = nn.Sequential(
            nn.Linear(conc_in, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        tokens = self.patch_embed(self.stem(x))
        for block in self.blocks:
            tokens = block(tokens)
        cls = self.norm(tokens)[:, 0]
        w_logits = self.width_head(cls)
        if self.width_condition:                       # IMPROVEMENT 3
            cls = torch.cat([cls, torch.softmax(w_logits, dim=1)], dim=1)
        return w_logits, self.conc_head(cls)


def run(args, train_data, val_data, test_data, out_dir, results_dir):
    banner("TECHNIQUE 4/4: MULTI-TASK PATCHED TRANSFORMER V2")
    t0 = time.time()

    X_tr, _, yc_tr, _ = train_data
    _, yw_te, yc_te, _ = test_data
    scaler = mw.TargetScaler(yc_tr)                    # IMPROVEMENT 1

    model = MultiTaskPatchedTransformerV2(
        seq_len=X_tr.shape[1], patch_size=args.patch_size, stem_channels=args.stem_channels,
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
        dropout=args.dropout, drop_path_rate=args.drop_path,
        pos_embed_std=args.pos_embed_std, pos_after_norm=args.pos_after_norm,
        width_condition=not args.no_width_condition,
    )
    log(f"  Tokens: {model.patch_embed.num_patches} patches + 1 [CLS] | embed={args.embed_dim}, "
        f"depth={args.depth}, heads={args.num_heads}")
    log(f"  dropout={args.dropout}, drop_path={args.drop_path}, "
        f"pos_embed_std={args.pos_embed_std}, pos_after_norm={args.pos_after_norm}, "
        f"width_condition={not args.no_width_condition}")

    best_state, history, best_mae = mw.fit_multitask(
        model, train_data, val_data, scaler, tag="Transformer",
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        alpha_width=args.alpha_width, patience=args.patience,
        warmup_epochs=args.warmup_epochs, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, huber_beta=args.huber_beta,
    )

    ckpt_path = out_dir / "mw_transformer.pt"
    torch.save({
        "model_state_dict": best_state,
        "args": {"seq_len": X_tr.shape[1], "patch_size": args.patch_size,
                 "stem_channels": args.stem_channels, "embed_dim": args.embed_dim,
                 "depth": args.depth, "num_heads": args.num_heads,
                 "dropout": args.dropout, "drop_path_rate": args.drop_path,
                 "pos_embed_std": args.pos_embed_std, "pos_after_norm": args.pos_after_norm,
                 "width_condition": not args.no_width_condition},
        "target_scaler": scaler.as_dict(),
        "best_val_mae": best_mae,
    }, ckpt_path)
    log(f"✓ Saved checkpoint -> {ckpt_path.name}")

    pred_w, pred_c = mw.predict_torch(model, test_data[0], scaler, desc="TF test")
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
    parser = argparse.ArgumentParser(description="Multi-task 1D-Patched Transformer.")
    mw.add_common_args(parser)
    parser.add_argument("--epochs", type=int, default=40,
                        help="Max epochs (default 40; BUILD-06's 25 never early-stopped)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Linear LR warmup before cosine decay (IMPROVEMENT 7)")
    parser.add_argument("--alpha-width", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--stem-channels", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--drop-path", type=float, default=0.05)
    parser.add_argument("--pos-embed-std", type=float, default=0.10,
                        help="Positional embedding init scale (was 0.02; see IMPROVEMENT 6)")
    parser.add_argument("--pos-after-norm", action="store_true",
                        help="Ablation: restore the old `norm(x) + pos` ordering")
    parser.add_argument("--no-width-condition", action="store_true",
                        help="Ablation: hide the width posterior from the concentration head")
    args = parser.parse_args()

    out_dir, results_dir = mw.setup_run(args, TAG)
    banner("MULTI-WIDTH SUITE — PATCHED TRANSFORMER V2")
    log(f"Data dir: {args.data_dir}")
    log(f"Out dir:  {out_dir}")

    train_data, val_data, test_data, _, _ = mw.load_data(
        args.data_dir, args.samples_per_conc, args.spectrum_len)

    run(args, train_data, val_data, test_data, out_dir, results_dir)
    log(f"[DONE] transformer finished in {(time.time() - mw._T_START)/60:.1f} min")


if __name__ == "__main__":
    main()
