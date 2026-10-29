"""Train the denoising spectrum autoencoder (INV-AE) on the generated parent-lattice data.

    PYTHONPATH=src python scripts/build/train_autoencoder.py --encoder cnn --extra-conv 1 --max-draws 300000 \
        --threads 8 --out results/inverse_design/ae_cnn

Input: parent_data.to_input_bandtop(raw); target: the pristine channel count on the same grid
(parent_data.pristine_target_bandtop). Trains on the train split (draw index % 20 in 0-17), early-stops on validation
(== 18); the test split is never read. AdamW, mean squared + absolute error, best epoch kept. Writes ae.pt and
history.json (with the baseline: the validation MSE of taking the disordered input itself as the reconstruction).
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

import paths
from shazam.inverse import data
from shazam.inverse.autoencoder import SpectrumAutoencoder
from shazam.inverse.runio import atomic_save, code_version
from tbribbon import parent_data

DEFAULT_DATA = paths.data_root() / "_generated" / "hexagonal_parent_v1"
NEEDED = ("index", "raw", "clean")


def make_arrays(arrays):
    """(x float32 [N, 417], target float32 [N, 416]) of one loaded split."""
    return (torch.as_tensor(parent_data.to_input_bandtop(arrays["raw"])),
            torch.as_tensor(parent_data.pristine_target_bandtop(arrays["clean"], arrays["raw"])))


@torch.no_grad()
def _evaluate(model, x, y, batch):
    model.eval()
    tot = dict(loss=0.0, mse=0.0, l1=0.0)
    for i in range(0, len(x), batch):
        out = model(x[i:i + batch])
        t = y[i:i + batch]
        n = len(t)
        tot["mse"] += torch.nn.functional.mse_loss(out, t).item() * n
        tot["l1"] += torch.nn.functional.l1_loss(out, t).item() * n
    tot = {k: v / len(x) for k, v in tot.items()}
    tot["loss"] = tot["mse"] + tot["l1"]
    return tot


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", choices=("cnn", "mlp"), default="cnn")
    ap.add_argument("--extra-conv", type=int, default=0)
    ap.add_argument("--mlp-hidden", type=int, nargs="+", default=[1024, 512])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-draws", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tr, va = data.load_split(a.data, "train", NEEDED), data.load_split(a.data, "val", NEEDED)
    if a.max_draws is not None:
        tr, va = ({k: v[s["index"] < a.max_draws] for k, v in s.items()} for s in (tr, va))
    xtr, ytr = make_arrays(tr)
    xva, yva = make_arrays(va)
    baseline = float(torch.nn.functional.mse_loss(xva[:, :-1], yva))
    print(f"[ae] {len(xtr):,} train / {len(xva):,} val rows ({time.time() - t0:.0f}s); baseline val MSE "
          f"(disordered input as the reconstruction) {baseline:.4f}", flush=True)
    model = SpectrumAutoencoder(a.encoder, a.extra_conv, tuple(a.mlp_hidden), a.dropout)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=a.weight_decay)
    gen = torch.Generator().manual_seed(a.seed)
    history = dict(train_loss=[], val_loss=[], val_mse=[], val_l1=[], seconds=[], best_epoch=None,
                   baseline_val_mse=baseline, cfg=model.cfg, args=vars(a), code_version=code_version(),
                   data_provenance=data.data_provenance(a.data), n_train=len(xtr), n_val=len(xva))
    best, best_state = float("inf"), None
    for epoch in range(a.epochs):
        t1 = time.time()
        model.train()
        perm = torch.randperm(len(xtr), generator=gen)
        tot = 0.0
        for i in range(0, len(xtr), a.batch):
            idx = perm[i:i + a.batch]
            loss = model.loss(xtr[idx], ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        v = _evaluate(model, xva, yva, a.batch)
        if not (np.isfinite(tot) and np.isfinite(v["loss"])):
            raise FloatingPointError(f"non-finite loss in epoch {epoch}")
        history["train_loss"].append(tot / len(xtr))
        history["val_loss"].append(v["loss"])
        history["val_mse"].append(v["mse"])
        history["val_l1"].append(v["l1"])
        history["seconds"].append(time.time() - t1)
        improved = v["loss"] < best
        if improved:
            best, history["best_epoch"] = v["loss"], epoch
            best_state = copy.deepcopy(model.state_dict())
            atomic_save(out / "ae.pt", model.save)
        atomic_save(out / "history.json", lambda p: p.write_text(json.dumps(history, indent=1)))
        print(f"[ae] epoch {epoch} train {history['train_loss'][-1]:.4f} val {v['loss']:.4f} (mse {v['mse']:.4f}, "
              f"l1 {v['l1']:.4f}) ({history['seconds'][-1]:.0f}s){' *' if improved else ''}", flush=True)
        if epoch - history["best_epoch"] >= a.patience:
            break
    model.load_state_dict(best_state)
    atomic_save(out / "ae.pt", model.save)
    print(f"[ae] best epoch {history['best_epoch']} val {best:.4f}; wrote {out}", flush=True)
    return history


if __name__ == "__main__":
    main()
