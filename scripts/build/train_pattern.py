"""Train the lattice-type classifier (shazam.inverse.pattern): bandtop spectrum -> one of 7 site-type patterns.

    TRANSMISSION_DATA=... PYTHONPATH=src python scripts/build/train_pattern.py --threads 16 \\
        --out results/inverse_design/pattern_v1

Train on the train split (draw index % 20 in 0-17), stop early on the validation cross-entropy (index % 20 == 18); the
test split is never read. Writes model.pt (best epoch) and history.json, whose `val_scores` hold the best model's
validation top-1, top-1 by true pattern and the confusion matrix (rows: truth, columns: prediction, order NAMES).

Regularisation (INV-PATTERN-2), all off by default: --dropout (embedding and head hidden layer), --label-smoothing (the
training loss only; validation and early stopping use plain cross-entropy) and --noise (Gaussian noise added to each
training spectrum, standard deviation = noise x that spectrum's RMS; the band-top value is left alone).
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import paths
from shazam.inverse import data
from shazam.inverse.pattern import NAMES, N_PATTERNS, PatternModel, pattern_labels
from shazam.inverse.runio import atomic_save, code_version
from tbribbon.parent_data import to_input_bandtop

DEFAULT_DATA = paths.data_root() / "_generated" / "hexagonal_parent_v1"


def add_noise(x, noise, gen):
    """x with Gaussian noise of standard deviation noise x row RMS on the spectrum (not on the band-top value)."""
    if not noise:
        return x
    spec = x[:, :-1]
    rms = spec.pow(2).mean(1, keepdim=True).sqrt()
    return torch.cat([spec + noise * rms * torch.randn(spec.shape, generator=gen), x[:, -1:]], 1)


def tensors(arrays):
    return (torch.as_tensor(to_input_bandtop(arrays["raw"]), dtype=torch.float32),
            torch.as_tensor(pattern_labels(arrays["on"])))


@torch.no_grad()
def val_loss(model, x, y, batch):
    model.eval()
    return sum(F.cross_entropy(model.logits(x[i:i + batch]), y[i:i + batch], reduction="sum").item()
               for i in range(0, len(x), batch)) / len(x)


def scores(model, x, y):
    pred = model.predict_proba(x).argmax(1).numpy()
    y = y.numpy()
    conf = np.zeros((N_PATTERNS, N_PATTERNS), int)
    np.add.at(conf, (y, pred), 1)
    return dict(n=int(len(y)), top1=float((pred == y).mean()), names=list(NAMES), confusion=conf.tolist(),
                by_pattern={NAMES[k]: float((pred[y == k] == k).mean()) for k in range(N_PATTERNS) if (y == k).any()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--extra-conv", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--noise", type=float, default=0.0, help="training-input noise, as a fraction of each row's RMS")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max-draws", type=int, default=None, help="use only draws with index below this")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tr, va = (data.load_split(a.data, s, ("index", "raw", "on")) for s in ("train", "val"))
    if a.max_draws is not None:
        tr, va = ({k: v[s["index"] < a.max_draws] for k, v in s.items()} for s in (tr, va))
    (xt, yt), (xv, yv) = tensors(tr), tensors(va)
    print(f"[pattern] {len(xt):,} train / {len(xv):,} val rows ({time.time() - t0:.0f}s)", flush=True)

    model = PatternModel(channels=xt.shape[1], embed=a.embed, extra_conv=a.extra_conv,
                         dropout=a.dropout)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=a.weight_decay)
    gen = torch.Generator().manual_seed(a.seed)
    h = dict(train_loss=[], val_loss=[], seconds=[], best_epoch=None, n_train=len(xt), n_val=len(xv),
             cfg=dict(vars(a)), model=model.cfg, data=str(a.data), code_version=code_version(),
             data_provenance=data.data_provenance(a.data))
    best, best_state = float("inf"), None
    for epoch in range(a.epochs):
        t1 = time.time()
        model.train()
        tot = 0.0
        for idx in torch.randperm(len(xt), generator=gen).split(a.batch):
            loss = model.loss(add_noise(xt[idx], a.noise, gen), yt[idx], a.label_smoothing)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        v = val_loss(model, xv, yv, 2000)
        if not np.isfinite(v):
            raise FloatingPointError(f"non-finite validation loss in epoch {epoch}")
        h["train_loss"].append(tot / len(xt))
        h["val_loss"].append(v)
        h["seconds"].append(time.time() - t1)
        improved = v < best
        if improved:
            best, h["best_epoch"], best_state = v, epoch, copy.deepcopy(model.state_dict())
            atomic_save(out / "model.pt", model.save)
        atomic_save(out / "history.json", lambda p: p.write_text(json.dumps(h, indent=1)))
        print(f"[pattern] epoch {epoch} train {h['train_loss'][-1]:.4f} val {v:.4f} ({h['seconds'][-1]:.0f}s)"
              f"{' *' if improved else ''}", flush=True)
        if epoch - h["best_epoch"] >= a.patience:
            break
    model.load_state_dict(best_state)
    h["val_scores"] = scores(model, xv, yv)
    atomic_save(out / "model.pt", model.save)
    atomic_save(out / "history.json", lambda p: p.write_text(json.dumps(h, indent=1)))
    s = h["val_scores"]
    print(f"[pattern] best epoch {h['best_epoch']} val {best:.4f}; top-1 {s['top1']:.3f}; "
          + ", ".join(f"{k} {v:.3f}" for k, v in s["by_pattern"].items()) + f"; wrote {out}", flush=True)
    return model, h


if __name__ == "__main__":
    main()
