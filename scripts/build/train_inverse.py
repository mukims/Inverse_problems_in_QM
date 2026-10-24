"""Train the inverse model (spec section 6) on the generated parent-lattice data.

    PYTHONPATH=src python scripts/build/train_inverse.py --exclude none --threads 16

Reads the train and validation splits (draw index % 20 in 0-17 and 18; the test split is never read here), encodes each
draw once, and trains with Adam and early stopping on the mean validation loss. `--exclude {hbn,kagome,silicene}` drops
the rows near that material from the training AND the validation rows (the hidden material must not steer early
stopping either). Writes model.pt (best epoch) and history.json to --out: by default model_v1 for --exclude none and
hidden_<material> otherwise (an excluded run never writes into model_v1). history.json records this run's code version and
the data's provenance.
"""
import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

import paths
from shazam.inverse import data
from shazam.inverse.encoding import STATES, active_slots, encode
from shazam.inverse.model import InverseModel
from shazam.inverse.runio import atomic_save, code_version
from tbribbon.parent_prior import excluded

DEFAULT_DATA = paths.data_root() / "_generated" / "hexagonal_parent_v1"
DEFAULT_OUT = paths.results_dir("inverse_design", "model_v1")


def default_out(exclude):
    """model_v1 for the full model, hidden_<material> for a run with --exclude <material>."""
    return DEFAULT_OUT if exclude == "none" else paths.results_dir("inverse_design", f"hidden_{exclude}")


@dataclass
class TrainConfig:
    """Training settings (a dataclass; `train` also accepts a plain dict with the same keys)."""
    epochs: int = 200
    batch: int = 512
    lr: float = 1e-3
    patience: int = 10
    seed: int = 0
    exclude: str = "none"                      # none, hbn, kagome or silicene


NEEDED = ("index", "x", "on", "lead", "params", "edge", "width", "density")      # the arrays `train` reads


def _config(cfg):
    return cfg if isinstance(cfg, TrainConfig) else TrainConfig(**cfg)


def prepare(arrays, hide="none"):
    """(tensors, excluded count): x float32 [N, 416], state int64 [N], z float32 [N, 14], mask bool [N, 14].

    Rows in the neighbourhood of `hide` are dropped; every remaining draw is encoded once."""
    draws = data.draws(arrays, slice(None))
    keep = np.array([hide == "none" or not excluded(d, hide) for d in draws], dtype=bool)
    enc = [encode(d) for d, k in zip(draws, keep) if k]
    if not enc:
        raise ValueError(f"no rows left after excluding {hide!r}")
    state, z, mask = zip(*enc)
    tensors = dict(x=torch.as_tensor(np.asarray(arrays["x"][keep], dtype=np.float32)),
                   state=torch.as_tensor(np.array(state), dtype=torch.long),
                   z=torch.as_tensor(np.stack(z), dtype=torch.float32),
                   mask=torch.as_tensor(np.stack(mask), dtype=torch.bool))
    return tensors, int((~keep).sum())


@torch.no_grad()
def _validation_loss(model, val, batch, seed):
    """Mean loss over the validation rows with a fixed noise/time draw, so epochs are comparable."""
    model.eval()
    tot = {}
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for i in range(0, len(val["x"]), batch):
            sl = slice(i, i + batch)
            _, parts = model.loss(val["x"][sl], val["state"][sl], val["z"][sl], val["mask"][sl])
            for k, v in parts.items():
                tot[k] = tot.get(k, 0.0) + v * len(val["x"][sl])
    return {k: v / len(val["x"]) for k, v in tot.items()}


def train(train_arrays, val_arrays, cfg, on_epoch=None):
    """Fit an InverseModel; returns (model with the best-validation weights, history dict).

    `cfg` is a TrainConfig (or a dict of its fields). `on_epoch(model, history, improved)` is called after each epoch."""
    cfg = _config(cfg)
    if cfg.epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {cfg.epochs}")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    tr, n_ex_tr = prepare(train_arrays, cfg.exclude)
    va, n_ex_va = prepare(val_arrays, cfg.exclude)
    masks = torch.as_tensor(np.stack([active_slots(s) for s in STATES]))
    model = InverseModel(len(STATES), state_masks=masks)                      # fixed table: sampling never depends on training
    opt = torch.optim.Adam(model.parameters(), cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    n = len(tr["x"])
    history = dict(train_loss=[], val_loss=[], train_ce=[], train_flow=[], val_ce=[], val_flow=[], seconds=[],
                   best_epoch=None, excluded=dict(hide=cfg.exclude, train=n_ex_tr, val=n_ex_va),
                   n_train=n, n_val=len(va["x"]), cfg=asdict(cfg))
    best, best_state = float("inf"), None
    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot = dict(loss=0.0, ce=0.0, flow=0.0)
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            loss, parts = model.loss(tr["x"][idx], tr["state"][idx], tr["z"][idx], tr["mask"][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k in tot:
                tot[k] += parts[k] * len(idx)
        v = _validation_loss(model, va, cfg.batch, cfg.seed + 1)
        if not (np.isfinite(tot["loss"]) and np.isfinite(v["loss"])):
            raise FloatingPointError(f"non-finite loss in epoch {epoch}: train {tot['loss'] / n}, val {v['loss']}")
        history["train_loss"].append(tot["loss"] / n)
        history["train_ce"].append(tot["ce"] / n)
        history["train_flow"].append(tot["flow"] / n)
        history["val_loss"].append(v["loss"])
        history["val_ce"].append(v["ce"])
        history["val_flow"].append(v["flow"])
        history["seconds"].append(time.time() - t0)
        improved = v["loss"] < best
        if improved:
            best, history["best_epoch"] = v["loss"], epoch
            best_state = copy.deepcopy(model.state_dict())
        if on_epoch:
            on_epoch(model, history, improved)
        if epoch - history["best_epoch"] >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return model.eval(), history


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=None, help="default: results/inverse_design/model_v1, or hidden_<material> with --exclude")
    ap.add_argument("--exclude", choices=("none", "hbn", "kagome", "silicene"), default="none")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None, help="torch intra-op threads (default: torch's own)")
    a = ap.parse_args(argv)
    code_version()                                     # this run's code version: fixed now, when it starts
    out = Path(a.out) if a.out else default_out(a.exclude)
    if a.exclude != "none" and out.resolve() == Path(DEFAULT_OUT).resolve():
        raise SystemExit(f"--exclude {a.exclude} would overwrite the full model in {out} (model_v1): use another --out")
    if a.threads:
        torch.set_num_threads(a.threads)
    out.mkdir(parents=True, exist_ok=True)
    extra = dict(data=str(a.data), code_version=code_version(), data_provenance=data.data_provenance(a.data),
                 threads=torch.get_num_threads())
    t0 = time.time()
    tr, va = data.load_split(a.data, "train", NEEDED), data.load_split(a.data, "val", NEEDED)
    print(f"[inverse] {len(tr['x']):,} train / {len(va['x']):,} val rows loaded ({time.time() - t0:.0f}s)", flush=True)

    def on_epoch(model, history, improved):
        e = len(history["val_loss"]) - 1
        print(f"[inverse] epoch {e} train {history['train_loss'][e]:.4f} val {history['val_loss'][e]:.4f} "
              f"({history['seconds'][e]:.0f}s){' *' if improved else ''}", flush=True)
        if improved:                                  # the file always holds the best model so far
            atomic_save(out / "model.pt", lambda p: torch.save({"cfg": model.cfg, "state_dict": model.state_dict()}, p))
        atomic_save(out / "history.json", lambda p: p.write_text(json.dumps({**history, **extra}, indent=1)))

    cfg = TrainConfig(a.epochs, a.batch, a.lr, a.patience, a.seed, a.exclude)
    model, history = train(tr, va, cfg, on_epoch)
    atomic_save(out / "model.pt", model.save)
    atomic_save(out / "history.json", lambda p: p.write_text(json.dumps({**history, **extra}, indent=1)))
    print(f"[inverse] best epoch {history['best_epoch']} val {min(history['val_loss']):.4f}; wrote {out}", flush=True)
    return history


if __name__ == "__main__":
    main()
