"""Train the inverse model (spec section 6) on the generated parent-lattice data.

    PYTHONPATH=src python scripts/build/train_inverse.py --exclude none --threads 16

Reads the train and validation splits (draw index % 20 in 0-17 and 18; the test split is never read here), encodes each
draw once, and trains with Adam and early stopping on the mean validation loss. `--exclude {hbn,kagome,silicene}` drops
the rows near that material from the training AND the validation rows (the hidden material must not steer early
stopping either). Writes model.pt (best epoch) and history.json to --out: by default model_v1 for --exclude none and
hidden_<material> otherwise (an excluded run never writes into model_v1). history.json records this run's code version and
the data's provenance. `--drop-pattern CHK` (repeatable) removes every draw of that site-type pattern from the train AND
validation rows (INV-NOCHK): the model never sees those lattices, so their states stay unknown (mask_known False) and
get no probability at inference.
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
from tbribbon import parent_data
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
    group_norm: bool = True                    # False: the encoder without GroupNorm (INV-NOGN)
    weight_decay: float = 0.01                 # AdamW's decoupled weight decay; 0 is plain Adam (the model_v1 run)
    schedule: str = "none"                     # none (constant rate) or plateau (halve it when validation stalls)
    input: str = "log"                         # log: the stored x; linear: T itself (parent_data.to_input_linear)
    extra_conv: int = 0                        # extra stride-1 conv blocks in the encoder (INV-DEPTH)
    encoder: str = "cnn"                       # cnn or mlp (INV-MLP)
    mlp_hidden: tuple = (512, 256)             # mlp encoder: hidden widths
    dropout: float = 0.0                       # mlp encoder: dropout after each hidden layer
    init_encoder: str = ""                     # path of a SpectrumAutoencoder ae.pt whose encoder starts the model (INV-AE)
    freeze_encoder: bool = False               # keep that encoder fixed (and in eval mode) while the heads train
    pool: str = "avg13"                        # cnn pooling before the Linear: avg13, gap or spectral (INV-POOL)
    embed: int = 128                           # length of the encoder's output vector, read by both heads (INV-EMBED)
    drop_pattern: tuple = ()                   # site-type patterns (pattern NAMES) left out: their states are unknown (INV-NOCHK)
    teachers: tuple = ()                       # model.pt paths whose averaged state probabilities are soft targets (INV-DISTIL)
    distill_alpha: float = 0.0                 # weight of the soft-target term in the state loss (1 - alpha on the hard label)
    distill_temp: float = 1.0                  # temperature of teachers and student in the soft-target term
    scale: str = "direct"                      # direct, or bandtop: the flow models the band top over the scale (INV-SCALE)


PLATEAU_FACTOR, PLATEAU_PATIENCE, MIN_LR = 0.5, 3, 1e-5     # the plateau schedule: halve after 3 epochs without a new best
NEEDED = ("index", "x", "on", "lead", "params", "edge", "width", "density")      # the arrays `train` reads


def _config(cfg):
    return cfg if isinstance(cfg, TrainConfig) else TrainConfig(**cfg)


def make_optimizer(model, cfg):
    """(AdamW, scheduler or None). With weight_decay 0 AdamW is exactly Adam; schedule "plateau" halves the rate (down to
    MIN_LR) after PLATEAU_PATIENCE epochs without a new best validation loss, and is stepped with that loss."""
    cfg = _config(cfg)
    if cfg.schedule not in ("none", "plateau"):
        raise ValueError(f"unknown schedule {cfg.schedule!r}; use none or plateau")
    opt = torch.optim.AdamW(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE,
                                                        min_lr=MIN_LR) if cfg.schedule == "plateau" else None)
    return opt, sched


def prepare(arrays, hide="none", input="log"):
    """(tensors, excluded count): x float32 [N, 416], state int64 [N], z float32 [N, 14], mask bool [N, 14].

    Rows in the neighbourhood of `hide` are dropped; every remaining draw is encoded once. x is the stored log input
    for input="log", or T itself from the stored raw spectra for input="linear" (the arrays then need "raw")."""
    if input not in ("log", "linear", "bandtop"):
        raise ValueError(f"unknown input {input!r}; use log, linear or bandtop")
    draws = data.draws(arrays, slice(None))
    keep = np.array([hide == "none" or not excluded(d, hide) for d in draws], dtype=bool)
    enc = [encode(d) for d, k in zip(draws, keep) if k]
    if not enc:
        raise ValueError(f"no rows left after excluding {hide!r}")
    state, z, mask = zip(*enc)
    x = (arrays["x"][keep] if input == "log" else parent_data.to_input_linear(arrays["raw"][keep]) if input == "linear"
         else parent_data.to_input_bandtop(arrays["raw"][keep]))
    tensors = dict(x=torch.as_tensor(np.asarray(x, dtype=np.float32)),
                   state=torch.as_tensor(np.array(state), dtype=torch.long),
                   z=torch.as_tensor(np.stack(z), dtype=torch.float32),
                   mask=torch.as_tensor(np.stack(mask), dtype=torch.bool))
    return tensors, int((~keep).sum())


@torch.no_grad()
def teacher_targets(teachers, x, temp=1.0, batch=4096):
    """[N, n_states] soft targets: the mean over `teachers` of softmax(logits / temp), unknown states masked out."""
    out = []
    for m in teachers:
        m.eval()
        out.append(torch.cat([torch.softmax(m.discrete(m.encode(x[i:i + batch])).masked_fill(~m.mask_known, -torch.inf)
                                            / temp, 1) for i in range(0, len(x), batch)]))
    return torch.stack(out).mean(0)


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


def train(train_arrays, val_arrays, cfg, on_epoch=None, weights=None):
    """Fit an InverseModel; returns (model with the best-validation weights, history dict).

    `cfg` is a TrainConfig (or a dict of its fields). `on_epoch(model, history, improved)` is called after each epoch.
    `weights` (optional, one per train row) weights the training loss (active learning's importance weights); the
    validation loss is never weighted."""
    cfg = _config(cfg)
    if cfg.epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {cfg.epochs}")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if weights is not None and cfg.exclude != "none":
        raise ValueError("row weights and --exclude cannot be combined (the weights are per loaded row)")
    teachers = [InverseModel.load(p) for p in cfg.teachers]
    for p, t in zip(cfg.teachers, teachers):
        if t.cfg.get("input", "log") != cfg.input:
            raise ValueError(f"teacher {p} reads input {t.cfg.get('input', 'log')!r}, the student {cfg.input!r}")
    tr, n_ex_tr = prepare(train_arrays, cfg.exclude, cfg.input)
    if weights is not None:
        if len(weights) != len(tr["x"]):
            raise ValueError(f"{len(weights)} weights for {len(tr['x'])} train rows")
        tr["w"] = torch.as_tensor(np.asarray(weights, dtype=np.float32))
    va, n_ex_va = prepare(val_arrays, cfg.exclude, cfg.input)
    masks = torch.as_tensor(np.stack([active_slots(s) for s in STATES]))
    model = InverseModel(len(STATES), channels=tr["x"].shape[1], state_masks=masks, group_norm=cfg.group_norm,
                         input=cfg.input,
                         extra_conv=cfg.extra_conv, encoder=cfg.encoder,
                         mlp_hidden=tuple(cfg.mlp_hidden), dropout=cfg.dropout, pool=cfg.pool,
                         embed=cfg.embed, scale=cfg.scale)   # fixed table: sampling never depends on training
    if cfg.drop_pattern:                           # left-out lattices: no probability in training or at inference
        from shazam.inverse.pattern import NAMES
        from tbribbon.parent_prior import PATTERNS
        gone = {PATTERNS[NAMES.index(n)] for n in cfg.drop_pattern}
        model.mask_known[torch.tensor([st.pattern in gone for st in STATES])] = False
    if teachers:                                   # soft targets only on states the student knows, renormalised
        soft = teacher_targets(teachers, tr["x"], cfg.distill_temp) * model.mask_known
        tr["soft"] = soft / soft.sum(1, keepdim=True).clamp_min(1e-12)
        del teachers
    frozen = []
    if cfg.init_encoder:
        from shazam.inverse.autoencoder import SpectrumAutoencoder
        SpectrumAutoencoder.load(cfg.init_encoder).copy_encoder_into(model)        # raises on any setting mismatch
        if cfg.freeze_encoder:
            frozen = [model.encoder] + ([model.band_top] if hasattr(model, "band_top") else [])
            for m in frozen:
                m.requires_grad_(False)
    elif cfg.freeze_encoder:
        raise ValueError("freeze_encoder needs init_encoder")
    opt, sched = make_optimizer(model, cfg)
    gen = torch.Generator().manual_seed(cfg.seed)
    n = len(tr["x"])
    history = dict(train_loss=[], val_loss=[], train_ce=[], train_flow=[], val_ce=[], val_flow=[], seconds=[], lr=[],
                   best_epoch=None, excluded=dict(hide=cfg.exclude, train=n_ex_tr, val=n_ex_va),
                   n_train=n, n_val=len(va["x"]), cfg=asdict(cfg))
    if "soft" in tr:
        history["train_kd"] = []
    best, best_state = float("inf"), None
    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        for m in frozen:                               # a frozen encoder keeps dropout off
            m.eval()
        perm = torch.randperm(n, generator=gen)
        tot = dict(loss=0.0, ce=0.0, flow=0.0, **({"kd": 0.0} if "soft" in tr else {}))
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            batch = (tr["x"][idx], tr["state"][idx], tr["z"][idx], tr["mask"][idx])
            kw = {"weight": tr["w"][idx]} if "w" in tr else {}
            if "soft" in tr:
                kw.update(soft=tr["soft"][idx], alpha=cfg.distill_alpha, temp=cfg.distill_temp)
            loss, parts = model.loss(*batch, **kw)
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
        if "kd" in tot:
            history["train_kd"].append(tot["kd"] / n)
        history["val_loss"].append(v["loss"])
        history["val_ce"].append(v["ce"])
        history["val_flow"].append(v["flow"])
        history["seconds"].append(time.time() - t0)
        history["lr"].append(opt.param_groups[0]["lr"])            # the rate this epoch trained with
        if sched is not None:
            sched.step(v["loss"])
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
    ap.add_argument("--weight-decay", type=float, default=0.01,
                    help="AdamW weight decay (default 0.01; 0 is plain Adam, as model_v1 was trained)")
    ap.add_argument("--schedule", choices=("none", "plateau"), default="none", help="learning-rate schedule")
    ap.add_argument("--input", choices=("log", "linear", "bandtop"), default="log",
                    help="log: the stored log-scaled x; linear: T itself; bandtop: T resampled to its own band top")
    ap.add_argument("--encoder", choices=("cnn", "mlp"), default="cnn", help="1D-conv or fully connected encoder")
    ap.add_argument("--pool", choices=("avg13", "gap", "spectral"), default="avg13", help="cnn pooling before the Linear")
    ap.add_argument("--init-encoder", default="", help="SpectrumAutoencoder ae.pt to start the encoder from")
    ap.add_argument("--freeze-encoder", action="store_true", help="keep the --init-encoder encoder fixed")
    ap.add_argument("--mlp-hidden", type=int, nargs="+", default=[512, 256], help="mlp encoder hidden widths")
    ap.add_argument("--dropout", type=float, default=0.0, help="mlp encoder dropout")
    ap.add_argument("--extra-conv", type=int, default=0, help="extra stride-1 conv blocks after the third conv")
    ap.add_argument("--embed", type=int, default=128, help="length of the encoder output both heads read")
    ap.add_argument("--no-group-norm", dest="group_norm", action="store_false", help="encoder without GroupNorm")
    ap.add_argument("--max-draws", type=int, default=None,
                    help="use only draws with index below this (keeps a run comparable while the data set grows)")
    ap.add_argument("--drop-pattern", action="append", default=[], choices=("C", "H", "K", "CH", "CK", "HK", "CHK"),
                    help="drop every draw of this site-type pattern from train and val (repeatable)")
    ap.add_argument("--extra-data", action="append", default=[],
                    help="an active-learning data root whose train rows are added, importance-weighted back to the "
                         "prior (repeatable; validation stays on --data)")
    ap.add_argument("--add-data", action="append", default=[],
                    help="a data root whose train rows are added as they are, unweighted (repeatable; validation "
                         "stays on --data). For generate_parent.py --near data: the training set then holds more "
                         "hosts near the known materials than the prior does")
    ap.add_argument("--teacher", action="append", default=[],
                    help="model.pt (or its folder) of a teacher; repeatable: their averaged state probabilities become "
                         "soft targets (INV-DISTIL)")
    ap.add_argument("--distill-alpha", type=float, default=0.7, help="weight of the soft-target term (with --teacher)")
    ap.add_argument("--distill-temp", type=float, default=2.0, help="soft-target temperature (with --teacher)")
    ap.add_argument("--scale", choices=("direct", "bandtop"), default="direct",
                    help="bandtop: the flow models the band top over the energy scale (needs --input bandtop)")
    a = ap.parse_args(argv)
    code_version()                                    # this run's code version: fixed now, when it starts
    if a.scale == "bandtop" and a.input != "bandtop":
        raise SystemExit("--scale bandtop needs --input bandtop (the band top is that input's last value)")
    if a.add_data and (a.extra_data or a.drop_pattern or a.exclude != "none" or a.max_draws is not None):
        raise SystemExit("--add-data cannot be combined with --extra-data, --drop-pattern, --exclude or --max-draws")
    out = Path(a.out) if a.out else default_out(a.exclude)
    if a.exclude != "none" and out.resolve() == Path(DEFAULT_OUT).resolve():
        raise SystemExit(f"--exclude {a.exclude} would overwrite the full model in {out} (model_v1): use another --out")
    if a.threads:
        torch.set_num_threads(a.threads)
    out.mkdir(parents=True, exist_ok=True)
    extra = dict(data=str(a.data), code_version=code_version(), data_provenance=data.data_provenance(a.data),
                 threads=torch.get_num_threads(), max_draws=a.max_draws)
    t0 = time.time()
    keys = NEEDED + (("raw",) if a.input != "log" else ())
    tr, va = data.load_split(a.data, "train", keys), data.load_split(a.data, "val", keys)
    if a.max_draws is not None:
        tr, va = ({k: v[s["index"] < a.max_draws] for k, v in s.items()} for s in (tr, va))
    if a.drop_pattern:
        if a.extra_data:
            raise SystemExit("--drop-pattern and --extra-data cannot be combined (the weights assume the full prior)")
        from shazam.inverse.pattern import NAMES, pattern_labels
        drop = [NAMES.index(n) for n in a.drop_pattern]
        tr, va = ({k: v[~np.isin(pattern_labels(s["on"]), drop)] for k, v in s.items()} for s in (tr, va))
        extra.update(drop_pattern=list(a.drop_pattern))
    weights = None
    if a.extra_data:
        from shazam.inverse import active
        prior_cells = active.cells_of_arrays(tr)
        extras = [data.load_split(r, "train", keys) for r in a.extra_data]
        tr = {k: np.concatenate([tr[k]] + [e[k] for e in extras]) for k in tr}
        weights = active.pooled_weights(np.concatenate([prior_cells] + [active.cells_of_arrays(e) for e in extras]),
                                        prior_cells)
        extra.update(extra_data=[dict(root=str(r), n_train=len(e["index"]), provenance=data.data_provenance(r))
                                 for r, e in zip(a.extra_data, extras)],
                     weight_stats=dict(min=float(weights.min()), max=float(weights.max()),
                                       ess=float(weights.sum() ** 2 / (weights ** 2).sum())))
    if a.add_data:
        added = [data.load_split(r, "train", keys) for r in a.add_data]
        tr = {k: np.concatenate([tr[k]] + [e[k] for e in added]) for k in tr}
        if len(np.unique(tr["index"])) != len(tr["index"]):
            raise SystemExit("--add-data: a draw index occurs twice (the same data given twice?)")
        extra.update(added_data=[dict(root=str(r), n_train=len(e["index"]), provenance=data.data_provenance(r))
                                 for r, e in zip(a.add_data, added)])
    print(f"[inverse] {len(tr['x']):,} train / {len(va['x']):,} val rows loaded ({time.time() - t0:.0f}s)", flush=True)

    def on_epoch(model, history, improved):
        e = len(history["val_loss"]) - 1
        print(f"[inverse] epoch {e} train {history['train_loss'][e]:.4f} val {history['val_loss'][e]:.4f} "
              f"lr {history['lr'][e]:.1e} ({history['seconds'][e]:.0f}s){' *' if improved else ''}", flush=True)
        if improved:                                  # the file always holds the best model so far
            atomic_save(out / "model.pt", lambda p: torch.save({"cfg": model.cfg, "state_dict": model.state_dict()}, p))
        atomic_save(out / "history.json", lambda p: p.write_text(json.dumps({**history, **extra}, indent=1)))

    cfg = TrainConfig(a.epochs, a.batch, a.lr, a.patience, a.seed, a.exclude, a.group_norm, a.weight_decay, a.schedule, a.input, a.extra_conv, a.encoder,
                      tuple(a.mlp_hidden), a.dropout, a.init_encoder, a.freeze_encoder, a.pool, a.embed,
                      tuple(a.drop_pattern),
                      tuple(str(Path(t) / "model.pt" if Path(t).is_dir() else Path(t)) for t in a.teacher),
                      a.distill_alpha if a.teacher else 0.0, a.distill_temp if a.teacher else 1.0, a.scale)
    model, history = train(tr, va, cfg, on_epoch, weights=weights)
    atomic_save(out / "model.pt", model.save)
    atomic_save(out / "history.json", lambda p: p.write_text(json.dumps({**history, **extra}, indent=1)))
    print(f"[inverse] best epoch {history['best_epoch']} val {min(history['val_loss']):.4f}; wrote {out}", flush=True)
    return history


if __name__ == "__main__":
    main()
