"""Fine-tune an inverse model with a truth reward (§5 of
docs/superpowers/specs/2026-11-04-reward-finetuning-design.md).

    TRANSMISSION_DATA=... PYTHONPATH=src nice -n 10 python scripts/build/finetune_inverse.py --reward physics \\
        --out results/inverse_design/model_v3_reward_physics --threads 16

Starts from --ref (default model_v2_depth_bandtop_600k; that file is only read) and fine-tunes every weight. A frozen
copy of the reference anchors the classifier. Per batch:

* --reward none (the C-more control): exactly the original loss, InverseModel.loss.
* --reward label / physics: the expected family cost over the 156 states, computed exactly from the cost table of
  scripts/build/reward_tables.py, plus beta KL(p || p_ref) plus lam CE. Then mu times the flow-matching loss on the
  true rows, plus a reward on flow samples of `flow_rows` rows: 0.25 width surrogate + 0.15 continuous cost, back-
  propagated through the last `grad_last` Euler steps.

After each epoch: the headline family top-1 (family + edge, probability summed over lead) and family NLL on a fixed,
seeded val subset. Early stopping on top-1 (ties: lower NLL). Writes model.pt (the best epoch) and history.json.
"""
import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import paths
from shazam.inverse import data, reward as rw
from shazam.inverse.encoding import encode
from shazam.inverse.model import InverseModel
from shazam.inverse.runio import atomic_save, atomic_write_json, code_version, sha256_of
from tbribbon import parent_data

DEFAULT_DATA = paths.data_root() / "_generated" / "hexagonal_parent_v1"
DEFAULT_REF = paths.results_dir("inverse_design", "model_v2_depth_bandtop_600k") / "model.pt"
DEFAULT_TABLES = paths.results_dir("inverse_design", "reward_tables")
NEEDED = ("index", "x", "raw", "on", "lead", "params", "edge", "width", "density")
MASKED = -1e9                             # logit of a state the model never saw (finite, so p log p stays 0)


@dataclass
class FinetuneConfig:
    reward: str                           # label, physics or none
    ref: str = str(DEFAULT_REF)
    beta: float = 0.1                     # KL(p || p_ref) weight
    lam: float = 0.5                      # cross-entropy weight
    mu: float = 1.0                       # flow-matching anchor weight
    grad_last: int = 2                    # Euler steps the flow reward back-propagates through
    steps: int = 64                       # Euler steps of the flow samples
    flow_true_share: float = 0.5          # share of flow-reward rows drawn at the true state (the rest from p)
    flow_rows: int = 64                   # rows per batch that get the flow reward
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 5
    batch: int = 512
    patience: int = 2
    seed: int = 0
    val_subset: int = 5000
    max_steps_per_epoch: int = 0          # 0: a full pass over the train rows


def prepare_rows(arrays, table, input_kind):
    """Tensors of one split: x (the model's input), state, z, mask, phys (lead-free truth vector) and, with a table
    ({"cost": (n, 156), "weights": (n, 3)} in row order), cost and truth (rw.truth_arrays)."""
    draws = data.draws(arrays, slice(None))
    enc = [encode(d) for d in draws]
    x = {"log": lambda: arrays["x"], "linear": lambda: parent_data.to_input_linear(arrays["raw"]),
         "bandtop": lambda: parent_data.to_input_bandtop(arrays["raw"])}[input_kind]()
    out = dict(x=torch.as_tensor(np.asarray(x, dtype=np.float32)),
               state=torch.as_tensor(np.array([e[0] for e in enc]), dtype=torch.long),
               z=torch.as_tensor(np.stack([e[1] for e in enc]), dtype=torch.float32),
               mask=torch.as_tensor(np.stack([e[2] for e in enc]), dtype=torch.bool),
               phys=torch.as_tensor(np.stack([rw.physical(d) for d in draws]), dtype=torch.float32))
    if table is not None:
        out["cost"] = torch.as_tensor(np.asarray(table["cost"], dtype=np.float32))
        out["truth"] = rw.truth_arrays(draws, table["weights"])
    return out


def classifier_loss(logits, ref_logits, cost, y, beta, lam):
    """E_p[cost] + beta KL(p || p_ref) + lam CE(logits, y); returns (loss, parts)."""
    logp = F.log_softmax(logits, 1)
    p = logp.exp()
    expected = (p * cost).sum(1).mean()
    kl = (p * (logp - F.log_softmax(ref_logits, 1))).sum(1).mean()
    ce = F.cross_entropy(logits, y)
    loss = expected + beta * kl + lam * ce
    return loss, dict(expected_cost=expected.item(), kl=kl.item(), ce=ce.item())


def _masked(model, logits):
    return logits.masked_fill(~model.mask_known, MASKED)


def flow_reward_loss(model, h, rows, logits, tensors, cfg):
    """0.25 width surrogate + 0.15 continuous cost of flow samples for the batch positions `rows` (§4, §5)."""
    n = len(rows)
    n_true = int(round(cfg.flow_true_share * n))
    true_state = tensors["state"][rows]
    sampled = torch.multinomial(F.softmax(logits[n_true:n].detach(), 1), 1)[:, 0] if n > n_true else true_state[:0]
    states = torch.cat([true_state[:n_true], sampled])
    z = model.sample_batch(h, states, torch.randn(n, model.n_cont), steps=cfg.steps, grad_last=cfg.grad_last)
    pred = rw.physical_torch(states, z)
    truth = tensors["phys"][rows]
    width = rw.width_surrogate(pred[:, -1] - truth[:, -1])
    cont = rw.continuous_cost(cfg.reward, truth, pred, true_state, states, *tensors["sd"])
    return (rw.WEIGHTS["width"] * width + rw.WEIGHTS["continuous"] * cont).mean()


def step_loss(model, ref, tensors, idx, cfg):
    """The loss of one batch (row indices idx) and its parts."""
    x, y, z, mask = (tensors[k][idx] for k in ("x", "state", "z", "mask"))
    if cfg.reward == "none":
        return model.loss(x, y, z, mask)
    h = model.encode(x)
    logits = _masked(model, model.discrete(h))
    with torch.no_grad():
        ref_logits = _masked(ref, ref.discrete(ref.encode(x)))
    cls, parts = classifier_loss(logits, ref_logits, tensors["cost"][idx], y, cfg.beta, cfg.lam)
    flow = model.flow_loss(h, y, z, mask)
    k = min(cfg.flow_rows, len(idx))
    reward = flow_reward_loss(model, h[:k], idx[:k], logits[:k], tensors, cfg) if k else torch.zeros(())
    loss = cls + cfg.mu * flow + reward
    return loss, dict(parts, flow=flow.item(), flow_reward=reward.item(), loss=loss.item())


@torch.no_grad()
def family_scores(model, x, truth_states, batch=2048):
    """(headline family top-1, family NLL) of the model on inputs x against true states."""
    model.eval()
    probs = torch.cat([F.softmax(_masked(model, model.discrete(model.encode(x[i:i + batch]))), 1)
                       for i in range(0, len(x), batch)]).numpy()
    fam = rw.family_probs(probs)
    true_fam = rw.FAMILY_OF[np.asarray(truth_states)]
    p_true = fam[np.arange(len(fam)), true_fam]
    return float((fam.argmax(1) == true_fam).mean()), float(-np.log(np.clip(p_true, 1e-12, None)).mean())


def finetune(train_arrays, val_arrays, tables, cfg):
    """(model with the best val-subset epoch, history)."""
    if cfg.reward not in ("label", "physics", "none"):
        raise ValueError(f"unknown reward {cfg.reward!r}: expected label, physics or none")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = InverseModel.load(cfg.ref)
    ref = None
    if cfg.reward != "none":
        ref = InverseModel.load(cfg.ref).eval()
        ref.requires_grad_(False)
    kind = model.cfg.get("input", "log")
    tr = prepare_rows(train_arrays, tables["train"] if tables else None, kind)
    phys = tr["phys"].numpy()
    tr["sd"] = (float(phys[:, -2].std()), float(phys[:, -1].std()))                  # train sd of log c and width
    sub = data.seeded_rows(len(val_arrays["index"]), cfg.val_subset)
    val_sub = {k: v[sub] for k, v in val_arrays.items()}
    va = prepare_rows(val_sub, None, kind)
    opt = torch.optim.AdamW(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)
    history = dict(cfg=asdict(cfg), ref_sha256=sha256_of(cfg.ref), code_version=code_version(), train=[], val=[],
                   seconds=[], best_epoch=None, n_train=len(tr["x"]), val_subset_index=val_sub["index"].tolist(),
                   sd_logc_w=list(tr["sd"]),
                   physics_scales=tables["scales"].to_json() if tables and "scales" in tables else None)
    top1, nll = family_scores(model, va["x"], va["state"])
    history["val_start"] = dict(family_top1=top1, family_nll=nll)
    best, best_state = (top1, -nll), copy.deepcopy(model.state_dict())
    n = len(tr["x"])
    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot, rows = {}, 0
        for j, i in enumerate(range(0, n, cfg.batch)):
            if cfg.max_steps_per_epoch and j >= cfg.max_steps_per_epoch:
                break
            idx = perm[i:i + cfg.batch]
            loss, parts = step_loss(model, ref, tr, idx, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in epoch {epoch}: {parts}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in parts.items():
                tot[k] = tot.get(k, 0.0) + v * len(idx)
            rows += len(idx)
        top1, nll = family_scores(model, va["x"], va["state"])
        history["train"].append({k: v / rows for k, v in tot.items()})
        history["val"].append(dict(family_top1=top1, family_nll=nll))
        history["seconds"].append(time.time() - t0)
        print(f"[finetune] epoch {epoch} val top-1 {top1:.4f} nll {nll:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if (top1, -nll) > best:
            best, best_state, history["best_epoch"] = (top1, -nll), copy.deepcopy(model.state_dict()), epoch
        last_best = -1 if history["best_epoch"] is None else history["best_epoch"]
        if epoch - last_best >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return model.eval(), history


def write_run(out, model, history):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_save(out / "model.pt", model.save)
    atomic_write_json(out / "history.json", history)


def load_tables(folder, arm, arrays_by_split):
    """{"train"/"val": {"cost", "weights"} in the arrays' row order, "scales"} from reward_tables.py's output."""
    folder = Path(folder)
    out = dict(scales=rw.PhysicsScales.from_json({k: v for k, v in json.loads((folder / "physics_scales.json")
                                                                               .read_text()).items()
                                                 if k != "code_version"}))
    for split, arrays in arrays_by_split.items():
        c = np.load(folder / f"{split}_cost_{arm}.npz")
        w = np.load(folder / f"{split}_band_weights.npz")
        if not (np.array_equal(c["index"], arrays["index"]) and np.array_equal(w["index"], arrays["index"])):
            raise ValueError(f"{split} tables do not match the data rows: rerun scripts/build/reward_tables.py")
        out[split] = dict(cost=c["cost"], weights=w["weights"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reward", choices=("label", "physics", "none"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--tables", default=str(DEFAULT_TABLES))
    ap.add_argument("--threads", type=int, default=None)
    for f, v in asdict(FinetuneConfig(reward="none")).items():
        if f != "reward":
            ap.add_argument("--" + f.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = FinetuneConfig(**{f: getattr(a, f) for f in asdict(FinetuneConfig(reward="none"))})
    train, val = (data.load_split(a.data, s, NEEDED) for s in ("train", "val"))
    tables = None if cfg.reward == "none" else load_tables(a.tables, cfg.reward, dict(train=train))
    model, history = finetune(train, val, tables, cfg)
    history["data_provenance"] = data.data_provenance(a.data)
    write_run(a.out, model, history)
    print(f"[finetune] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
