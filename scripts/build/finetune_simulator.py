"""Phase 2: fine-tune an inverse model with the simulator reward, GRPO style (§6 of
docs/superpowers/specs/2026-11-04-reward-finetuning-design.md).

    TRANSMISSION_DATA=... PYTHONPATH=src nice -n 10 python scripts/build/finetune_simulator.py \\
        --start results/inverse_design/model_v3_more_ce/model.pt \
        --out results/inverse_design/model_v4_simreward [--resume]

Each step takes `targets_per_step` train spectra (their labels unused), draws a group of `group` candidates per target
from the model, simulates each once with a fresh disorder seed (4e9 + 1000 step + j) and scores it with
sim_reward.reward. Advantages are normalised within each group. The update:

* classifier: the policy gradient -A log p(state), plus beta KL(p || p_start) against the frozen start model;
* flow: reward-weighted flow matching on the group's own candidates, weights softmax(A / tau) within the group;
* anchor: `anchor` times the original loss on a labelled train batch, so the model stays in the prior box.

A candidate outside the prior box, or whose simulation fails, gets no reward; a group with fewer than 2 rewards is
skipped. The run stops after `max_sims` simulations or when a file named STOP appears in --out (never SIGSTOP the
workers: the stall watchdog kills the parent on resume). Every `checkpoint_every` steps, on STOP and at the end it
writes checkpoint.pt (model, optimiser, step, simulations used, RNG states, target order); --resume continues from
it exactly. At the end: model.pt and history.json.
"""
import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import paths
from shazam.inverse import data, sim_reward as sr
from shazam.inverse.encoding import decode, encode
from shazam.inverse.model import InverseModel
from shazam.inverse.runio import atomic_save, atomic_write_json, code_version, sha256_of
from tbribbon import parent_data

DEFAULT_DATA = paths.data_root() / "_generated" / "hexagonal_parent_v1"
DEFAULT_M0 = paths.results_dir("inverse_design", "model_v4_simreward") / "m0.json"
NEEDED = ("index", "x", "raw", "on", "lead", "params", "edge", "width", "density")
SEED_BASE = 4 * 10**9
MASKED = -1e9


@dataclass
class SimConfig:
    start: str                            # model.pt to start from (also the frozen KL reference)
    group: int = 8
    targets_per_step: int = 16
    max_sims: int = 100_000
    beta: float = 0.1
    tau: float = 1.0
    anchor: float = 0.1
    anchor_batch: int = 64
    lr: float = 5e-5
    seed: int = 0
    checkpoint_every: int = 20
    steps: int = 64                       # Euler steps of the candidates
    n_jobs: int = 22


def group_advantages(r):
    """(r - nanmean) / nanstd per row (population std; 0 where the std is 0); rows with fewer than 2 finite values are
    all NaN, and NaN stays NaN."""
    r = np.asarray(r, dtype=np.float64)
    out = np.full_like(r, np.nan)
    for i, row in enumerate(r):
        ok = np.isfinite(row)
        if ok.sum() < 2:
            continue
        sd = row[ok].std()
        out[i, ok] = 0.0 if sd == 0 else (row[ok] - row[ok].mean()) / sd
    return out


def _masked_logits(model, h):
    return model.discrete(h).masked_fill(~model.mask_known, MASKED)


@torch.no_grad()
def propose(model, x, group, steps):
    """(states (T, G), z (T, G, n_cont)) candidates for the targets x (T, C)."""
    h = model.encode(x)
    states = torch.multinomial(F.softmax(_masked_logits(model, h), 1), group, replacement=True)
    flat = states.reshape(-1)
    z = model.sample_batch(h.repeat_interleave(group, 0), flat, torch.randn(len(flat), model.n_cont), steps=steps)
    return states, z.reshape(len(x), group, -1)


def grpo_step(model, ref, opt, x, states, z, rewards, labelled, cfg):
    """One update from candidates (states (T, G), z (T, G, n_cont)) and their rewards (T, G; NaN = none)."""
    A = group_advantages(rewards)
    valid = np.isfinite(A)
    if not valid.any():
        return dict(skipped=True)
    T, G = states.shape
    v = torch.as_tensor(valid.reshape(-1))
    adv = torch.as_tensor(np.nan_to_num(A).reshape(-1), dtype=torch.float32)
    model.train()
    h = model.encode(x)
    logp = F.log_softmax(_masked_logits(model, h), 1)
    flat = states.reshape(-1)
    lp = logp.repeat_interleave(G, 0)[torch.arange(T * G), flat]
    pg = -(adv[v] * lp[v]).sum() / v.sum()
    with torch.no_grad():
        ref_logp = F.log_softmax(_masked_logits(ref, ref.encode(x)), 1)
    kl = (logp.exp() * (logp - ref_logp)).sum(1).mean()
    w = torch.where(v.reshape(T, G), adv.reshape(T, G) / cfg.tau, torch.full((T, G), -torch.inf))
    w = torch.softmax(w, 1).nan_to_num(0.0).reshape(-1)
    zf = z.reshape(T * G, -1)
    fm = model.flow_loss(h.repeat_interleave(G, 0)[v], flat[v], zf[v], model.state_masks[flat[v]], weight=w[v])
    loss = pg + cfg.beta * kl + fm
    anchor = torch.zeros(())
    if labelled is not None and cfg.anchor > 0:
        idx = torch.randint(0, len(labelled["x"]), (cfg.anchor_batch,))
        anchor, _ = model.loss(labelled["x"][idx], labelled["state"][idx], labelled["z"][idx], labelled["mask"][idx])
        loss = loss + cfg.anchor * anchor
    opt.zero_grad()
    loss.backward()
    opt.step()
    return dict(skipped=False, pg=pg.item(), kl=kl.item(), fm=fm.item(), anchor=anchor.item(), n_valid=int(v.sum()),
                reward_mean=float(np.nanmean(rewards)))


def _save_checkpoint(out, model, opt, state):
    payload = dict(model=model.state_dict(), opt=opt.state_dict(), torch_rng=torch.get_rng_state(),
                   **dict(state, perm=torch.as_tensor(state["perm"])))
    atomic_save(out / "checkpoint.pt", lambda p: torch.save(payload, p))


def run(cfg, out, targets, labelled, simulate=sr.simulate_candidates, m0=None, on_step=None, resume=False):
    """The GRPO loop (see the module docstring). targets: {"x": model inputs (N, C), "target_x": v4 inputs (N, 416)}.
    Returns {"step", "sims_used", "stopped"}."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = InverseModel.load(cfg.start)
    ref = InverseModel.load(cfg.start).eval()
    ref.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), cfg.lr)
    n = len(targets["x"])
    ck = out / "checkpoint.pt"
    if resume and ck.exists():
        c = torch.load(ck, map_location="cpu", weights_only=True)      # tensors and plain Python values only
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        torch.set_rng_state(c["torch_rng"])
        rng = np.random.default_rng()
        rng.bit_generator.state = c["np_rng"]
        state = dict(step=c["step"], sims_used=c["sims_used"], perm=c["perm"].numpy(), pos=c["pos"],
                     history=c["history"])
    else:
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        state = dict(step=0, sims_used=0, perm=rng.permutation(n), pos=0, history=[])
    t0 = time.time()

    def snapshot():
        _save_checkpoint(out, model, opt, dict(state, np_rng=rng.bit_generator.state))

    while state["sims_used"] < cfg.max_sims:
        if (out / "STOP").exists():
            snapshot()
            return dict(step=state["step"], sims_used=state["sims_used"], stopped=True)
        if state["pos"] + cfg.targets_per_step > n:
            state["perm"], state["pos"] = rng.permutation(n), 0
        rows = state["perm"][state["pos"]:state["pos"] + cfg.targets_per_step]
        state["pos"] += cfg.targets_per_step
        x = targets["x"][torch.as_tensor(rows)]
        states, z = propose(model, x, cfg.group, cfg.steps)
        draws = [decode(int(s), zz) for s, zz in zip(states.reshape(-1).numpy(), z.reshape(-1, z.shape[-1]).numpy())]
        seeds = [SEED_BASE + 1000 * state["step"] + j for j in range(len(draws))]
        sims = simulate(draws, seeds, cfg.n_jobs)
        r = np.full(len(draws), np.nan)
        for j, (d, sx) in enumerate(zip(draws, sims)):
            if sx is not None:
                t = targets["target_x"][rows[j // cfg.group]]
                r[j] = sr.reward(sx, t, m0(d.width, d.density) if m0 else 1.0)
        state["sims_used"] += sum(sr.in_box(d) for d in draws)
        info = grpo_step(model, ref, opt, x, states, z, r.reshape(len(rows), cfg.group), labelled, cfg)
        state["step"] += 1
        state["history"].append(dict(info, step=state["step"], sims_used=state["sims_used"], seconds=time.time() - t0))
        if state["step"] % 10 == 0:
            print(f"[sim] step {state['step']} sims {state['sims_used']} reward {info.get('reward_mean', 'nan'):.4g} "
                  f"kl {info.get('kl', float('nan')):.4f} ({time.time() - t0:.0f} s)", flush=True)
        if on_step:
            on_step(state["step"])
        if state["step"] % cfg.checkpoint_every == 0:
            snapshot()
    snapshot()
    atomic_save(out / "model.pt", model.save)
    atomic_write_json(out / "history.json", dict(cfg=asdict(cfg), start_sha256=sha256_of(cfg.start),
                                                code_version=code_version(), steps=state["history"],
                                                sims_used=state["sims_used"]))
    return dict(step=state["step"], sims_used=state["sims_used"], stopped=False)


def _inputs(arrays, kind):
    return np.asarray({"log": lambda: arrays["x"], "bandtop": lambda: parent_data.to_input_bandtop(arrays["raw"]),
                       "linear": lambda: parent_data.to_input_linear(arrays["raw"])}[kind](), dtype=np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--m0", default=str(DEFAULT_M0))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    for f, v in asdict(SimConfig(start="")).items():
        if f != "start":
            ap.add_argument("--" + f.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = SimConfig(**{f: getattr(a, f) for f in asdict(SimConfig(start=""))})
    tr = data.load_split(a.data, "train", NEEDED)
    kind = InverseModel.load(cfg.start).cfg.get("input", "log")
    x = torch.as_tensor(_inputs(tr, kind))
    enc = [encode(d) for d in data.draws(tr, slice(None))]
    labelled = dict(x=x, state=torch.as_tensor([e[0] for e in enc]), z=torch.as_tensor(np.stack([e[1] for e in enc]),
                                                                                       dtype=torch.float32),
                    mask=torch.as_tensor(np.stack([e[2] for e in enc])))
    m0 = sr.M0Model.from_json(json.loads(Path(a.m0).read_text())["model"])
    res = run(cfg, a.out, dict(x=x, target_x=tr["x"]), labelled, m0=m0, resume=a.resume)
    print(f"[sim] {res}", flush=True)


if __name__ == "__main__":
    main()
