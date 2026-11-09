"""Diagnose the inverse model per cell and write the proposal of one active-learning round
(docs/superpowers/specs/2026-11-04-active-learning-design.md, steps 0 and 1). No simulation.

    TRANSMISSION_DATA=... PYTHONPATH=src python scripts/build/acquire_round.py \\
        --now results/inverse_design/model_v2_depth_bandtop_600k --prev results/inverse_design/model_v2_depth_bandtop \\
        --out results/inverse_design/active/cells_round1.json

Per v1 val row and model: family error (family with edge, probability summed over the unit type), width error (modal
width of the flow samples off by more than 1) and log s error (median log s of the samples off by more than LOG_S_TOL).
Per cell (family with edge x width bin): error rates of the current model on every val row, and the learning slope
(error of the previous model minus the current one, per doubling of the training set) on the val rows that neither
model's run read for early stopping beyond the previous run's --max-draws (index >= the previous run's max_draws), both
shrunk towards their site-type pattern (Bayesian part). The expected gain tilts the prior (active.tilted_proposal); the
acceptance table that generate_parent.py --proposal reads goes into --out with the whole per-cell diagnosis.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import paths
from shazam.inverse import active, data
from shazam.inverse.encoding import SLOT, STATES, _bounds, encode
from shazam.inverse.model import InverseModel
from shazam.inverse.reward import FAMILY_OF, N_FAMILIES, family_probs, pattern_name
from tbribbon.parent_data import to_input_bandtop

DEFAULT_DATA = paths.data_root() / "_generated" / "hexagonal_parent_v1"
KEYS = ("index", "raw", "on", "lead", "params", "edge", "width", "density")
LOG_S_TOL = 0.1             # a 10% error in the energy scale s counts as wrong
SAMPLES, STEPS, BATCH = 20, 64, 100
PATTERNS = ("C", "H", "K", "CH", "CK", "HK", "CHK")
_FAM_STATE = {int(f): k for k, f in enumerate(FAMILY_OF)}                 # one state of every family
FAMILY_PATTERN = np.array([PATTERNS.index(pattern_name(STATES[_FAM_STATE[f]].pattern)) for f in range(N_FAMILIES)])


def family_name(f):
    s = STATES[_FAM_STATE[f]]
    sw = ([("Δ=0" if s.delta_zero else "Δ≠0"), ("t_AA" if s.t_AA_on else "no t_AA")] if s.pattern[1] else [])
    if sum(s.pattern) == 1:
        sw.append("ε=0" if s.eps_zero else "ε≠0")
    if sum(s.pattern) == 3:
        sw.append("bonds " + "/".join(n for n, o in zip(("CH", "CK", "HK"), s.cross) if o))
    return f"{pattern_name(s.pattern)} {s.edge}" + (" " + ", ".join(sw) if sw else "")


def _decode_slot(z, name):
    lo, hi = _bounds(name)
    return np.clip(lo + (hi - lo) / (1 + np.exp(-z)), lo, hi)


@torch.no_grad()
def row_errors(model, x, true_family, width, log_s, seed=0):
    """(family_err, width_err, log_s_err) booleans per row: SAMPLES posterior samples per row, batched."""
    model.eval()
    torch.manual_seed(seed)
    fam_err, w_err, s_err = [], [], []
    for i in range(0, len(x), BATCH):
        sl = slice(i, i + BATCH)
        h = model.encode(x[sl])
        logits = model.discrete(h).masked_fill(~model.mask_known, -torch.inf)
        prob = F.softmax(logits, 1)
        fam_err.append(family_probs(prob.numpy()).argmax(1) != true_family[sl])
        st = torch.multinomial(prob, SAMPLES, replacement=True).reshape(-1)
        z = model.sample_batch(h.repeat_interleave(SAMPLES, 0), st, torch.randn(len(st), model.n_cont), steps=STEPS)
        z = z.numpy().reshape(len(h), SAMPLES, -1)
        w = np.rint(_decode_slot(z[..., SLOT["w"]], "w")).astype(int)
        mode = np.array([np.bincount(r).argmax() for r in w])
        w_err.append(np.abs(mode - width[sl]) > 1)
        s_err.append(np.abs(np.median(_decode_slot(z[..., SLOT["log_s"]], "log_s"), 1) - log_s[sl]) > LOG_S_TOL)
    return {k: np.concatenate(v) for k, v in zip(("family", "width", "log_s"), (fam_err, w_err, s_err))}


def _max_draws(model_dir):
    h = json.loads((Path(model_dir) / "history.json").read_text())
    return int(h.get("max_draws") or h["data_provenance"]["n_draws"]), int(h["n_train"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--now", required=True, help="model directory of the current model (model.pt + history.json)")
    ap.add_argument("--prev", required=True, help="model directory of the model trained on fewer draws")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-rows", type=int, default=None, help="smoke runs: a seeded subset of the val rows")
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    va = data.load_split(a.data, "val", KEYS)
    if a.max_rows:
        rows = data.seeded_rows(len(va["index"]), a.max_rows)
        va = {k: v[rows] for k, v in va.items()}
    enc = [encode(d) for d in data.draws(va, slice(None))]
    true_family = FAMILY_OF[np.array([e[0] for e in enc])]
    width, log_s = va["width"].astype(int), np.log(va["params"][:, data.PARAM_COLUMNS.index("s")])
    cells = active.cells_of_arrays(va)
    x = torch.as_tensor(to_input_bandtop(va["raw"]))
    print(f"[acquire] {len(x):,} val rows ({time.time() - t0:.0f}s)", flush=True)

    (n_max_now, n_now), (n_max_prev, n_prev) = _max_draws(a.now), _max_draws(a.prev)
    errs = {}
    for name, d in (("now", a.now), ("prev", a.prev)):
        errs[name] = row_errors(InverseModel.load(Path(d) / "model.pt"), x, true_family, width, log_s)
        print(f"[acquire] {name}: " + ", ".join(f"{k} {v.mean():.3f}" for k, v in errs[name].items())
              + f" ({time.time() - t0:.0f}s)", flush=True)

    fresh = va["index"] >= n_max_prev                   # rows the previous run never early-stopped on
    groups = FAMILY_PATTERN[active.family_of_cell(np.arange(active.N_CELLS))]
    doublings = np.log2(n_now / n_prev)
    n_all = np.bincount(cells, minlength=active.N_CELLS)
    n_fresh = np.bincount(cells[fresh], minlength=active.N_CELLS)
    err, slope, raw = {}, {}, {}
    for k in ("family", "width", "log_s"):
        hits = np.bincount(cells, weights=errs["now"][k], minlength=active.N_CELLS)
        err[k] = active.shrink_rates(hits, n_all, groups)
        r_now = np.bincount(cells[fresh], weights=errs["now"][k][fresh], minlength=active.N_CELLS) / np.maximum(n_fresh, 1)
        r_prev = np.bincount(cells[fresh], weights=errs["prev"][k][fresh], minlength=active.N_CELLS) / np.maximum(n_fresh, 1)
        se = np.sqrt((r_now * (1 - r_now) + r_prev * (1 - r_prev) + 0.25 / np.maximum(n_fresh, 1))
                     / np.maximum(n_fresh, 1)) / doublings
        slope[k] = active.shrink_means((r_prev - r_now) / doublings, np.where(n_fresh > 0, se, 1e3), groups)
        raw[k] = dict(rate_now=(hits / np.maximum(n_all, 1)).tolist(), slope=((r_prev - r_now) / doublings).tolist())
    gain = active.expected_gain(err, slope)

    tr = data.load_split(a.data, "train", KEYS[:1] + KEYS[2:])
    tr = {k: v[tr["index"] < n_max_now] for k, v in tr.items()}
    p = np.bincount(active.cells_of_arrays(tr), minlength=active.N_CELLS).astype(float)
    p /= p.sum()
    q, gamma = active.tilted_proposal(p, gain)
    accept = active.acceptance(p, q)

    names = [f"{family_name(c // active.N_WBINS)} w{active.WIDTH_EDGES[c % active.N_WBINS]}-"
             f"{active.WIDTH_EDGES[c % active.N_WBINS + 1] - 1}" for c in range(active.N_CELLS)]
    out = dict(now=str(a.now), prev=str(a.prev), n_train_now=n_now, n_train_prev=n_prev, n_val=len(x),
               n_val_fresh=int(fresh.sum()), log_s_tol=LOG_S_TOL, samples=SAMPLES, gamma=gamma,
               val_rates={m: {k: float(v.mean()) for k, v in e.items()} for m, e in errs.items()},
               accept=accept.tolist(),
               cells=[dict(cell=c, name=names[c], pattern=PATTERNS[groups[c]], n_val=int(n_all[c]),
                           n_fresh=int(n_fresh[c]), p=float(p[c]), q=float(q[c]), gain=float(gain[c]),
                           err={k: float(err[k][c]) for k in err}, slope={k: float(slope[k][c]) for k in slope},
                           raw={k: {kk: raw[k][kk][c] for kk in raw[k]} for k in raw})
                      for c in range(active.N_CELLS)])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))

    print(f"\n[acquire] gamma {gamma:.2f}; by pattern: prior share -> proposal share")
    for i, pat in enumerate(PATTERNS):
        m = groups == i
        print(f"  {pat:4s} {p[m].sum():.3f} -> {q[m].sum():.3f}   gain share {gain[m].sum() / max(gain.sum(), 1e-12):.3f}")
    print("\n[acquire] top 15 cells by gain (err family/width/log_s, slope family/width/log_s, p -> q)")
    for c in np.argsort(-gain)[:15]:
        print(f"  {names[c]:46s} err {err['family'][c]:.2f}/{err['width'][c]:.2f}/{err['log_s'][c]:.2f} "
              f"slope {slope['family'][c]:+.3f}/{slope['width'][c]:+.3f}/{slope['log_s'][c]:+.3f} "
              f"{p[c]:.4f} -> {q[c]:.4f}")
    print(f"\n[acquire] wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
