"""The targets and guards of an active-learning round on one shared val split
(docs/superpowers/specs/2026-11-04-active-learning-design.md, step 4). No simulation.

    TRANSMISSION_DATA=... PYTHONPATH=src python scripts/evaluate/eval_active_round.py --out results/inverse_design/active/eval_controls.json \\
        results/inverse_design/model_v2_depth_bandtop results/inverse_design/model_v2_depth_bandtop_600k ...

Every model is scored on the same rows: all v1 val draws (index % 20 == 18), whatever --max-draws it was trained with.
Per model:

* targets: family top-1 by site-type pattern (37 families: pattern + switches, probability summed over edge and unit
  type, as in failure_modes.py), width within +-1 (modal width of SAMPLES flow samples), log s within LOG_S_TOL (median
  of the samples), and unconditional SBC of log s and w (KS statistic D and p-value of the rank of the truth);
* guards: the headline family + edge top-1 (probability summed over the unit type) and its top-label ECE.

McNemar z of every model against the first one for the headline and the per-row width hit.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import paths
from shazam.inverse import calibration, data
from shazam.inverse.encoding import SLOT, STATES, _bounds, encode
from shazam.inverse.model import InverseModel
from shazam.inverse.reward import FAMILY_OF, family_probs, pattern_name
from tbribbon.parent_data import to_input_bandtop

DEFAULT_DATA = paths.data_root() / "_generated" / "hexagonal_parent_v1"
KEYS = ("index", "raw", "on", "lead", "params", "edge", "width", "density")
SAMPLES, STEPS, BATCH, LOG_S_TOL = 20, 64, 100, 0.1
PATTERNS = ("C", "H", "K", "CH", "CK", "HK", "CHK")
_FAM37 = {}
FAMILY37_OF = np.array([_FAM37.setdefault((s.pattern, s.delta_zero, s.eps_zero, s.t_AA_on, s.cross), len(_FAM37))
                        for s in STATES])
_ONEHOT37 = np.eye(len(_FAM37))[FAMILY37_OF]


def _decode_slot(z, name):
    lo, hi = _bounds(name)
    return np.clip(lo + (hi - lo) / (1 + np.exp(-z)), lo, hi)


@torch.no_grad()
def evaluate(model, x, truth_state, width, log_s, seed=0):
    model.eval()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out = dict(fam74_top=[], fam74_p=[], fam37_top=[], w_mode=[], s_med=[], w_rank=[], s_rank=[])
    for i in range(0, len(x), BATCH):
        sl = slice(i, i + BATCH)
        h = model.encode(x[sl])
        prob = F.softmax(model.discrete(h).masked_fill(~model.mask_known, -torch.inf), 1)
        p74 = family_probs(prob.numpy())
        out["fam74_top"].append(p74.argmax(1))
        out["fam74_p"].append(p74.max(1))
        out["fam37_top"].append((prob.numpy() @ _ONEHOT37).argmax(1))
        st = torch.multinomial(prob, SAMPLES, replacement=True).reshape(-1)
        z = model.sample_batch(h.repeat_interleave(SAMPLES, 0), st, torch.randn(len(st), model.n_cont), steps=STEPS)
        z = z.numpy().reshape(len(h), SAMPLES, -1)
        w = np.rint(_decode_slot(z[..., SLOT["w"]], "w")).astype(int)
        s = _decode_slot(z[..., SLOT["log_s"]], "log_s")
        out["w_mode"].append(np.array([np.bincount(r).argmax() for r in w]))
        out["s_med"].append(np.median(s, 1))
        out["w_rank"].append([calibration.sbc_rank(t, r, rng) for t, r in zip(width[sl], w)])
        out["s_rank"].append([calibration.sbc_rank(t, r, rng) for t, r in zip(log_s[sl], s)])
    return {k: np.concatenate(v) for k, v in out.items()}


def _ks(ranks, rng):
    from scipy import stats
    u = (np.asarray(ranks, float) + rng.uniform(size=len(ranks))) / (SAMPLES + 1)
    r = stats.kstest(u, "uniform")
    return dict(D=float(r.statistic), p=float(r.pvalue))


def _mcnemar(a, b):
    """z of b against a (positive: b is right more often); a, b boolean per row."""
    only_b, only_a = int((b & ~a).sum()), int((a & ~b).sum())
    return dict(only_new=only_b, only_ref=only_a, z=float((only_b - only_a) / np.sqrt(max(only_a + only_b, 1))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="model directories (model.pt + history.json); the first is the reference")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None, help="smoke runs: a seeded subset of the val rows")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    va = data.load_split(a.data, "val", KEYS)
    if a.max_rows:
        rows = data.seeded_rows(len(va["index"]), a.max_rows)
        va = {k: v[rows] for k, v in va.items()}
    truth_state = np.array([encode(d)[0] for d in data.draws(va, slice(None))])
    width, log_s = va["width"].astype(int), np.log(va["params"][:, data.PARAM_COLUMNS.index("s")])
    x = torch.as_tensor(to_input_bandtop(va["raw"]))
    pat = np.array([PATTERNS.index(pattern_name(STATES[k].pattern)) for k in truth_state])
    true74, true37 = FAMILY_OF[truth_state], FAMILY37_OF[truth_state]
    print(f"[eval] {len(x):,} val rows ({time.time() - t0:.0f}s)", flush=True)

    res, hits = {}, {}
    for d in a.models:
        r = evaluate(InverseModel.load(Path(d) / "model.pt"), x, truth_state, width, log_s)
        h = json.loads((Path(d) / "history.json").read_text())
        rng = np.random.default_rng(1)
        head = r["fam74_top"] == true74
        w1 = np.abs(r["w_mode"] - width) <= 1
        s_ok = np.abs(r["s_med"] - log_s) <= LOG_S_TOL
        fam = r["fam37_top"] == true37
        hits[d] = dict(head=head, w1=w1)
        res[d] = dict(
            n_train=h["n_train"], best_epoch=h["best_epoch"], val_loss=min(h["val_loss"]), seed=h["cfg"]["seed"],
            headline_top1=float(head.mean()), headline_ece=calibration.expected_calibration_error(r["fam74_p"], head),
            family_top1=float(fam.mean()), width_within1=float(w1.mean()), log_s_ok=float(s_ok.mean()),
            sbc_log_s=_ks(r["s_rank"], rng), sbc_w=_ks(r["w_rank"], rng),
            by_pattern={p: dict(n=int((pat == i).sum()), family_top1=float(fam[pat == i].mean()),
                                width_within1=float(w1[pat == i].mean()), log_s_ok=float(s_ok[pat == i].mean()))
                        for i, p in enumerate(PATTERNS)})
        print(f"[eval] {Path(d).name}: headline {res[d]['headline_top1']:.4f}, family {res[d]['family_top1']:.4f}, "
              f"width±1 {res[d]['width_within1']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    ref = a.models[0]
    for d in a.models[1:]:
        res[d]["mcnemar_vs_ref"] = {k: _mcnemar(hits[ref][k], hits[d][k]) for k in ("head", "w1")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(n_val=len(x), samples=SAMPLES, log_s_tol=LOG_S_TOL, reference=ref, models=res),
                                indent=1))

    names = [Path(d).name.replace("model_v2_depth_bandtop", "base") for d in a.models]
    print("\n" + f"{'':22s}" + "".join(f"{n:>22s}" for n in names))
    rows = [("n_train", lambda m: f"{m['n_train']:,}"), ("val loss", lambda m: f"{m['val_loss']:.4f}"),
            ("headline top-1", lambda m: f"{m['headline_top1']:.4f}"), ("headline ECE", lambda m: f"{m['headline_ece']:.4f}"),
            ("family top-1", lambda m: f"{m['family_top1']:.4f}")]
    rows += [(f"  family {p}", (lambda p: lambda m: f"{m['by_pattern'][p]['family_top1']:.4f}")(p)) for p in PATTERNS]
    rows += [("width ±1", lambda m: f"{m['width_within1']:.4f}")]
    rows += [(f"  width ±1 {p}", (lambda p: lambda m: f"{m['by_pattern'][p]['width_within1']:.4f}")(p)) for p in PATTERNS]
    rows += [("log s within 10%", lambda m: f"{m['log_s_ok']:.4f}"),
             ("SBC log s D", lambda m: f"{m['sbc_log_s']['D']:.4f}"), ("SBC w D", lambda m: f"{m['sbc_w']['D']:.4f}"),
             ("McNemar head z", lambda m: f"{m['mcnemar_vs_ref']['head']['z']:+.1f}" if "mcnemar_vs_ref" in m else "ref"),
             ("McNemar w±1 z", lambda m: f"{m['mcnemar_vs_ref']['w1']['z']:+.1f}" if "mcnemar_vs_ref" in m else "ref")]
    for label, f in rows:
        print(f"{label:22s}" + "".join(f"{f(res[d]):>22s}" for d in a.models))
    print(f"\n[eval] wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
