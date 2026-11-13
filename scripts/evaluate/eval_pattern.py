"""Lattice-type (site-type pattern, 7 classes) accuracy on the v1 val split, for lattice-type classifiers
(shazam.inverse.pattern) and full inverse models (the 156-state probabilities summed over each pattern). No simulation.

    TRANSMISSION_DATA=... PYTHONPATH=src python scripts/evaluate/eval_pattern.py --out results/inverse_design/pattern_v1/eval_pattern.json \\
        results/inverse_design/model_v2_depth_bandtop_600k results/inverse_design/pattern_v1

The first model is the reference for McNemar z (positive: the other model is right more often). Prints top-1, top-1 by
true pattern, top-2 and the confusion matrix of each model, and for full inverse models the 156-state top-1 too.

`--drop-pattern CHK` (repeatable; INV-NOCHK) scores only the val rows of the other patterns and gives the dropped
patterns no probability in every model (the rest renormalised), so a model trained on all seven is compared as if told
the dropped lattices do not exist.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import paths
from shazam.inverse import calibration, data
from shazam.inverse.encoding import STATES, encode
from shazam.inverse.model import InverseModel
from shazam.inverse.pattern import NAMES, N_PATTERNS, PatternModel, pattern_labels
from tbribbon.parent_prior import PATTERNS
from tbribbon.parent_data import to_input_bandtop

DEFAULT_DATA = paths.data_root() / "_generated" / "hexagonal_parent_v1"
_STATE_TO_PATTERN = np.eye(N_PATTERNS)[[PATTERNS.index(s.pattern) for s in STATES]]          # [156, 7]


@torch.no_grad()
def state_probs(model_dir, x):
    """[N, 156] state probabilities of a full inverse model (states it does not know get none), or None for a
    lattice-type classifier."""
    ck = torch.load(Path(model_dir) / "model.pt", map_location="cpu", weights_only=True)
    if "head.0.weight" in ck["state_dict"]:
        return None
    model = InverseModel.load(Path(model_dir) / "model.pt")
    out = []
    for i in range(0, len(x), 2000):
        logits = model.discrete(model.encode(x[i:i + 2000])).masked_fill(~model.mask_known, -torch.inf)
        out.append(F.softmax(logits, 1).numpy())
    return np.concatenate(out)


@torch.no_grad()
def pattern_probs(model_dir, x):
    """[N, 7] pattern probabilities of a lattice-type classifier or a full inverse model."""
    ps = state_probs(model_dir, x)
    return PatternModel.load(Path(model_dir) / "model.pt").predict_proba(x).numpy() if ps is None else ps @ _STATE_TO_PATTERN


def without(p, drop):
    """Probabilities with the `drop` columns zeroed and each row renormalised."""
    p = p.copy()
    p[:, drop] = 0.0
    return p / p.sum(1, keepdims=True)


def ensemble_probs(state_ps, members):
    """[N, 156] mean state probabilities of the full inverse models `members` (keys of `state_ps`, whose values are
    state_probs results; None marks a lattice-type-only model, which cannot join)."""
    if any(state_ps[m] is None for m in members):
        raise ValueError("ensembles take full inverse models only (156-state probabilities)")
    return np.mean([state_ps[m] for m in members], 0)


# Fixed regime bins (INV-DISTIL), so runs stay comparable: [lo, hi) except the last bin, which includes hi.
REGIMES = dict(width=(("4-8", 4, 9), ("9-13", 9, 14), ("14-18", 14, 19), ("19-22", 19, 23)),
               density=(("0.0025-0.005", 0.0025, 0.005), ("0.005-0.01", 0.005, 0.01), ("0.01-0.025", 0.01, 0.025),
                        ("0.025-0.05", 0.025, 0.0501)),
               cross=(("0.05-0.3", 0.05, 0.3), ("0.3-0.6", 0.3, 0.6), ("0.6-1", 0.6, 1.0001)))
TWO_TYPE = [NAMES.index(n) for n in ("CH", "CK", "HK")]


def regime_accuracy(hit, va, y):
    """Top-1 by width, edge and impurity density over all rows and over the two-type lattices (CH, CK, HK), and by the
    cross hopping between the two types on the two-type lattices: {subset: {regime: {bin: {n, top1}}}}."""
    from tbribbon.parent_prior import EDGES, FIELDS
    cross = va["params"][:, [FIELDS.index(f) for f in ("t_CH", "t_CK", "t_HK")]].sum(1)
    values = dict(width=va["width"], density=va["density"], cross=cross)

    def cell(m):
        return dict(n=int(m.sum()), top1=float(hit[m].mean()) if m.any() else float("nan"))

    out = {}
    for subset, sel in (("all", np.ones(len(y), bool)), ("two_type", np.isin(y, TWO_TYPE))):
        r = {k: {name: cell(sel & (values[k] >= lo) & (values[k] < hi)) for name, lo, hi in bins}
             for k, bins in REGIMES.items() if k != "cross" or subset == "two_type"}
        r["edge"] = {e: cell(sel & (va["edge"] == i)) for i, e in enumerate(EDGES)}
        out[subset] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="model directories; the first is the McNemar reference")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--ensemble-sizes", nargs="+", type=int, default=[],
                    help="also score the ensemble (mean state probabilities) of the first k models, for each k")
    ap.add_argument("--drop-pattern", action="append", default=[], choices=NAMES,
                    help="score without this site-type pattern (repeatable)")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    va = data.load_split(a.data, "val", ("index", "raw", "on", "lead", "params", "edge", "width", "density"))
    drop = [NAMES.index(n) for n in a.drop_pattern]
    keep = ~np.isin(pattern_labels(va["on"]), drop)
    va = {k: v[keep] for k, v in va.items()}
    x = torch.as_tensor(to_input_bandtop(va["raw"]), dtype=torch.float32)
    y = pattern_labels(va["on"])
    true_state = np.array([encode(d)[0] for d in data.draws(va, slice(None))])
    dropped_states = _STATE_TO_PATTERN[:, drop].any(1) if drop else np.zeros(len(STATES), bool)
    res, hits = {}, {}
    state_ps = {d: state_probs(d, x) for d in a.models}
    entries = list(a.models)
    for k in a.ensemble_sizes:                       # the ensemble of the first k models, scored like another model
        if not 2 <= k <= len(a.models):
            raise SystemExit(f"--ensemble-sizes {k}: needs 2 to {len(a.models)} models")
        state_ps[f"ensemble{k}"] = ensemble_probs(state_ps, a.models[:k])
        entries.append(f"ensemble{k}")
    for d in entries:
        ps = state_ps[d]
        p = PatternModel.load(Path(d) / "model.pt").predict_proba(x).numpy() if ps is None else ps @ _STATE_TO_PATTERN
        if drop:
            p = without(p, drop)
        pred = p.argmax(1)
        hits[d] = pred == y
        conf = np.zeros((N_PATTERNS, N_PATTERNS), int)
        np.add.at(conf, (y, pred), 1)
        top2 = (np.argsort(-p, 1)[:, :2] == y[:, None]).any(1)
        res[d] = dict(top1=float(hits[d].mean()), top2=float(top2.mean()),
                      ece=calibration.expected_calibration_error(p.max(1), hits[d]),
                      by_pattern={NAMES[k]: float(hits[d][y == k].mean()) for k in range(N_PATTERNS) if k not in drop},
                      confusion=conf.tolist(), regimes=regime_accuracy(hits[d], va, y))
        if ps is not None:
            res[d]["state_top1"] = float(((without(ps, dropped_states) if drop else ps).argmax(1) == true_state).mean())
    ref = a.models[0]
    for d in entries[1:]:
        only_new, only_ref = int((hits[d] & ~hits[ref]).sum()), int((hits[ref] & ~hits[d]).sum())
        res[d]["mcnemar_vs_ref"] = dict(only_new=only_new, only_ref=only_ref,
                                        z=float((only_new - only_ref) / np.sqrt(max(only_new + only_ref, 1))))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(n_val=int(len(y)), names=list(NAMES), drop_pattern=list(a.drop_pattern),
                                     reference=ref, models=res), indent=1))

    names = [Path(d).name for d in entries]
    w = max(24, *(len(n) + 2 for n in names))
    print(f"{len(y):,} val spectra" + (f" (without {', '.join(a.drop_pattern)})" if drop else "") + "\n" + f"{'':18s}" + "".join(f"{n:>{w}s}" for n in names))
    rows = [("top-1", lambda m: f"{m['top1']:.4f}"), ("top-2", lambda m: f"{m['top2']:.4f}"),
            ("ECE", lambda m: f"{m['ece']:.4f}"),
            ("state top-1", lambda m: f"{m['state_top1']:.4f}" if "state_top1" in m else "-")]
    rows += [(f"  {k}", (lambda k: lambda m: f"{m['by_pattern'][k]:.4f}")(k)) for k in NAMES if k not in a.drop_pattern]
    rows += [("McNemar z", lambda m: f"{m['mcnemar_vs_ref']['z']:+.1f}" if "mcnemar_vs_ref" in m else "ref")]
    for label, f in rows:
        print(f"{label:18s}" + "".join(f"{f(res[d]):>{w}s}" for d in entries))
    for subset in ("all", "two_type"):
        print(f"\ntop-1 by regime, {'all spectra' if subset == 'all' else 'two-type lattices (CH, CK, HK)'}")
        for regime, bins in res[entries[0]]["regimes"][subset].items():
            for b, c in bins.items():
                print(f"  {regime + ' ' + b:22s} n={c['n']:6d}" + "".join(
                    f"{res[d]['regimes'][subset][regime][b]['top1']:>{w}.4f}" for d in entries))
    for d, n in zip(entries, names):
        print(f"\n{n}: confusion (rows truth, columns prediction)\n{'':6s}" + "".join(f"{k:>7s}" for k in NAMES))
        for k, r in zip(NAMES, res[d]["confusion"]):
            print(f"{k:6s}" + "".join(f"{v:7d}" for v in r))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
