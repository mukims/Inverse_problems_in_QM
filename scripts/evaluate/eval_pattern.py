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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="model directories; the first is the McNemar reference")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=None)
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
    for d in a.models:
        ps = state_probs(d, x)
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
                      confusion=conf.tolist())
        if ps is not None:
            res[d]["state_top1"] = float(((without(ps, dropped_states) if drop else ps).argmax(1) == true_state).mean())
    ref = a.models[0]
    for d in a.models[1:]:
        only_new, only_ref = int((hits[d] & ~hits[ref]).sum()), int((hits[ref] & ~hits[d]).sum())
        res[d]["mcnemar_vs_ref"] = dict(only_new=only_new, only_ref=only_ref,
                                        z=float((only_new - only_ref) / np.sqrt(max(only_new + only_ref, 1))))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(n_val=int(len(y)), names=list(NAMES), drop_pattern=list(a.drop_pattern),
                                     reference=ref, models=res), indent=1))

    names = [Path(d).name for d in a.models]
    w = max(24, *(len(n) + 2 for n in names))
    print(f"{len(y):,} val spectra" + (f" (without {', '.join(a.drop_pattern)})" if drop else "") + "\n" + f"{'':18s}" + "".join(f"{n:>{w}s}" for n in names))
    rows = [("top-1", lambda m: f"{m['top1']:.4f}"), ("top-2", lambda m: f"{m['top2']:.4f}"),
            ("ECE", lambda m: f"{m['ece']:.4f}"),
            ("state top-1", lambda m: f"{m['state_top1']:.4f}" if "state_top1" in m else "-")]
    rows += [(f"  {k}", (lambda k: lambda m: f"{m['by_pattern'][k]:.4f}")(k)) for k in NAMES if k not in a.drop_pattern]
    rows += [("McNemar z", lambda m: f"{m['mcnemar_vs_ref']['z']:+.1f}" if "mcnemar_vs_ref" in m else "ref")]
    for label, f in rows:
        print(f"{label:18s}" + "".join(f"{f(res[d]):>{w}s}" for d in a.models))
    for d, n in zip(a.models, names):
        print(f"\n{n}: confusion (rows truth, columns prediction)\n{'':6s}" + "".join(f"{k:>7s}" for k in NAMES))
        for k, r in zip(NAMES, res[d]["confusion"]):
            print(f"{k:6s}" + "".join(f"{v:7d}" for v in r))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
