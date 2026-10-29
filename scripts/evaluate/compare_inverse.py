"""Compare two trained inverse models on the validation split (draws with index < 300,000; the test split is never read).

Each model gets the spectrum transform it was trained on (its cfg "input": "log" is the stored x, "linear" is
parent_data.to_input_linear of the stored raw spectra). On every validation draw: the classifier's top state, and over
200 posterior samples (seeded by draw index) the modal width and whether the true state / width appear at all. Broken
down by width and site types, with a paired McNemar z for the top state and the modal width. Generalises
results/inverse_design/model_v1_nogn/compare_nogn.py (INV-NOGN), which fed both models the log input.

    PYTHONPATH=src python scripts/evaluate/compare_inverse.py results/inverse_design/model_v1 results/inverse_design/model_v1_linear

Writes compare_<A>.{txt,json} into B's folder.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import paths
from shazam.inverse import data
from shazam.inverse.encoding import decode, encode
from shazam.inverse.model import InverseModel
from tbribbon.parent_data import to_input_bandtop, to_input_linear

DATA = paths.data_root() / "_generated" / "hexagonal_parent_v1"
NEEDED = ("index", "x", "raw", "on", "lead", "params", "edge", "width", "density")
SAMPLES = 200
MAX_DRAWS = 300_000


def model_input(model, va):
    kind = model.cfg.get("input", "log")
    return {"log": lambda: va["x"], "linear": lambda: to_input_linear(va["raw"]),
            "bandtop": lambda: to_input_bandtop(va["raw"])}[kind]()


def evaluate(model, x, index, state, width):
    with torch.no_grad():
        h = model.encode(torch.as_tensor(x))
        logits = model.discrete(h).masked_fill(~model.mask_known, float("-inf"))
    out = dict(state_top=(logits.argmax(1).numpy() == state), width_mode=[], state_in=[], width_in=[])
    for i in range(len(state)):
        with torch.random.fork_rng():
            torch.manual_seed(int(index[i]))
            st, z = model.sample(torch.as_tensor(x[i:i + 1]), SAMPLES)
        w = np.array([decode(int(s), zz).width for s, zz in zip(st.numpy(), z.numpy())])
        out["width_mode"].append(np.bincount(w).argmax() == width[i])
        out["state_in"].append(state[i] in st.numpy())
        out["width_in"].append(width[i] in w)
    return {k: np.asarray(v, bool) for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a", type=Path, help="folder of the reference model (model.pt, history.json)")
    ap.add_argument("b", type=Path, help="folder of the model under test; the comparison is written here")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=None, help="only the first n validation draws (a smoke test)")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    va = data.load_split(DATA, "val", NEEDED)
    va = {k: v[va["index"] < MAX_DRAWS][:a.n] for k, v in va.items()}
    state = np.array([encode(d)[0] for d in data.draws(va, slice(None))])
    width = va["width"].astype(int)
    pattern = np.array(["".join(t for t, o in zip("CHK", on) if o) for on in va["on"]])
    names = {"A": a.a.name, "B": a.b.name}
    res, lines = {}, []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    for key, folder in (("A", a.a), ("B", a.b)):
        model = InverseModel.load(folder / "model.pt")
        res[key] = evaluate(model, model_input(model, va), va["index"], state, width)
        say(f"{names[key]} (input {model.cfg.get('input', 'log')}) evaluated ({time.time() - t0:.0f} s)")

    keys = ("state_top", "width_mode", "state_in", "width_in")
    groups = {"all": np.ones(len(state), bool)}
    groups.update({f"w {lo}-{hi - 1}": (width >= lo) & (width < hi) for lo, hi in ((4, 8), (8, 13), (13, 18), (18, 23))})
    groups.update({p: pattern == p for p in ("C", "H", "K", "CH", "CK", "HK", "CHK")})
    summary = dict(a=str(a.a), b=str(a.b))
    say(f"\nA = {names['A']}, B = {names['B']}")
    say(f"{'':10s} {'n':>6s} | " + " | ".join(f"{k:>21s}" for k in keys))
    say(f"{'':10s} {'':6s} | " + " | ".join(f"{'A':>6s} {'B':>6s} {'diff':>7s}" for _ in keys))
    for g, sel in groups.items():
        row = {k: (float(res["A"][k][sel].mean()), float(res["B"][k][sel].mean())) for k in keys}
        summary[g] = dict(n=int(sel.sum()), **row)
        say(f"{g:10s} {sel.sum():6d} | " + " | ".join(f"{x:6.3f} {y:6.3f} {y - x:+7.3f}" for x, y in row.values()))
    for k in ("state_top", "width_mode"):
        x, y = res["A"][k], res["B"][k]
        b_only, a_only = int((~x & y).sum()), int((x & ~y).sum())
        z = (b_only - a_only) / max(np.sqrt(b_only + a_only), 1)
        summary[f"mcnemar_{k}"] = dict(b_only=b_only, a_only=a_only, z=float(z))
        say(f"McNemar {k}: right only with B {b_only}, only with A {a_only}, z = {z:+.1f}")
    for key, folder in (("A", a.a), ("B", a.b)):
        h = json.loads((folder / "history.json").read_text())
        e = h["best_epoch"]
        summary[f"history_{key}"] = dict(best_epoch=e, val_loss=h["val_loss"][e], val_ce=h["val_ce"][e],
                                         val_flow=h["val_flow"][e], epochs=len(h["val_loss"]))
        say(f"{names[key]}: best epoch {e} of {len(h['val_loss'])}, val {h['val_loss'][e]:.3f} "
            f"(ce {h['val_ce'][e]:.3f}, flow {h['val_flow'][e]:.3f})")
    stem = a.b / f"compare_{names['A']}"
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    stem.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {stem}.{{txt,json}} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
