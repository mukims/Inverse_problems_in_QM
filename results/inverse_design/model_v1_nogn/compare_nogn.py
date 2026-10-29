"""INV-NOGN: does dropping GroupNorm from the encoder keep more host information? model_v1 vs model_v1_nogn.

Both models were trained identically (same 270,000 train / 15,000 validation draws with index < 300,000, seed 0,
early stopping on the validation loss); only the encoder's GroupNorm differs. On every validation draw: the classifier's
top state, and over 200 posterior samples (seeded by draw index, as nn_probe.py) the modal width and whether the true
state / width appear at all. Broken down by width and site types; the test split is never read.

    PYTHONPATH=src python compare_nogn.py
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch

import paths
from shazam.inverse import data
from shazam.inverse.encoding import decode, encode
from shazam.inverse.model import InverseModel

HERE = Path(__file__).parent
DATA = paths.data_root() / "_generated" / "hexagonal_parent_v1"
MODELS = {"model_v1": REPO / "results/inverse_design/model_v1/model.pt", "nogn": HERE / "model.pt"}
NEEDED = ("index", "x", "on", "lead", "params", "edge", "width", "density")
SAMPLES = 200


def evaluate(model, va, state, width):
    with torch.no_grad():
        h = model.encode(torch.as_tensor(va["x"]))
        logits = model.discrete(h).masked_fill(~model.mask_known, float("-inf"))
    out = dict(state_top=(logits.argmax(1).numpy() == state), width_mode=[], state_in=[], width_in=[])
    for i in range(len(state)):
        with torch.random.fork_rng():
            torch.manual_seed(int(va["index"][i]))
            st, z = model.sample(torch.as_tensor(va["x"][i:i + 1]), SAMPLES)
        w = np.array([decode(int(s), zz).width for s, zz in zip(st.numpy(), z.numpy())])
        out["width_mode"].append(np.bincount(w).argmax() == width[i])
        out["state_in"].append(state[i] in st.numpy())
        out["width_in"].append(width[i] in w)
    return {k: np.asarray(v, bool) for k, v in out.items()}


def main():
    torch.set_num_threads(4)
    t0 = time.time()
    va = data.load_split(DATA, "val", NEEDED)
    va = {k: v[va["index"] < 300_000] for k, v in va.items()}
    state = np.array([encode(d)[0] for d in data.draws(va, slice(None))])
    width = va["width"].astype(int)
    pattern = np.array(["".join(t for t, o in zip("CHK", on) if o) for on in va["on"]])
    res = {}
    for name, path in MODELS.items():
        res[name] = evaluate(InverseModel.load(path), va, state, width)
        print(f"{name} evaluated ({time.time() - t0:.0f} s)", flush=True)

    keys = ("state_top", "width_mode", "state_in", "width_in")
    groups = {"all": np.ones(len(state), bool)}
    groups.update({f"w {lo}-{hi - 1}": (width >= lo) & (width < hi) for lo, hi in ((4, 8), (8, 13), (13, 18), (18, 23))})
    groups.update({p: pattern == p for p in ("C", "H", "K", "CH", "CK", "HK", "CHK")})
    summary = {}
    print(f"\n{'':10s} {'n':>6s} | " + " | ".join(f"{k:>21s}" for k in keys))
    print(f"{'':10s} {'':6s} | " + " | ".join(f"{'v1':>6s} {'nogn':>6s} {'diff':>7s}" for _ in keys))
    for g, sel in groups.items():
        row = {k: (float(res["model_v1"][k][sel].mean()), float(res["nogn"][k][sel].mean())) for k in keys}
        summary[g] = dict(n=int(sel.sum()), **row)
        print(f"{g:10s} {sel.sum():6d} | " + " | ".join(f"{a:6.3f} {b:6.3f} {b - a:+7.3f}" for a, b in row.values()))
    # paired test on the headline numbers: draws one model gets right and the other wrong
    for k in ("state_top", "width_mode"):
        a, b = res["model_v1"][k], res["nogn"][k]
        n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
        z = (n01 - n10) / max(np.sqrt(n01 + n10), 1)
        summary[f"mcnemar_{k}"] = dict(nogn_only=n01, v1_only=n10, z=float(z))
        print(f"McNemar {k}: right only with nogn {n01}, only with v1 {n10}, z = {z:+.1f}")
    hist = {n: json.loads((p.parent / "history.json").read_text()) for n, p in MODELS.items()}
    for n, h in hist.items():
        b = h["best_epoch"]
        summary[f"history_{n}"] = dict(best_epoch=b, val_ce=h["val_ce"][b], val_flow=h["val_flow"][b], epochs=len(h["val_loss"]))
        print(f"{n}: best epoch {b} of {len(h['val_loss'])}, val ce {h['val_ce'][b]:.3f}, flow {h['val_flow'][b]:.3f}")
    (HERE / "compare_nogn.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {HERE / 'compare_nogn.json'} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
