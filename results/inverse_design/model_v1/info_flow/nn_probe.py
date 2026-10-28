"""Where is information lost: the input, or the encoder? Nearest-neighbour probe on the validation split.

For each validation draw, find its nearest training draw (L2) in
  - raw:   the 416-channel model input x (what the encoder sees),
  - embed: the 128-number encoder output h = model.encode(x), each dim standardised over train,
and record whether the neighbour shares the full discrete state, the width, the edge + site types, and state + width.
Also "any of the 10 nearest" (a top-10 analogue). As a reference, the model itself on the same draws: classifier
argmax state, and the mode / membership of state and width over 200 posterior samples.

    python nn_probe.py [--n 5000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path("/run/media/shardul/storage/machine_learning/transmission_github/transmissions")
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch

from shazam.inverse import data
from shazam.inverse.encoding import decode, encode
from shazam.inverse.model import InverseModel

HERE = Path(__file__).parent
DATA = Path("/run/media/shardul/storage/machine_learning/data/_generated/hexagonal_parent_v1")
MODEL = REPO / "results/inverse_design/model_v1/model.pt"
NEEDED = ("index", "x", "on", "lead", "params", "edge", "width", "density")
S_COL = list(data.PARAM_COLUMNS).index("s")


def labels(a):
    state = np.array([encode(d)[0] for d in data.draws(a, slice(None))])
    key = a["edge"].astype(int) * 8 + (a["on"].astype(int) * [4, 2, 1]).sum(1)
    return dict(state=state, width=a["width"].astype(int), key=key)


def knn(q, ref, k=10, chunk=1024):
    """Indices of the k nearest rows of ref for each row of q (squared L2), via chunked matmul."""
    ref = torch.as_tensor(ref, dtype=torch.float32)
    rn = (ref * ref).sum(1)
    out = []
    for i in range(0, len(q), chunk):
        qc = torch.as_tensor(q[i:i + chunk], dtype=torch.float32)
        d = rn[None, :] - 2 * qc @ ref.T
        out.append(d.topk(k, largest=False).indices.numpy())
    return np.concatenate(out)


def embed(model, x, chunk=4096):
    with torch.no_grad():
        return np.concatenate([model.encode(torch.as_tensor(x[i:i + chunk])).numpy() for i in range(0, len(x), chunk)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--samples", type=int, default=200)
    a = ap.parse_args()
    torch.set_num_threads(16)
    t0 = time.time()
    model = InverseModel.load(MODEL).eval()
    tr = data.load_split(DATA, "train", NEEDED)
    va = data.load_split(DATA, "val", NEEDED)
    rows = np.sort(np.random.default_rng(0).choice(len(va["index"]), size=a.n, replace=False))
    va = {k: v[rows] for k, v in va.items()}
    Ltr, Lva = labels(tr), labels(va)
    print(f"loaded train {len(tr['index'])}, val {len(rows)}  ({time.time() - t0:.0f} s)", flush=True)

    htr, hva = embed(model, tr["x"]), embed(model, va["x"])
    mu, sd = htr.mean(0), htr.std(0) + 1e-6
    spaces = {"raw": (tr["x"], va["x"]), "embed": ((htr - mu) / sd, (hva - mu) / sd), "embed_unstd": (htr, hva)}
    nn = {}
    for name, (R, Q) in spaces.items():
        nn[name] = knn(Q, R)
        print(f"knn {name} done ({time.time() - t0:.0f} s)", flush=True)

    # the model itself on the same draws
    mdl = dict(argmax=[], state_mode=[], width_mode=[], state_in=[], width_in=[])
    with torch.no_grad():
        logits = model.discrete(torch.as_tensor(hva)).masked_fill(~model.mask_known, float("-inf"))
    mdl["argmax"] = logits.argmax(1).numpy()
    for i in range(len(rows)):
        with torch.random.fork_rng():
            torch.manual_seed(int(va["index"][i]))
            st, z = model.sample(torch.as_tensor(va["x"][i:i + 1]), a.samples)
        st = st.numpy()
        w = np.array([decode(int(s), zz).width for s, zz in zip(st, z.numpy())])
        mdl["state_mode"].append(np.bincount(st).argmax())
        mdl["width_mode"].append(np.bincount(w).argmax())
        mdl["state_in"].append(Lva["state"][i] in st)
        mdl["width_in"].append(Lva["width"][i] in w)
        if i % 1000 == 999:
            print(f"model samples {i + 1} ({time.time() - t0:.0f} s)", flush=True)

    out = HERE / "nn_probe.jsonl"
    with out.open("w") as f:
        for i in range(len(rows)):
            rec = dict(index=int(va["index"][i]), s=float(va["params"][i, S_COL]),
                       pattern="".join(t for t, o in zip("CHK", va["on"][i]) if o), edge=int(va["edge"][i]),
                       width=int(Lva["width"][i]), density=float(va["density"][i]))
            for name in spaces:
                j = nn[name][i]
                for lab in ("state", "width", "key"):
                    rec[f"{name}_{lab}_1"] = bool(Ltr[lab][j[0]] == Lva[lab][i])
                    rec[f"{name}_{lab}_10"] = bool((Ltr[lab][j] == Lva[lab][i]).any())
                both = (Ltr["state"][j] == Lva["state"][i]) & (Ltr["width"][j] == Lva["width"][i])
                rec[f"{name}_both_1"], rec[f"{name}_both_10"] = bool(both[0]), bool(both.any())
            rec.update(model_state_argmax=bool(mdl["argmax"][i] == Lva["state"][i]),
                       model_state_mode=bool(mdl["state_mode"][i] == Lva["state"][i]),
                       model_width_mode=bool(mdl["width_mode"][i] == Lva["width"][i]),
                       model_state_in=bool(mdl["state_in"][i]), model_width_in=bool(mdl["width_in"][i]))
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
