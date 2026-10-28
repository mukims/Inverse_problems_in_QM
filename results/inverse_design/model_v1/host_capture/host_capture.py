"""How often does model_v1 get the host, width and edge into its top 5 candidates, ignoring the concentration?

Same 2,000 test rows as the calibration gate (seeded choice, seed 0). Per draw: 1,000 samples, the usual candidates
(top 5), then for each candidate:
  key    right edge and site-type pattern (what today's top-5 recall counts)
  state  right full discrete state: pattern, edge, lead, delta/eps zero switches, t_AA on, cross bonds
  hostw  the true host values AND width lie inside the candidate's own 99% region, computed on the host slots + w only
         (log_c dropped): Mahalanobis(truth) <= 99th percentile of the members' distances in that subspace
  host   the same on the host slots alone (no width)
A draw is "found" at a level when any of its top 5 candidates passes every check up to that level.

    python host_capture.py [--n-test 2000] [--out host_capture.jsonl]
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
from shazam.inverse.calibration import _vectors
from shazam.inverse.candidates import candidates, full_vector, mahalanobis
from shazam.inverse.encoding import SLOT, encode
from shazam.inverse.model import InverseModel

DATA = Path("/run/media/shardul/storage/machine_learning/data/_generated/hexagonal_parent_v1")
MODEL = REPO / "results/inverse_design/model_v1/model.pt"
NEEDED = ("index", "x", "on", "lead", "params", "edge", "width", "density")
HOST = list(range(SLOT["log_c"]))                         # log_s .. t_HK
HOSTW = HOST + [SLOT["w"]]


def inside(truth, members, idx):
    """Is truth inside the members' own 99% region on the slots idx?"""
    v = members[:, idx]
    cov = np.cov(v, rowvar=False) + 1e-4 * np.eye(len(idx))
    mean = v.mean(axis=0)
    return bool(np.ravel(mahalanobis(truth[idx], mean, cov))[0] <= np.percentile(mahalanobis(v, mean, cov), 99))


def evaluate(model, x, draw, seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        states, z = model.sample(torch.as_tensor(x).reshape(1, -1), 1000)
    cands, p_un = candidates(states, z)
    vec = _vectors(states, z)
    truth = full_vector(draw)
    state, _, _ = encode(draw)
    key = (draw.edge, tuple(draw.params.pattern))
    rows = []
    for c in cands:
        m = vec[c.members]
        rows.append(dict(p=c.probability, s=int(c.state), key=(c.edge, tuple(c.pattern)) == key, state=c.state == state,
                         host=inside(truth, m, HOST), hostw=inside(truth, m, HOSTW),
                         w_med=float(c.draw.width), logc_med=float(np.log(c.draw.density))))
    return dict(cands=rows, p_unassigned=p_un, true_state=int(state), w=int(draw.width),
                logc=float(np.log(draw.density)), pattern="".join(t for t, o in zip("CHK", draw.params.pattern) if o),
                edge=draw.edge)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--out", default=str(Path(__file__).with_name("host_capture.jsonl")))
    a = ap.parse_args()
    torch.set_num_threads(16)
    model = InverseModel.load(MODEL)
    arrays = data.load_split(DATA, "test", NEEDED)
    rows = np.sort(np.random.default_rng(0).choice(len(arrays["index"]), size=2000, replace=False))[:a.n_test]
    draws, xs = data.draws(arrays, rows), arrays["x"][rows]
    out = Path(a.out)
    done = {json.loads(l)["index"] for l in out.read_text().splitlines()} if out.exists() else set()
    t0 = time.time()
    with out.open("a") as f:
        for k, (r, d, x) in enumerate(zip(rows, draws, xs)):
            idx = int(arrays["index"][r])
            if idx in done:
                continue
            rec = evaluate(model, x, d, seed=idx)
            f.write(json.dumps(dict(index=idx, **rec)) + "\n")
            f.flush()
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(rows)} draws, {(time.time() - t0) / (k + 1 - len(done) or 1):.2f} s per draw",
                      flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
