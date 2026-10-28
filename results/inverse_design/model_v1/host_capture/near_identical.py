"""Are the structures model_v1 gets "wrong" physically near-identical to the truth?

Targets, from host_capture.jsonl: every draw whose top 5 has the right edge and site types but not the full structure
("miss"), and 300 random draws whose host and width were found ("found", the baseline for what a right answer looks
like). Each draw is re-sampled with the same seed, so its candidates are the ones host_capture saw. Compared candidate:
the most probable one with the right edge and site types (miss), or the first that passed every check (found).

Per draw: the structural fields that differ, the values that decide how close a wrong switch is to the boundary, and
the distance between clean spectra (open channels on the 416-point grid, no impurities), d = sum|T_cand - T_true| /
sum T_true, for
  d_host   the candidate's host at the TRUE width (structure and host values only)
  d_full   the candidate as reported (its own width)
  d_ref    a random other test host with the same site types, edge and width (what "a different host" looks like)

    python near_identical.py [--jobs 22]
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path("/run/media/shardul/storage/machine_learning/transmission_github/transmissions")
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch

from shazam.inverse import data
from shazam.inverse.candidates import candidates
from shazam.inverse.encoding import STATES, encode
from shazam.inverse.model import InverseModel
from tbribbon.parent_data import clean_channels
from tbribbon.parent_prior import CROSS, WITHIN

HERE = Path(__file__).parent
DATA = Path("/run/media/shardul/storage/machine_learning/data/_generated/hexagonal_parent_v1")
MODEL = REPO / "results/inverse_design/model_v1/model.pt"
NEEDED = ("index", "x", "on", "lead", "params", "edge", "width", "density", "clean")
FIELDS = ("lead", "delta_zero", "eps_zero", "t_AA_on", "cross")


def ok(c, checks):
    return all(c[k] for k in checks)


def dist(a, b):
    return float(np.abs(np.asarray(a, float) - b).sum() / max(np.sum(b), 1))


def host_info(p):
    return dict(lead=p.lead, s=p.s, delta=p.delta, t_AA=p.t_AA, eps={t: getattr(p, n) for t, n in
                (("C", "eps_C"), ("H", "eps_H_mean"), ("K", "eps_K"))},
                within={t: getattr(p, n) for t, n in WITHIN.items()},
                cross={"".join(k): getattr(p, n) for k, n in CROSS.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=22)
    ap.add_argument("--limit", type=int, default=None, help="only this many targets (smoke test)")
    a = ap.parse_args()
    hc = {r["index"]: r for r in map(json.loads, (HERE / "host_capture.jsonl").read_text().splitlines())}
    miss = [i for i, r in hc.items() if any(c["key"] for c in r["cands"]) and not any(ok(c, ("key", "state"))
                                                                                     for c in r["cands"])]
    found = [i for i, r in hc.items() if any(ok(c, ("key", "state", "host", "hostw")) for c in r["cands"])]
    rng = np.random.default_rng(1)
    targets = [(i, "miss") for i in miss] + [(int(i), "found") for i in rng.choice(found, 300, replace=False)]
    if a.limit:
        targets = targets[:a.limit]

    torch.set_num_threads(16)
    model = InverseModel.load(MODEL)
    arrays = data.load_split(DATA, "test", NEEDED)
    row_of = {int(i): k for k, i in enumerate(arrays["index"])}
    keyed = {}
    for k in range(len(arrays["index"])):
        d = data.draws(arrays, k)[0]
        keyed.setdefault((d.params.pattern, d.edge, d.width), []).append(k)

    jobs, recs = [], []
    for idx, kind in targets:
        k = row_of[idx]
        d = data.draws(arrays, k)[0]
        with torch.random.fork_rng():
            torch.manual_seed(idx)
            states, z = model.sample(torch.as_tensor(arrays["x"][k]).reshape(1, -1), 1000)
        cands, _ = candidates(states, z)
        state, _, _ = encode(d)
        key = (d.edge, tuple(d.params.pattern))
        if kind == "miss":
            c = next(c for c in cands if (c.edge, tuple(c.pattern)) == key)
        else:
            flags = hc[idx]["cands"]
            c = cands[next(j for j, f in enumerate(flags) if ok(f, ("key", "state", "host", "hostw")))]
        t, s = STATES[state], STATES[c.state]
        others = [j for j in keyed[(d.params.pattern, d.edge, d.width)] if j != k]
        ref = int(rng.choice(others)) if others else None
        recs.append(dict(index=idx, kind=kind, pattern="".join(x for x, o in zip("CHK", d.params.pattern) if o),
                         width=d.width, cand_width=c.draw.width, p=c.probability,
                         wrong=[f for f in FIELDS if getattr(t, f) != getattr(s, f)],
                         true=host_info(d.params), cand=host_info(c.draw.params)))
        jobs.append((c.draw._replace(width=d.width), c.draw))
        recs[-1]["_k"], recs[-1]["_ref"] = k, ref

    with Pool(a.jobs) as pool:
        spectra = pool.starmap(clean_pair, jobs, chunksize=4)
    out = HERE / "near_identical.jsonl"
    with out.open("w") as f:
        for r, (host_t, full_t) in zip(recs, spectra):
            true_t = arrays["clean"][r.pop("_k")]
            ref = r.pop("_ref")
            r.update(d_host=dist(host_t, true_t), d_full=dist(full_t, true_t),
                     d_ref=dist(arrays["clean"][ref], true_t) if ref is not None else None)
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} records to {out}")


def clean_pair(host_at_true_width, as_reported):
    return clean_channels(host_at_true_width).tolist(), clean_channels(as_reported).tolist()


if __name__ == "__main__":
    main()
