"""Step-1 baseline of model_v1: is the true host found, judged by the recreation check (the misfit noise floor)?

For a seeded random sample of the 2,000 calibration test draws: re-sample model_v1's top 5 candidates (same seed per draw
as host_capture), then simulate 50 disordered ribbons (VERIFY_SEEDS) of
  - the TRUE host (the ceiling: a real member fails ~6% of the time by construction), and
  - each candidate in order, at the test draw's TRUE concentration (concentration is step 2's job, so it is profiled out
    by fixing it at the value where the misfit minimum lies), stopping at the first candidate with the right edge and
    site types that recreates the target.
A candidate is "found" when recreation_check passes: the target's misfit to the ensemble median is within the 95th
percentile of the ensemble's own leave-one-out misfits. The clean-spectrum distance of each checked candidate is stored
too, to calibrate the cheap proxy later. Resumable: one JSON line per draw.

    python recreation_baseline.py [--n 150] [--jobs 22]
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
from shazam.inverse.candidates import candidates
from shazam.inverse.encoding import encode
from shazam.inverse.model import InverseModel
from shazam.inverse.verify import VERIFY_SEEDS, log_likelihood, recreation_check, simulate_inputs
from tbribbon.parent import device_cells_for
from tbribbon.parent_data import clean_channels, n_impurities

HERE = Path(__file__).parent
DATA = Path("/run/media/shardul/storage/machine_learning/data/_generated/hexagonal_parent_v1")
MODEL = REPO / "results/inverse_design/model_v1/model.pt"
NEEDED = ("index", "x", "on", "lead", "params", "edge", "width", "density", "clean")


def check(draw, target_x, jobs):
    """Recreation check of `draw` (density as given) against target_x on the 50 VERIFY_SEEDS."""
    n_cells = device_cells_for(draw.params, draw.edge)
    sims = simulate_inputs(draw, n_impurities(draw, n_cells), VERIFY_SEEDS, n_jobs=jobs)
    r = recreation_check(target_x, sims)
    return dict(misfit=r.misfit, threshold=r.threshold, p=r.p_value, recreated=r.recreated,
                log_L=log_likelihood(target_x, sims))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--jobs", type=int, default=22)
    ap.add_argument("--out", default=str(HERE / "recreation_baseline.jsonl"))
    a = ap.parse_args()
    torch.set_num_threads(8)
    model = InverseModel.load(MODEL)
    arrays = data.load_split(DATA, "test", NEEDED)
    rows2000 = np.sort(np.random.default_rng(0).choice(len(arrays["index"]), size=2000, replace=False))
    rows = np.random.default_rng(7).choice(rows2000, size=a.n, replace=False)          # random order, so a partial run is a fair sample
    out = Path(a.out)
    done = {json.loads(l)["index"] for l in out.read_text().splitlines()} if out.exists() else set()
    t0, n_new = time.time(), 0
    with out.open("a") as f:
        for k, r in enumerate(rows):
            idx = int(arrays["index"][r])
            if idx in done:
                continue
            d = data.draws(arrays, r)[0]
            x = arrays["x"][r]
            with torch.random.fork_rng():
                torch.manual_seed(idx)
                states, z = model.sample(torch.as_tensor(x).reshape(1, -1), 1000)
            cands, p_un = candidates(states, z)
            state, _, _ = encode(d)
            key = (d.edge, tuple(d.params.pattern))
            truth = check(d, x, a.jobs)
            checked, found_at = [], None
            for j, c in enumerate(cands):
                rec = dict(rank=j + 1, p=c.probability, key=(c.edge, tuple(c.pattern)) == key, state=c.state == state,
                           width=int(c.draw.width))
                if rec["key"]:
                    cd = c.draw._replace(density=d.density)
                    rec.update(check(cd, x, a.jobs))
                    ct = clean_channels(cd)
                    rec["d_clean"] = float(np.abs(ct - arrays["clean"][r]).sum() / max(arrays["clean"][r].sum(), 1))
                    if rec["recreated"]:
                        found_at = j + 1
                checked.append(rec)
                if found_at:
                    break
            n_new += 1
            f.write(json.dumps(dict(index=idx, pattern="".join(t for t, o in zip("CHK", d.params.pattern) if o),
                                    edge=d.edge, width=int(d.width), density=float(d.density), truth=truth,
                                    found_at=found_at, cands=checked, p_unassigned=p_un)) + "\n")
            f.flush()
            el = time.time() - t0
            print(f"{len(done) + n_new}/{len(rows)} draws  found_at={found_at}  truth_recreated={truth['recreated']}  "
                  f"{el / n_new / 60:.1f} min per draw, {el / 3600:.1f} h so far", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
