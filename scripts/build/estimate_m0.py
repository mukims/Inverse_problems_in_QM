"""Seed-to-seed misfit m0 of one host, the scale of phase 2's simulator reward (spec §6, plan Task 8).

    TRANSMISSION_DATA=... PYTHONPATH=src nice -n 10 python scripts/build/estimate_m0.py --hosts 300 --jobs 22

Simulates --hosts seeded train draws with 10 disorder seeds each (range(3e9, 3e9 + 10): disjoint from the VERIFY, FINAL
and OPT seeds and from phase 2's candidate seeds). A host's m0 is the median leave-one-out member misfit of the
recreation check (member against the median of the others) after smoothing over k channels. Reports the noise ratio
m0(k) / m0(1) for k = 1, 3, 5, 9, fits sim_reward.M0Model at k = SMOOTH_CHANNELS and writes
results/inverse_design/model_v4_simreward/m0.json.
"""
import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

import paths
from shazam.arrangement import misfit
from shazam.inverse import data, sim_reward as sr
from shazam.inverse.runio import atomic_write_json, code_version

DEFAULT_DATA = paths.data_root() / "_generated" / "hexagonal_parent_v1"
DEFAULT_OUT = paths.results_dir("inverse_design", "model_v4_simreward") / "m0.json"
SEEDS = tuple(range(3 * 10**9, 3 * 10**9 + 10))
KS = (1, 3, 5, 9)
NEEDED = ("index", "on", "lead", "params", "edge", "width", "density")


def host_m0(ensemble, k):
    """Median over members of misfit(member, median of the other members), after smoothing over k channels."""
    ens = sr.smooth(np.asarray(ensemble, dtype=np.float64), k)
    return float(np.median([misfit(np.median(np.delete(ens, j, axis=0), axis=0), ens[j]) for j in range(len(ens))]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--hosts", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=22)
    a = ap.parse_args(argv)
    t0 = time.time()
    tr = data.load_split(a.data, "train", NEEDED)
    rows = data.seeded_rows(len(tr["index"]), a.hosts, seed=1)
    draws = data.draws(tr, rows)
    jobs = [(d, s) for d in draws for s in SEEDS]
    with mp.get_context("fork").Pool(a.jobs) as pool:
        xs = pool.map(sr._one, jobs, chunksize=1)
    print(f"[m0] {len(jobs)} simulations ({time.time() - t0:.0f} s)", flush=True)
    per_host = []
    for i, d in enumerate(draws):
        ens = [x for x in xs[i * len(SEEDS):(i + 1) * len(SEEDS)] if x is not None]
        if len(ens) >= 5:
            per_host.append(dict(index=int(tr["index"][rows[i]]), width=d.width, density=d.density,
                                 m0={str(k): host_m0(ens, k) for k in KS}))
    m1 = np.array([h["m0"]["1"] for h in per_host])
    ratio = {str(k): float(np.median(np.array([h["m0"][str(k)] for h in per_host]) / m1)) for k in KS}
    k = sr.SMOOTH_CHANNELS
    model = sr.M0Model.fit(np.array([h["width"] for h in per_host], float), np.array([h["density"] for h in per_host]),
                           np.array([h["m0"][str(k)] for h in per_host]))
    out = dict(model=model.to_json(), smooth_channels=k, noise_ratio_median=ratio, n_hosts=len(per_host),
               seeds=[SEEDS[0], SEEDS[-1]], hosts=per_host, code_version=code_version(),
               data_provenance=data.data_provenance(a.data), seconds=time.time() - t0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(Path(a.out), out)
    print(f"[m0] noise ratio m0(k)/m0(1): {ratio}; model {model.coef}; wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
