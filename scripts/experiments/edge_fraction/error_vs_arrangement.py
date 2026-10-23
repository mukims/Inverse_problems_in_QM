#!/usr/bin/env python
"""Do model errors depend on where the impurities sit?

For one ribbon, one density at a time (impurity count fixed), the rank correlation between two kinds of error and
every arrangement descriptor of src/shazam/arrangement.py:

  * lookup: the ensemble lookup's concentration estimate minus the true density, on test seeds 850-999 (never used to
    build the catalogue), spectra matched to the right device only;
  * edge fraction: out-of-fold residual (predicted minus true) of the edge-fraction regression, all seeds.

Spearman correlation with a bootstrap 95% interval; "significant" uses a Bonferroni-corrected p-value over every
comparison in the run.

    PYTHONPATH=src python scripts/experiments/edge_fraction/error_vs_arrangement.py --model hbn/armchair/N10
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import paths
from atlaslib.energy import on_axis
from atlaslib.lookup import Catalogue, lookup
from shazam.arrangement import across_axis, descriptors
from tbribbon.configuration import device_atom_coordinates

_spec = importlib.util.spec_from_file_location("run_edge_fraction", Path(__file__).resolve().parent / "run_edge_fraction.py")
ef = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ef)

TEST_MIN = 850          # lookup_v4 was built from seeds 0-699 and calibrated on 700-849


def rank_correlation(err, desc, n_boot=1000, seed=0, n_tests=1, alpha=0.05):
    """Spearman rho between an error and a descriptor, its bootstrap interval and Bonferroni-corrected significance."""
    err, desc = np.asarray(err, dtype=float), np.asarray(desc, dtype=float)
    if np.ptp(desc) == 0 or np.ptp(err) == 0:
        return {"rho": float("nan"), "ci": [float("nan"), float("nan")], "p": float("nan"), "significant": False}
    rho, p = spearmanr(err, desc)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(err), len(err))
        if np.ptp(desc[i]) > 0 and np.ptp(err[i]) > 0:
            boot.append(spearmanr(err[i], desc[i])[0])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"rho": float(rho), "ci": [float(lo), float(hi)], "p": float(p), "significant": bool(p < alpha / n_tests)}


def lookup_errors(cat, model, folder, key, density):
    """(seeds, estimate - true density, matched-the-right-device) for the test seeds of one cloud."""
    e_ev, _ = on_axis(cat.spec, model, np.load(folder / "energies_t.npy"))
    T = np.load(folder / f"cloud_{key}.npy")
    seeds = np.load(folder / f"cloud_{key}_seeds.npy")
    sel = np.flatnonzero(seeds >= TEST_MIN)
    est, right = [], []
    for row in T[sel]:
        r = lookup(cat, e_ev, row)
        est.append(r.concentration)
        right.append(r.device == model.model_id)
    return seeds[sel], np.array(est) - density, np.array(right)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model", default="hbn/armchair/N10")
    ap.add_argument("--catalogue", default=str(paths.results_dir("material_atlas", "lookup_v4")))
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    t0 = time.time()
    folder = paths.require(paths.model_dir(a.model))
    meta = json.loads((folder / "meta.json").read_text())
    device = device_atom_coordinates(a.model, n_cells=meta.get("pristine_n_cells"))
    across = across_axis(a.model)
    cat = Catalogue.load(a.catalogue)
    model = next(m for m in cat.models if m.model_id == a.model)
    clouds = {k: v for k, v in sorted(meta["clouds"].items()) if "source" not in v}
    log = lambda s: print(f"[err-arr] {s} ({time.time() - t0:.0f}s)", flush=True)

    rows = {}
    for key, info in clouds.items():
        seeds = np.load(folder / f"cloud_{key}_seeds.npy")
        with np.load(folder / f"cloud_{key}_config.npz") as z:
            if not np.array_equal(z["seeds"], seeds):
                raise ValueError(f"{key}: configuration rows are not in the cloud's seed order")
            desc = descriptors(z["coords"], z["sublattice"], device, across)
        X = ef.features(np.load(folder / f"cloud_{key}.npy"), np.load(folder / "pristine.npy"))
        resid = ef.out_of_fold(X, desc["edge_frac"], seeds, ef.fit_xgb) - desc["edge_frac"]
        t_seeds, lerr, right = lookup_errors(cat, model, folder, key, info["density"])
        at = np.searchsorted(seeds, t_seeds)
        rows[key] = dict(info=info, desc=desc, resid=resid, lerr=lerr[right], lidx=at[right], n_wrong=int((~right).sum()))
        log(f"{key}: lookup error {1e4 * lerr[right].mean():+.1f} +- {1e4 * lerr[right].std():.1f} (x1e-4) on "
            f"{right.sum()} test spectra ({(~right).sum()} matched to another device)")

    names = list(next(iter(rows.values()))["desc"])
    n_tests = 2 * len(names) * len(rows)
    res = {"model": a.model, "catalogue": a.catalogue, "n_tests": n_tests, "densities": {}}
    for key, r in rows.items():
        out = {"n_impurities": r["info"]["n_impurities"], "lookup_n": int(len(r["lerr"])), "lookup_wrong_device": r["n_wrong"],
               "lookup_error_mean": float(r["lerr"].mean()), "lookup_error_std": float(r["lerr"].std()),
               "lookup": {}, "edge_residual": {}}
        for n in names:
            out["lookup"][n] = rank_correlation(r["lerr"], r["desc"][n][r["lidx"]], a.n_boot, n_tests=n_tests)
            out["edge_residual"][n] = rank_correlation(r["resid"], r["desc"][n], a.n_boot, n_tests=n_tests)
        res["densities"][key] = out

    for kind, title in (("lookup", "lookup concentration error"), ("edge_residual", "edge-fraction residual")):
        print(f"\nSpearman rho: {title} vs descriptor   (* 95% interval excludes 0, ** Bonferroni over {n_tests} tests)")
        print(f"{'descriptor':22s}" + "".join(f"{k:>14s}" for k in res["densities"]))
        for n in names:
            cells = []
            for k, v in res["densities"].items():
                c = v[kind][n]
                mark = "**" if c["significant"] else ("*" if c["ci"][0] > 0 or c["ci"][1] < 0 else "")
                cells.append("           nan" if np.isnan(c["rho"]) else f"{c['rho']:+12.3f}{mark:2s}")
            print(f"{n:22s}" + "".join(cells))

    out = paths.results_dir("edge_fraction", create=True) / f"{a.model.replace('/', '_')}_error_vs_arrangement.json"
    out.write_text(json.dumps(res, indent=2))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
