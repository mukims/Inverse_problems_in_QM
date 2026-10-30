"""ERGO-1 part B: does the inverse model's error follow the impurity arrangement of the one disorder seed it saw?

Under the ergodic hypothesis a single spectrum is its ensemble average plus fluctuations that energy averaging removes,
so at a fixed impurity count the arrangement of that seed should not matter. For every validation draw (index <
300,000) this rebuilds the exact impurity sites of the generator (`impurity_shifts` with seed = draw index, the call of
`parent_data.simulate_raw`) and measures how far the seed's arrangement is from typical:
* scatterer classes: sites grouped by host on-site energy (an impurity shifts every site by the same V, so classes
  with different host energies scatter differently). chi2 = sum_k (n_k - n p_k)^2 / (n p_k) over the classes, with
  p_k the classes' share of the sites; 0 classes of difference for single-class hosts;
* edge share: z_edge = (n_edge - n p_edge) / sqrt(n p_edge (1 - p_edge)) for the sites within one bond length of an
  edge.
Against the per-draw results of per_draw_model.py (posterior median of log c minus the truth, its 90% interval,
modal width): errors by impurity count, single- against multi-class hosts, and Spearman correlations of |error| with
chi2 and |z_edge| within impurity-count bins (so the count, which sets both, is held roughly fixed).

    TRANSMISSION_DATA=... PYTHONPATH=src python results/inverse_design/ergodic/ergodic_parent.py model_v2_depth_bandtop

Writes parent_arrangement.npz (per draw) and parent_summary.{json,txt} next to this file.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import paths
from shazam.inverse import data
from tbribbon.disorder import impurity_shifts
from tbribbon.parent import parent_ribbon
from tbribbon.parent_data import IMPURITY_V

OUT = Path(__file__).resolve().parent
DATA = paths.data_root() / "_generated" / "hexagonal_parent_v1"
MAX_DRAWS = 300_000
KEYS = ("params", "on", "lead", "edge", "width", "density", "n_imp", "n_cells")
N_BINS = ((1, 5), (6, 15), (16, 40), (41, 100), (101, 10 ** 6))
EDGE_DIST = 1.0


def arrangement(draw, n_imp, n_cells, seed):
    """(classes, chi2, z_edge) of the generator's impurity sites for this draw and seed."""
    rb = parent_ribbon(draw.params, draw.edge, draw.width)
    spc = rb.H0.shape[0]
    shifts = impurity_shifts(n_cells, spc, n_imp, seed, IMPURITY_V)
    site = np.flatnonzero(shifts.ravel()) % spc
    eps = np.round(np.real(np.diag(rb.H0)), 6)
    levels, cls = np.unique(eps, return_inverse=True)
    p = np.bincount(cls, minlength=len(levels)) / spc
    n_k = np.bincount(cls[site], minlength=len(levels))
    chi2 = float(((n_k - n_imp * p) ** 2 / (n_imp * p)).sum()) if len(levels) > 1 else 0.0
    across = int(np.argmax(np.ptp(rb.positions, axis=0)))
    y = rb.positions[:, across]
    edge = np.minimum(y - y.min(), y.max() - y) < EDGE_DIST
    pe = edge.mean()
    ne = edge[site].sum()
    z_edge = float((ne - n_imp * pe) / np.sqrt(n_imp * pe * (1 - pe))) if 0 < pe < 1 else 0.0
    return len(levels), chi2, z_edge


def main(argv=None):
    model_name = (argv or sys.argv[1:])[0]
    t0 = time.time()
    va = data.load_split(DATA, "val", KEYS)
    va = {k: v[va["index"] < MAX_DRAWS] for k, v in va.items()}
    cache = OUT / "parent_arrangement.npz"
    if cache.exists() and np.array_equal(np.load(cache)["index"], va["index"]):
        arr = dict(np.load(cache))
    else:
        rows = []
        for i, d in enumerate(data.draws(va, slice(None))):
            rows.append(arrangement(d, int(va["n_imp"][i]), int(va["n_cells"][i]), int(va["index"][i])))
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(va['index'])} arrangements ({time.time() - t0:.0f} s)", flush=True)
        rows = np.array(rows)
        arr = dict(index=va["index"], classes=rows[:, 0].astype(int), chi2=rows[:, 1], z_edge=rows[:, 2])
        np.savez(cache, **arr)
    per_path = OUT / f"per_draw_{model_name}.npz"
    if not per_path.exists():
        print(f"wrote {cache.name}; {per_path.name} not there yet, run per_draw_model.py first")
        return
    per = np.load(per_path)
    if not np.array_equal(per["index"], va["index"]):
        raise ValueError("per-draw results and the validation split are not the same draws")
    n = va["n_imp"]
    lc = np.log(va["density"])
    err = per["logc_med"] - lc
    cover = (per["logc_q05"] <= lc) & (lc <= per["logc_q95"])
    multi = arr["classes"] > 1
    summary, lines = {}, []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say(f"{model_name}: {len(n)} validation draws; multi-class hosts {multi.mean():.3f}")
    say(f"{'n_imp':>10s} {'class':>6s} {'draws':>6s} | {'med err':>8s} {'MAD err':>8s} {'cover90':>8s} "
        f"{'rank':>6s} {'width':>6s} {'state':>6s} | {'rho(|e|,chi2)':>14s} {'rho(|e|,|z_edge|)':>18s}")
    for lo, hi in N_BINS:
        for label, sel_c in (("single", ~multi), ("multi", multi)):
            sel = (n >= lo) & (n <= hi) & sel_c
            if sel.sum() < 30:
                continue
            ae = np.abs(err[sel])
            rc = spearmanr(ae, arr["chi2"][sel])[0] if label == "multi" else np.nan
            re = spearmanr(ae, np.abs(arr["z_edge"][sel]))[0]
            row = dict(draws=int(sel.sum()), median_err=float(np.median(err[sel])), mad_err=float(np.median(ae)),
                       cover90=float(cover[sel].mean()), mean_rank=float(per["logc_rank"][sel].mean()),
                       width_mode=float(per["width_mode"][sel].mean()), state_top=float(per["state_top"][sel].mean()),
                       rho_abs_err_chi2=float(rc), rho_abs_err_edge=float(re))
            summary[f"n {lo}-{hi} {label}"] = row
            say(f"{f'{lo}-{hi}':>10s} {label:>6s} {row['draws']:6d} | {row['median_err']:+8.3f} {row['mad_err']:8.3f} "
                f"{row['cover90']:8.3f} {row['mean_rank']:6.3f} {row['width_mode']:6.3f} {row['state_top']:6.3f} | "
                f"{rc:14.3f} {re:18.3f}")
    signed = {}
    for lo, hi in N_BINS:
        sel = (n >= lo) & (n <= hi)
        if sel.sum() >= 30:
            signed[f"n {lo}-{hi}"] = float(spearmanr(err[sel], arr["z_edge"][sel])[0])
    summary["rho_signed_err_vs_z_edge"] = signed
    say("signed log c error vs z_edge (Spearman), by n_imp: " + ", ".join(f"{k} {v:+.3f}" for k, v in signed.items()))
    (OUT / "parent_summary.json").write_text(json.dumps(summary, indent=1))
    (OUT / "parent_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote parent_summary.{{json,txt}} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
