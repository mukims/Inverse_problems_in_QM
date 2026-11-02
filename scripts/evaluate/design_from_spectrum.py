"""Design a ribbon host from a target transmission spectrum (spec sections 7 and 8, output).

    PYTHONPATH=src python scripts/evaluate/design_from_spectrum.py --target T.npy --energies E_eV.npy

--target is a .npy of the transmission T(E) and --energies a .npy of the same length with the energies in eV (from 0 and
increasing, covering the full 0 to 8.3 eV window). The inverse model proposes candidate hosts; every realistic one is
recreated in the simulator on fresh disorder seeds (refined and re-checked when that fails), and the JSON report lists,
per candidate, the model probability, the updated probability, the check, the parameters (canonical and in eV), the
edge, width, impurity count and density with their 90% intervals, and the nearest named material; plus p_unassigned.

--model       default: results/inverse_design/model_v1/model.pt
--update      JSON {"kappa", "log_L0"} fitted on validation draws; default update.json next to the model when present.
              Without it the updated probabilities are renormalised over the returned candidates and the report says so.
              An update fitted with other check settings (refine budget, samples, ...) or on another model.pt (sha256) is
              refused unless --allow-update-mismatch, as in eval_inverse.py end-to-end.
--threads     torch intra-op threads (default: torch's own)
--out         default: <target stem>_design.json next to the target (written atomically)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import paths
from shazam.inverse.design import design, target_input, update_mismatch
from shazam.inverse.model import InverseModel
from shazam.inverse.runio import atomic_write_json, code_version, sha256_of

DEFAULT_MODEL = paths.results_dir("inverse_design", "model_v1") / "model.pt"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", required=True, help=".npy of the transmission T(E)")
    ap.add_argument("--energies", required=True, help=".npy of the energies (eV) of --target")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--update", default=None, help='JSON {"kappa", "log_L0"}; default update.json next to the model if present')
    ap.add_argument("--out", default=None, help="default: <target stem>_design.json next to the target")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--refine-budget", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=22)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None, help="torch intra-op threads (default: torch's own)")
    ap.add_argument("--allow-update-mismatch", action="store_true",
                    help="use update.json even though it was fitted with other check settings or on another model")
    a = ap.parse_args(argv)
    code_version()                                                 # this run's code version: fixed now, when it starts
    if a.threads:
        torch.set_num_threads(a.threads)

    target = Path(a.target)
    x = target_input(np.load(a.energies), np.load(target))
    out = Path(a.out) if a.out else target.with_name(target.stem + "_design.json")
    out.parent.mkdir(parents=True, exist_ok=True)                  # a bad --out fails now, not after the simulations
    update_path = Path(a.update) if a.update else Path(a.model).with_name("update.json")
    update, raw, mismatch = None, {}, None
    sha = sha256_of(a.model)
    if update_path.exists():
        raw = json.loads(update_path.read_text())
        missing = [k for k in ("kappa", "log_L0") if k not in raw]
        if missing:
            raise SystemExit(f"{update_path} lacks {', '.join(missing)}: expected a JSON object with kappa and log_L0")
        update = {"kappa": float(raw["kappa"]), "log_L0": float(raw["log_L0"])}
        if not (np.isfinite(update["kappa"]) and update["kappa"] > 0 and np.isfinite(update["log_L0"])):
            raise SystemExit(f"{update_path}: kappa must be finite and > 0 and log_L0 finite, got {update}")
        settings = dict(no_refine=a.refine_budget == 0, refine_budget=a.refine_budget, n_samples=a.n_samples, seeds=None,
                        n_cells=None)
        mismatch = update_mismatch(raw, settings, sha)
        if mismatch and not a.allow_update_mismatch:
            raise SystemExit(f"{update_path} was fitted with other settings than this run ({mismatch}): kappa and log_L0 "
                             "depend on the model and on how the checks were made. Re-run eval_inverse.py fit-update with "
                             "these settings, or pass --allow-update-mismatch")
        if mismatch:
            print(f"[design] WARNING: update fitted with other settings ({mismatch}); allowed by --allow-update-mismatch",
                  flush=True)
    elif a.update:
        raise SystemExit(f"--update file not found: {update_path}")
    provenance = dict(model=str(a.model), model_sha256=sha, update_file=str(update_path) if update else None,
                      update_flags=raw.get("flags") if update else None, update_mismatch=mismatch)
    report = design(x, InverseModel.load(a.model), n_samples=a.n_samples, update=update,
                    refine_budget=a.refine_budget, n_jobs=a.n_jobs, seed=a.seed, provenance=provenance)
    atomic_write_json(out, report, indent=2)
    print(f"[design] {len(report['candidates'])} candidates, p_unassigned {report['p_unassigned']:.3f}, "
          f"{report['n_simulations']} simulations, {report['seconds']:.0f} s -> {out}", flush=True)
    return report


if __name__ == "__main__":
    main()
