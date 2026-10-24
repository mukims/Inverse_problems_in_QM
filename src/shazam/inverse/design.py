"""From a target spectrum to a JSON report of candidate hosts (spec sections 7 and 8, output).

`design` samples the inverse model, groups the samples into candidates, and checks every realistic candidate in the
simulator: recreate on the verification seeds; when that fails, refine on the optimisation seeds and, if the refinement
reached the recreation threshold, re-check the refined host on the final-check seeds. The log-likelihood of the target
under the ensemble behind each reported check feeds the updated probabilities. Unrealistic candidates are reported
and never simulated.
"""
import time
from dataclasses import asdict

import numpy as np
import torch

from shazam.inverse import candidates as candidates_mod
from shazam.inverse.encoding import STATES
from shazam.inverse.posterior import updated_probabilities
from shazam.inverse.refine import refine
from shazam.inverse.verify import (FINAL_SEEDS, OPT_SEEDS, VERIFY_SEEDS, log_likelihood, recreation_check,
                                   simulate_inputs)
from tbribbon.parent import device_cells_for
from tbribbon.parent_data import E_WINDOW, V4, n_impurities

NO_UPDATE_WARNING = ("updated probabilities are renormalised over the returned candidates: "
                     "no fitted update (kappa, log_L0) was supplied")
_EV_FIELDS = ("eps_C", "eps_H_mean", "delta", "eps_K", "t_CC", "t_AB", "t_KK", "t_AA", "t_CH", "t_CK", "t_HK")


def target_input(energies_ev, T):
    """The v4 network input (416, float32) of a target spectrum T sampled at `energies_ev` (eV, increasing from 0).

    ValueError when the lengths differ, any value is not finite, the energies do not cover the full 0 to 8.3 eV window,
    or the spectrum is zero everywhere (after the v4 rounding and despiking)."""
    e, T = np.asarray(energies_ev, dtype=np.float64), np.asarray(T, dtype=np.float64)
    if e.ndim != 1 or T.shape != e.shape:
        raise ValueError(f"energies and T must be 1-D arrays of the same length, got {e.shape} and {T.shape}")
    if not (np.isfinite(e).all() and np.isfinite(T).all()):
        raise ValueError("target contains non-finite values")
    if e.size < 2 or abs(e[0]) > 1e-9 or e[-1] < E_WINDOW - V4.step_t / 2:
        raise ValueError(f"full 0-{E_WINDOW} eV window required, got {e[0] if e.size else 'no'} to "
                         f"{e[-1] if e.size else 'no'} eV")
    if np.any(np.diff(e) <= 0):
        raise ValueError("energies must be strictly increasing")
    x = V4.to_input(T, e)[0]
    if not x.any():
        raise ValueError("target has no transmission")
    return x.astype(np.float32)


def _parameters(draw):
    """The host as {"canonical": HexParams fields, "ev": s, energies and hoppings times s (eV)}."""
    p = draw.params
    canonical = {k: (v if isinstance(v, (str, bool)) else float(v)) for k, v in asdict(p).items()}
    ev = {"s": float(p.s)} | {k: float(getattr(p, k) * p.s) for k in _EV_FIELDS}
    return {"canonical": canonical, "ev": ev}


def _check_dict(check, seeds):
    return {"p_value": float(check.p_value), "recreated": bool(check.recreated), "misfit": float(check.misfit),
            "threshold": float(check.threshold), "seeds": seeds}


def _verdict(cand, target_x, n_imp, n_cells, n_jobs, refine_budget, seeds, rng):
    """(check dict, log_L, refined_parameters or None, simulations used) of one realistic candidate."""
    verify_seeds, final_seeds, opt_seeds = seeds
    sims = simulate_inputs(cand.draw, n_imp, verify_seeds, n_jobs=n_jobs, n_cells=n_cells)
    check = recreation_check(target_x, sims)
    used = len(verify_seeds)
    if check.recreated:
        return _check_dict(check, "verify"), log_likelihood(target_x, sims), None, used
    res = refine(cand.draw, n_imp, target_x, check.threshold, budget=refine_budget, seeds=opt_seeds, n_jobs=n_jobs,
                 n_cells=n_cells, rng=rng)
    used += res.n_simulations
    refined = _parameters(res.draw) | {"optimisation_misfit": float(res.misfit), "stopped_early": bool(res.stopped_early)}
    if not res.stopped_early:                                  # R5: the budget ran out; report the original verify check
        return _check_dict(check, "verify"), log_likelihood(target_x, sims), refined, used
    final = simulate_inputs(res.draw, n_imp, final_seeds, n_jobs=n_jobs, n_cells=n_cells)
    used += len(final_seeds)
    return _check_dict(recreation_check(target_x, final), "final"), log_likelihood(target_x, final), refined, used


def design(target_x, model, n_samples=1000, update=None, refine_budget=200, n_jobs=22, seed=0, n_cells=None,
           seeds_override=None):
    """The report (a JSON-serialisable dict) for the v4 input `target_x` (416,): see the module docstring.

    update      {"kappa", "log_L0"} fitted on validation draws, or None: kappa = 1 and no background mass, so the
                updated probabilities are renormalised over the returned candidates and the report says so.
    n_cells     device length of every simulation (None = the host's own, as in the data); seeds_override: use only the
                first k seeds of each of VERIFY_SEEDS, FINAL_SEEDS and OPT_SEEDS. Both are for the tests only.
    """
    t0 = time.time()
    target_x = np.asarray(target_x, dtype=np.float32)
    if target_x.shape != (416,) or not np.isfinite(target_x).all():
        raise ValueError(f"target_x must be 416 finite values (a v4 input), got shape {target_x.shape}")
    seeds = tuple(s if seeds_override is None else s[:seeds_override] for s in (VERIFY_SEEDS, FINAL_SEEDS, OPT_SEEDS))
    fitted = update is not None and np.isfinite(update["log_L0"])
    kappa, log_L0 = (float(update["kappa"]), float(update["log_L0"])) if update is not None else (1.0, -np.inf)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        states, z = model.sample(target_x[None, :], n_samples)
    cands, p_unassigned = candidates_mod.candidates(np.asarray(states), np.asarray(z))

    rows, n_simulations = [], 0
    for rank, cand in enumerate(cands, start=1):
        draw = cand.draw
        n_imp = n_impurities(draw, n_cells if n_cells is not None else device_cells_for(draw.params, draw.edge))
        row = {"rank": rank, "model_probability": float(cand.probability), "updated_probability": None,
               "realistic": bool(cand.realistic), "box_edge_warning": [str(n) for n in cand.box_edge],
               "discrete_state": {k: (list(v) if isinstance(v, tuple) else v) for k, v in STATES[cand.state]._asdict().items()},
               "parameters": _parameters(draw), "edge": cand.edge, "width": int(draw.width), "impurity_count": int(n_imp),
               "impurity_count_90": [int(v) for v in cand.n_imp_interval], "density": float(draw.density),
               "density_90": [float(v) for v in cand.intervals["c"]], "nearest_material": cand.nearest_material,
               "material_distance": float(cand.material_distance), "check": None, "log_likelihood": None,
               "refined_parameters": None}
        if cand.realistic:
            rng = np.random.default_rng([seed, rank])
            row["check"], row["log_likelihood"], row["refined_parameters"], used = _verdict(
                cand, target_x, n_imp, n_cells, n_jobs, refine_budget, seeds, rng)
            n_simulations += used
        rows.append(row)

    real = [r for r in rows if r["realistic"]]
    p = np.array([r["model_probability"] for r in real])
    # R24: p0 is everything not returned as realistic: unassigned, unrealistic and clusters beyond the top 5
    p0 = 1.0 - float(p.sum())
    log_L = np.array([r["log_likelihood"] for r in real])
    if np.isfinite(log_L0):
        updated = updated_probabilities(p, log_L, p0, kappa, log_L0)
    else:                                                      # no background: renormalise over the returned candidates
        updated = updated_probabilities(p, log_L, 0.0, kappa, 0.0)
    for r, u in zip(real, updated):
        r["updated_probability"] = float(u)

    report = {"p_unassigned": float(p_unassigned), "p_background": max(p0, 0.0), "update_fitted": bool(fitted),
              "update": {"kappa": kappa, "log_L0": log_L0} if fitted else None,
              "n_samples": int(n_samples), "candidates": rows, "n_simulations": int(n_simulations),
              "seconds": time.time() - t0}
    if not fitted:
        report["warning"] = NO_UPDATE_WARNING
    return report
