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
from shazam.inverse.runio import code_version
from shazam.inverse.verify import (FINAL_SEEDS, OPT_SEEDS, VERIFY_SEEDS, log_likelihood, recreation_check,
                                   simulate_inputs)
from tbribbon.parent import device_cells_for
from tbribbon.parent_data import E_WINDOW, V4, bandtop_k, n_impurities, resample_to_top
from tbribbon.parent_prior import FIELDS, box_edge_warnings

NO_UPDATE_WARNING = ("updated probabilities are renormalised over the returned candidates: "
                     "no fitted update (kappa, log_L0) was supplied")
TOP_ONLY_WARNING = ("updated probabilities are not computed: only the top candidate was checked in the simulator "
                    "(check='top')")
# The settings an update (kappa, log_L0) depends on: they change how the checks behind its log-likelihoods were made.
UPDATE_SETTINGS = ("no_refine", "refine_budget", "n_samples", "seeds", "n_cells")
_EV_FIELDS = FIELDS[1:]                                    # energies and hoppings: in units of s, so times s gives eV


def target_input(energies_ev, T):
    """The v4 network input (416, float32) of a target spectrum T sampled at `energies_ev` (eV, increasing from 0).

    ValueError when the lengths differ, any value is not finite, the energies do not cover the full 0 to 8.3 eV window,
    or the spectrum is zero everywhere (after the v4 rounding and despiking)."""
    e, T = np.asarray(energies_ev, dtype=np.float64), np.asarray(T, dtype=np.float64)
    if e.ndim != 1 or T.shape != e.shape:
        raise ValueError(f"energies and T must be 1-D arrays of the same length, got {e.shape} and {T.shape}")
    if not (np.isfinite(e).all() and np.isfinite(T).all()):
        raise ValueError("target contains non-finite values")
    if np.any(np.diff(e) <= 0):
        raise ValueError("energies must be strictly increasing")
    if e.size and e[0] < -1e-9 < e[-1]:                        # crop to E >= 0, starting exactly at 0 (interpolated)
        keep = e > 1e-9
        e, T = np.concatenate([[0.0], e[keep]]), np.concatenate([[np.interp(0.0, e, T)], T[keep]])
    if e.size < 2 or abs(e[0]) > 1e-9 or e[-1] < E_WINDOW - V4.step_t / 2:
        raise ValueError(f"full 0-{E_WINDOW} eV window required, got {e[0] if e.size else 'no'} to "
                         f"{e[-1] if e.size else 'no'} eV")
    x = V4.to_input(T, e)[0]
    if not x.any():
        raise ValueError("target has no transmission")
    return x.astype(np.float32)


def model_input(target_x, kind="log"):
    """The network input a model was trained on (its cfg "input": "log", "linear" or "bandtop") for the v4 input
    `target_x` (416,), which stays the space the simulator checks compare in.

    Exact: the v4 input is log1p of the cleaned, 3-decimal T over log1p(cap), so expm1 and the same rounding give back
    parent_data.to_input_linear of the spectrum, and the band-top input follows from it as in to_input_bandtop."""
    if kind == "log":
        return np.asarray(target_x, dtype=np.float32)
    if kind not in ("linear", "bandtop"):
        raise ValueError(f"unknown model input {kind!r}: expected 'log', 'linear' or 'bandtop'")
    lin = np.round(np.expm1(np.asarray(target_x, dtype=np.float64) * np.log1p(V4.cap)), 3).astype(np.float32)[None]
    if kind == "linear":
        return lin[0]
    k = bandtop_k(lin)
    return np.concatenate([resample_to_top(lin, k)[0], (k / lin.shape[1]).astype(np.float32)])


def background_mass(p_realistic, n_samples):
    """p0 of the updated probabilities (ruling R25): max(1 - sum(p_realistic), 1 / (n_samples + 1)).

    The floor (add-one smoothing) keeps a background term when every sample sits in a returned realistic candidate, so a
    poor fit still lowers the updated probabilities instead of being renormalised to 100%. Used by `design` and by the
    records the update (kappa, log_L0) is fitted on."""
    return max(1.0 - float(np.sum(p_realistic)), 1.0 / (n_samples + 1))


def update_mismatch(fitted, settings, model_sha256=None):
    """{name: (update.json value, this run's value)} for what differs between the fitted update and this run.

    fitted        the update.json dict (its "flags" are the fit-update run's settings, "model_sha256" the fitted model's hash)
    settings      this run's values of UPDATE_SETTINGS
    model_sha256  this run's model hash; None skips the model comparison
    An update without recorded flags (or without a model hash, when one is compared) is a mismatch of its own."""
    flags, out = fitted.get("flags"), {}
    if not flags:
        out["flags"] = ("none recorded", "needed")
    else:
        out.update({k: (flags.get(k), settings.get(k)) for k in UPDATE_SETTINGS if flags.get(k) != settings.get(k)})
    if model_sha256 is not None and fitted.get("model_sha256") != model_sha256:
        out["model_sha256"] = (fitted.get("model_sha256") or "none recorded", model_sha256)
    return out


def _parameters(draw):
    """The host as {"canonical": HexParams fields, "ev": s, energies and hoppings times s (eV)}."""
    p = draw.params
    canonical = {k: (v if isinstance(v, (str, bool)) else float(v)) for k, v in asdict(p).items()}
    ev = {"s": float(p.s)} | {k: float(getattr(p, k) * p.s) for k in _EV_FIELDS}
    return {"canonical": canonical, "ev": ev}


def _check_dict(check, seeds):
    """The check as JSON; `host` names the row key holding the host that was simulated (R29): the medoid's "parameters" for
    the verification check, "refined_parameters" for the final check of an early-stopped refinement."""
    return {"p_value": float(check.p_value), "recreated": bool(check.recreated), "misfit": float(check.misfit),
            "threshold": float(check.threshold), "seeds": seeds,
            "host": "refined_parameters" if seeds == "final" else "parameters"}


def _verdict(cand, target_x, n_imp, n_cells, n_jobs, refine_budget, seeds, rng):
    """(check dict, log_L, refined_parameters or None, simulations used) of one realistic candidate."""
    verify_seeds, final_seeds, opt_seeds = seeds
    sims = simulate_inputs(cand.draw, n_imp, verify_seeds, n_jobs=n_jobs, n_cells=n_cells)
    check = recreation_check(target_x, sims)
    used = len(verify_seeds)
    if check.recreated or refine_budget <= 0:                  # a budget of 0 means no refinement at all
        return _check_dict(check, "verify"), log_likelihood(target_x, sims), None, used
    res = refine(cand.draw, n_imp, target_x, check.threshold, budget=refine_budget, seeds=opt_seeds, n_jobs=n_jobs,
                 n_cells=n_cells, rng=rng, z_samples=cand.z_members)
    used += res.n_simulations
    best_res = res
    best_draw = res.draw
    best_n_imp = n_imp

    # Adjacent width fallback: if refinement on medoid width did not stop early,
    # and an adjacent width w' in {w-1, w+1} holds >= 20% of the candidate's members,
    # try a quick 2-eval check on w' before conceding failure.
    if not res.stopped_early and cand.member_widths is not None and len(opt_seeds) > 0:
        w_hat = cand.draw.width
        for w_alt in (w_hat - 1, w_hat + 1):
            if w_alt < 4:
                continue
            alt_sel = cand.member_widths == w_alt
            if alt_sel.mean() >= 0.20:
                alt_draw = cand.draw._replace(width=w_alt)
                alt_n_cells = n_cells if n_cells is not None else device_cells_for(alt_draw.params, alt_draw.edge)
                alt_n_imp = n_impurities(alt_draw, alt_n_cells)
                alt_budget = 2 * len(opt_seeds)
                alt_z = cand.z_members[alt_sel] if cand.z_members is not None else None
                res_alt = refine(alt_draw, alt_n_imp, target_x, check.threshold, budget=alt_budget, seeds=opt_seeds,
                                 n_jobs=n_jobs, n_cells=n_cells, rng=rng, z_samples=alt_z)
                used += res_alt.n_simulations
                if res_alt.stopped_early or res_alt.misfit < best_res.misfit:
                    best_res = res_alt
                    best_draw = res_alt.draw
                    best_n_imp = alt_n_imp
                if best_res.stopped_early:
                    break
    refined = _parameters(best_draw) | {"optimisation_misfit": float(best_res.misfit), "stopped_early": bool(best_res.stopped_early),
                                       "box_edge_warning": [str(n) for n in box_edge_warnings(best_draw)]}
    if not best_res.stopped_early:                                  # R5: the budget ran out; report the original verify check
        return _check_dict(check, "verify"), log_likelihood(target_x, sims), refined, used
    final = simulate_inputs(best_draw, best_n_imp, final_seeds, n_jobs=n_jobs, n_cells=n_cells)
    used += len(final_seeds)
    return _check_dict(recreation_check(target_x, final), "final"), log_likelihood(target_x, final), refined, used


def design(target_x, model, n_samples=1000, update=None, refine_budget=200, n_jobs=22, seed=0, n_cells=None,
           seeds_override=None, check="all", provenance=None, drop_patterns=()):
    """The report (a JSON-serialisable dict) for the v4 input `target_x` (416,): see the module docstring.

    update      {"kappa", "log_L0"} fitted on validation draws, or None: kappa = 1 and no background mass, so the
                updated probabilities are renormalised over the returned candidates and the report says so.
    n_cells     device length of every simulation (None = the host's own, as in the data); seeds_override: use only the
                first k seeds of each of VERIFY_SEEDS, FINAL_SEEDS and OPT_SEEDS. Both are for the tests only.
    refine_budget  simulations per refinement; 0 = never refine (a failed check stays failed).
    check       "all" (default): every realistic candidate is checked in the simulator; "top": only the rank-1 candidate
                (a cheaper run when only the top candidate's verdict is scored). Unchecked candidates have check null and no
                updated probability; with "top" no candidate has one, and the report says so.
    provenance  what the caller knows about its inputs, merged into the report's `provenance`: "model" (path),
                "model_sha256", "update_file", "update_flags" (the fit-update run's flags) and anything else. The report adds
                the code version, the seeds used (first seed and count of each contiguous tuple), n_samples, seed, check,
                refine_budget and n_cells.
    drop_patterns  optional tuple of pattern names (e.g. ("CHK",)) to mask out at inference time.
    """
    if isinstance(model, (list, tuple)):
        from shazam.inverse.model import EnsembleInverseModel
        model = EnsembleInverseModel(model, drop_patterns=drop_patterns)
    if check not in ("all", "top"):
        raise ValueError(f"check must be 'all' or 'top', got {check!r}")
    t0 = time.time()
    target_x = np.asarray(target_x, dtype=np.float32)
    if target_x.shape != (416,) or not np.isfinite(target_x).all():
        raise ValueError(f"target_x must be 416 finite values (a v4 input), got shape {target_x.shape}")
    seeds = tuple(s if seeds_override is None else s[:seeds_override] for s in (VERIFY_SEEDS, FINAL_SEEDS, OPT_SEEDS))
    fitted = update is not None
    kappa, log_L0 = (float(update["kappa"]), float(update["log_L0"])) if fitted else (1.0, -np.inf)
    if fitted and not (np.isfinite(kappa) and kappa > 0 and np.isfinite(log_L0)):
        raise ValueError(f"update needs a finite kappa > 0 and a finite log_L0, got kappa={kappa}, log_L0={log_L0}")

    x_model = model_input(target_x, getattr(model, "cfg", {}).get("input", "log"))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        try:
            states, z = model.sample(x_model[None, :], n_samples, drop_patterns=drop_patterns)
        except TypeError:
            states, z = model.sample(x_model[None, :], n_samples)
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
               "density_90": [float(v) for v in cand.intervals["c"]],
               "intervals_90": {k: [float(lo), float(hi)] for k, (lo, hi) in cand.intervals.items()},
               "nearest_material": cand.nearest_material,
               "material_distance": float(cand.material_distance), "check": None, "log_likelihood": None,
               "refined_parameters": None}
        if cand.realistic and (check == "all" or rank == 1):
            rng = np.random.default_rng([seed, rank])
            row["check"], row["log_likelihood"], row["refined_parameters"], used = _verdict(
                cand, target_x, n_imp, n_cells, n_jobs, refine_budget, seeds, rng)
            n_simulations += used
            if row["check"]["seeds"] == "final":                   # the refined host is the verdict: show its box-edge warning too
                row["box_edge_warning"] = sorted(set(row["box_edge_warning"]) | set(row["refined_parameters"]["box_edge_warning"]))
        rows.append(row)

    real = [r for r in rows if r["check"] is not None]
    # R24: p0 is everything not returned as realistic: unassigned, unrealistic and clusters beyond the top 5. It is taken over
    # the realistic rows whether or not they were all checked (check="top" leaves the others unchecked, not background).
    # R25: with a fitted update it is floored at 1 / (n_samples + 1); without one there is no background at all.
    p_realistic = np.array([r["model_probability"] for r in rows if r["realistic"]])
    p0 = background_mass(p_realistic, n_samples) if fitted else 1.0 - float(p_realistic.sum())
    if check == "all":
        p = np.array([r["model_probability"] for r in real])
        log_L = np.array([r["log_likelihood"] for r in real])
        if fitted:
            updated = updated_probabilities(p, log_L, p0, kappa, log_L0)
        else:                                                  # no fitted update: renormalise over the returned candidates
            updated = updated_probabilities(p, log_L, 0.0, kappa, 0.0)
        for r, u in zip(real, updated):
            r["updated_probability"] = float(u)

    report = {"p_unassigned": float(p_unassigned), "p_background": max(p0, 0.0), "update_fitted": bool(fitted),
              "update": {"kappa": kappa, "log_L0": log_L0} if fitted else None, "check": check,
              "updated_probabilities_computed": check == "all",
              "n_samples": int(n_samples), "candidates": rows, "n_simulations": int(n_simulations),
              "seconds": time.time() - t0,
              "provenance": {"code_version": code_version(), "model": None, "model_sha256": None, "update_file": None,
                             "update_flags": None, "n_samples": int(n_samples), "seed": int(seed), "check": check,
                             "refine_budget": int(refine_budget), "n_cells": n_cells,
                             "seeds": {name: {"first": int(t[0]) if t else None, "n": len(t)} for name, t in
                                       zip(("verify", "final", "optimisation"), seeds)}} | dict(provenance or {})}
    if check == "top":
        report["warning"] = TOP_ONLY_WARNING
    elif not fitted:
        report["warning"] = NO_UPDATE_WARNING
    return report
