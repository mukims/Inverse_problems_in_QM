#!/usr/bin/env python
"""
width_id.py -- step 1 of the AGNR inference pipeline.

Given an unknown ("test signature") transmission spectrum, identify the ribbon
WIDTH by comparing it against the pristine reference library, and produce the
evidence an agent needs to judge whether the answer is trustworthy:

  * band_gap_of(spec)          -> transport gap edge extracted from the signature
  * misfit_widths(spec, lib)   -> ranked candidate widths with misfit scores
  * validate_band_gap(...)     -> does the measured gap agree with the chosen
                                  width's pristine gap? (the agent's check)
  * identify_width(...)        -> the full step-1 result dict

Why a gap-aware misfit
----------------------
Impurities suppress transmission but do NOT move the band edges: the pristine
gap is a width fingerprint that survives disorder. So we score candidates on two
independent signals:

  1. gap agreement  -- |gap(signature) - gap(pristine_m)|; robust to concentration.
  2. shape misfit   -- L1 distance between the signature and the pristine spectrum
                       measured only where the pristine is conducting, and after
                       normalising out the overall transmission suppression that
                       impurities cause (a scale-free comparison).

A signature can only have transmission where its own ribbon conducts, so any
width whose pristine gap is LARGER than the signature's observed onset is
physically impossible -- those are hard-rejected before scoring. This is the
"which widths are logical" filter.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agnr_lib as A  # noqa: E402


# ----------------------------------------------------------------------
def load_library(path="~/agnr_infer/pristine_library.npz"):
    d = np.load(os.path.expanduser(path))
    return {k: d[k] for k in d.files}


def band_gap_of(spec, thresh=0.05):
    """Transport gap edge of a (possibly disordered) signature."""
    return A.band_gap(np.asarray(spec, float), thresh=thresh)


# ----------------------------------------------------------------------
def misfit_widths(spec, lib, mode="IR", gap_thresh=0.05, gap_tol=0.06,
                  reject_impossible=True):
    """Rank candidate widths for `spec`.

    Returns a list of dicts (best first) with keys:
        width, gap_ref, gap_diff, shape_misfit, score, possible, reason
    """
    spec = np.asarray(spec, float)
    widths = lib["widths"]
    T = lib["T_IR"] if mode == "IR" else lib["T_IL"]
    gaps = lib["gap_IR"] if mode == "IR" else lib["gap_IL"]
    w = lib["w"]

    g_obs, i_obs = band_gap_of(spec, thresh=gap_thresh)

    out = []
    for k, m in enumerate(widths):
        ref = T[k]
        g_ref = float(gaps[k])

        # --- physical possibility filter -------------------------------
        # the signature conducts from g_obs onward; a ribbon whose own gap
        # extends materially beyond that cannot produce this signature.
        possible = True
        reason = ""
        if reject_impossible and g_ref > g_obs + gap_tol:
            possible = False
            reason = f"pristine gap {g_ref:.2f} > observed onset {g_obs:.2f}"

        # --- gap agreement ---------------------------------------------
        gap_diff = abs(g_ref - g_obs)

        # --- scale-free shape misfit over the conducting region --------
        band = ref > 1e-9
        if band.sum() == 0:
            shape = np.inf
        else:
            a = spec[band]
            b = ref[band]
            # impurities suppress overall magnitude; compare shape, not scale
            sa = a.sum()
            shape = (np.mean(np.abs(a / sa - b / b.sum())) * len(a)
                     if sa > 0 else np.inf)

        score = gap_diff + 0.1 * shape        # gap dominates; shape breaks ties
        out.append(dict(width=int(m), gap_ref=g_ref, gap_diff=float(gap_diff),
                        shape_misfit=float(shape), score=float(score),
                        possible=bool(possible), reason=reason))

    out.sort(key=lambda r: (not r["possible"], r["score"]))
    return out, {"gap_observed": float(g_obs), "gap_index": int(i_obs)}


def validate_band_gap(spec, width, lib, mode="IR", gap_thresh=0.05, tol=0.03):
    """Agent check: is the band gap consistent with the claimed width?

    Returns dict with ok / measured / expected / delta / verdict text.
    """
    widths = list(lib["widths"])
    gaps = lib["gap_IR"] if mode == "IR" else lib["gap_IL"]
    if int(width) not in widths:
        return {"ok": False, "verdict": f"width {width} not in library"}
    g_ref = float(gaps[widths.index(int(width))])
    g_obs, _ = band_gap_of(spec, thresh=gap_thresh)
    delta = abs(g_obs - g_ref)
    ok = delta <= tol
    return {"ok": bool(ok), "measured": float(g_obs), "expected": g_ref,
            "delta": float(delta), "tol": tol,
            "verdict": ("gap consistent with width %d (|Δ|=%.3f ≤ %.3f)" % (width, delta, tol))
            if ok else
            ("gap INCONSISTENT with width %d (|Δ|=%.3f > %.3f)" % (width, delta, tol))}


def degeneracy_note(ranked, tol=0.02):
    """Widths that are effectively tied with the best -- the agent must know."""
    if not ranked:
        return []
    best = ranked[0]["score"]
    return [r["width"] for r in ranked[1:] if r["possible"] and r["score"] - best <= tol]


def identify_width(spec, lib, mode="IR", gap_thresh=0.05):
    """Full step-1 result."""
    ranked, obs = misfit_widths(spec, lib, mode=mode, gap_thresh=gap_thresh)
    viable = [r for r in ranked if r["possible"]]
    best = (viable or ranked)[0]
    ties = degeneracy_note(ranked)
    val = validate_band_gap(spec, best["width"], lib, mode=mode, gap_thresh=gap_thresh)
    return {
        "observed_gap": obs["gap_observed"],
        "best_width": best["width"],
        "best_score": best["score"],
        "gap_validation": val,
        "ties": ties,
        "n_rejected": len(ranked) - len(viable),
        "top5": ranked[:5],
        "ranked": ranked,
    }


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Identify AGNR width from a signature.")
    ap.add_argument("signature", help="path to .npy spectrum (300,)")
    ap.add_argument("--library", default="~/agnr_infer/pristine_library.npz")
    ap.add_argument("--mode", default="IR", choices=["IR", "IL"])
    args = ap.parse_args()

    spec = np.load(os.path.expanduser(args.signature))
    lib = load_library(args.library)
    res = identify_width(spec, lib, mode=args.mode)
    print(json.dumps({k: v for k, v in res.items() if k != "ranked"}, indent=2, default=str))
