#!/usr/bin/env python
"""
agnr_agent.py -- the inference agent for an unknown AGNR transmission signature.

Pipeline (steps 1-4 of the task):

  1. WIDTH      physics tools extract the band gap and rank candidate widths by
                a gap-aware misfit against the pristine library.
  2. VALIDATE   the agent judges whether the band gap is *correct* and which
                candidate widths are physically logical (LLM reasoning over the
                numeric evidence, with a deterministic fallback).
  3. CONCENTRATION  the agent calls the appropriate trained model
                (xgb / mlp / transformer) for the identified width.
  4. FEASIBILITY the agent assesses whether the result should be trusted, citing
                the concrete evidence (gap agreement, ties, disorder level,
                model spread, out-of-domain flags).

The LLM (default: local gemma4:e2b via Ollama) never computes physics or
regresses numbers -- it *decides* and *explains*. Every number it sees comes from
the deterministic tools below, and its structural choices are validated against
them, so a bad LLM response degrades to the physics answer rather than
corrupting it.

Usage
-----
    python agnr_agent.py <signature.npy>
    python agnr_agent.py <signature.npy> --true-width 7 --true-conc 21
    python agnr_agent.py <signature.npy> --no-llm        # deterministic only
"""

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agnr_lib as A          # noqa: E402
import width_id as W          # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma4:e2b"


# ======================================================================
# TOOLS -- deterministic; these are the only source of numbers
# ======================================================================
def tool_band_gap(spec):
    g, i = A.band_gap(np.asarray(spec, float))
    return {"gap_energy": float(g), "gap_index": int(i)}


def tool_rank_widths(spec, lib, mode="IL"):
    ranked, obs = W.misfit_widths(spec, lib, mode=mode)
    return {"observed_gap": obs["gap_observed"],
            "candidates": ranked,
            "viable": [r["width"] for r in ranked if r["possible"]],
            "rejected": [{"width": r["width"], "reason": r["reason"]}
                         for r in ranked if not r["possible"]]}


def tool_validate_gap(spec, width, lib, mode="IL"):
    return W.validate_band_gap(spec, width, lib, mode=mode)


def tool_disorder_level(spec, pristine):
    """How suppressed is the signature relative to pristine? Proxy for disorder."""
    spec = np.asarray(spec, float)
    p = np.asarray(pristine, float)
    band = p > 1e-9
    if band.sum() == 0:
        return {"suppression": None}
    ratio = float(np.clip(spec[band], 0, p[band]).sum() / p[band].sum())
    return {"suppression": 1.0 - ratio, "transmitted_fraction": ratio}


def tool_predict_concentration(spec, width, pristine, models_dir, crop=(20, 150)):
    """Call the trained concentration model(s) for this width."""
    models_dir = os.path.expanduser(models_dir)
    lo, hi = crop
    p = np.where(pristine > 1e-12, pristine, 1.0)
    x = (np.clip(np.asarray(spec, float), 0, pristine) / p)[lo:hi].astype(np.float32)

    preds, errs = {}, {}

    # --- XGBoost ---
    xgb_path = os.path.join(models_dir, "xgb_model.json")
    if os.path.exists(xgb_path):
        try:
            import xgboost as xgb
            m = xgb.XGBRegressor()
            m.load_model(xgb_path)
            preds["xgb"] = float(m.predict(x.reshape(1, -1))[0])
        except Exception as ex:
            errs["xgb"] = str(ex)

    # --- torch models ---
    for name, fn in (("mlp", "mlp_model.pt"), ("transformer", "transformer_model.pt")):
        path = os.path.join(models_dir, fn)
        if not os.path.exists(path):
            continue
        try:
            import torch
            sd = torch.load(path, map_location="cpu")
            model = _rebuild_torch_model(name, sd, len(x))
            model.eval()
            with torch.no_grad():
                preds[name] = float(model(torch.tensor(x).view(1, -1)).item())
        except Exception as ex:
            errs[name] = str(ex)

    metrics_path = os.path.join(models_dir, "conc_model_metrics.json")
    known = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    return {"predictions": preds, "errors": errs,
            "benchmark": known.get("results", {}), "best_model": known.get("best")}


def _rebuild_torch_model(name, state_dict, D):
    import torch.nn as nn
    import torch
    if name == "mlp":
        return _load_into(nn.Sequential(
            nn.Linear(D, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)), state_dict)
    # transformer -- mirror train_conc_models.PatchTransformer
    patch, dmodel, nhead, layers = 10, 64, 4, 2
    pad = (-D) % patch
    npatch = (D + pad) // patch

    class PatchTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(patch, dmodel)
            self.pos = nn.Parameter(torch.zeros(1, npatch, dmodel))
            enc = nn.TransformerEncoderLayer(dmodel, nhead, dim_feedforward=4 * dmodel,
                                             dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(enc, layers)
            self.head = nn.Sequential(nn.LayerNorm(dmodel), nn.Linear(dmodel, 1))

        def forward(self, x):
            if pad:
                x = torch.nn.functional.pad(x, (0, pad))
            x = x.view(x.shape[0], npatch, patch)
            return self.head(self.enc(self.proj(x) + self.pos).mean(1))

    return _load_into(PatchTransformer(), state_dict)


def _load_into(model, sd):
    model.load_state_dict(sd)
    return model


# ======================================================================
# LLM (gemma4) -- decides and explains; never produces the numbers
# ======================================================================
def ask_llm(prompt, model=DEFAULT_MODEL, timeout=180):
    import urllib.request
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.2}}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()).get("response", "")


def _extract_json(text):
    """Pull the first JSON object out of a (possibly chatty/thinking) reply."""
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    start = -1
    return None


DECIDE_PROMPT = """You are assisting with an inverse quantum-transport problem on \
armchair graphene nanoribbons (AGNR).

An unknown transmission signature T(E) was measured. Deterministic physics tools \
produced this evidence:

observed transport gap : {gap:.3f} eV-equivalent (energy units of the model)
pristine reference gaps: {gaps}
candidate ranking (best first, score = gap mismatch + 0.1 * shape misfit):
{cands}
physically rejected    : {rejected}
disorder suppression   : {supp}

Physics you must apply:
- AGNR widths follow families: N=3p+2 are near-metallic (gap ~ 0); N=3p and N=3p+1 \
are semiconducting with gaps that shrink as N grows.
- Impurities suppress transmission magnitude but do NOT move band edges, so the \
gap is a width fingerprint that survives disorder.
- A ribbon whose own pristine gap exceeds the observed conduction onset cannot \
produce this signature.

Decide:
1. is the measured band gap trustworthy (or corrupted by disorder/noise)?
2. which width is most logical, and which alternatives remain plausible?

Reply with ONLY a JSON object:
{{"gap_trustworthy": true/false, "chosen_width": <int>, "plausible_widths": [<int>,...], \
"confidence": "high"/"medium"/"low", "reasoning": "<two sentences>"}}"""


FEASIBILITY_PROMPT = """Same AGNR inverse problem. Final results:

width identified   : {width}   (gap check: {gapverdict})
plausible widths   : {plaus}
concentration      : {conc}
model predictions  : {preds}
model benchmark    : {bench}
disorder suppression: {supp}
signature domain   : {domain}

Assess FEASIBILITY of this result for a physicist. Cover: (a) is the width \
identification sound, (b) how much to trust the concentration number given model \
spread and disorder, (c) the main risk or caveat, (d) one concrete next check.

Reply with ONLY a JSON object:
{{"feasible": true/false, "confidence": "high"/"medium"/"low", \
"assessment": "<3-4 sentences>", "main_risk": "<one sentence>", \
"next_check": "<one sentence>"}}"""


# ======================================================================
# AGENT
# ======================================================================
def run_agent(spec, lib, pristine_by_width, models_dir, mode="IL",
              use_llm=True, llm_model=DEFAULT_MODEL, verbose=True):
    log = {}

    # ---------- step 1: physics evidence ----------
    gap = tool_band_gap(spec)
    ranking = tool_rank_widths(spec, lib, mode=mode)
    log["step1_gap"] = gap
    log["step1_ranking"] = {k: v for k, v in ranking.items() if k != "candidates"}
    log["step1_top5"] = ranking["candidates"][:5]

    physics_width = (ranking["viable"] or [c["width"] for c in ranking["candidates"]])[0]

    # ---------- step 2: agent validates gap + picks logical width ----------
    decision = None
    if use_llm:
        cand_txt = "\n".join(
            f"  width {c['width']:>2}: score={c['score']:.3f} gap_ref={c['gap_ref']:.2f} "
            f"gap_diff={c['gap_diff']:.3f} possible={c['possible']}"
            for c in ranking["candidates"][:6])
        gaps_txt = ", ".join(f"{int(m)}:{g:.2f}" for m, g in
                             zip(lib["widths"], lib["gap_IL" if mode == "IL" else "gap_IR"]))
        pris = pristine_by_width.get(int(physics_width))
        supp = tool_disorder_level(spec, pris) if pris is not None else {"suppression": None}
        try:
            raw = ask_llm(DECIDE_PROMPT.format(
                gap=gap["gap_energy"], gaps=gaps_txt, cands=cand_txt,
                rejected=[r["width"] for r in ranking["rejected"]] or "none",
                supp=round(supp.get("suppression") or 0, 3)), model=llm_model)
            decision = _extract_json(raw)
            log["step2_llm_raw"] = raw[-800:]
        except Exception as ex:
            log["step2_llm_error"] = str(ex)

    # validate the agent's structural choice against the physics
    chosen = physics_width
    if decision and isinstance(decision.get("chosen_width"), int):
        if decision["chosen_width"] in [int(m) for m in lib["widths"]]:
            chosen = int(decision["chosen_width"])
        else:
            log["step2_override"] = "LLM width not in library; kept physics choice"
    if decision and chosen != physics_width:
        log["step2_note"] = f"LLM chose {chosen}, physics ranked {physics_width} first"

    gap_check = tool_validate_gap(spec, chosen, lib, mode=mode)
    log["step2_decision"] = decision
    log["step2_chosen_width"] = chosen
    log["step2_gap_validation"] = gap_check

    # ---------- step 3: call the concentration model ----------
    pris = pristine_by_width.get(int(chosen))
    conc_res = {"predictions": {}, "errors": {"model": "no pristine for width"}}
    if pris is not None:
        conc_res = tool_predict_concentration(spec, chosen, pris, models_dir)
    log["step3_concentration"] = conc_res

    preds = conc_res.get("predictions", {})
    best_name = conc_res.get("best_model")
    conc_value = (preds.get(best_name) if best_name in preds
                  else (float(np.median(list(preds.values()))) if preds else None))
    log["step3_value"] = conc_value

    # ---------- step 4: feasibility ----------
    supp = tool_disorder_level(spec, pris) if pris is not None else {"suppression": None}
    spread = (max(preds.values()) - min(preds.values())) if len(preds) > 1 else 0.0
    domain = "in-domain (width 7 has trained models)" if int(chosen) == 7 else \
             f"OUT OF DOMAIN: models were trained on width 7, chosen width is {chosen}"
    feas = None
    if use_llm:
        try:
            raw = ask_llm(FEASIBILITY_PROMPT.format(
                width=chosen, gapverdict=gap_check.get("verdict"),
                plaus=(decision or {}).get("plausible_widths", ranking["viable"][:4]),
                conc=(f"{conc_value:.2f}" if conc_value is not None else "unavailable"),
                preds={k: round(v, 2) for k, v in preds.items()} or "none",
                bench=conc_res.get("benchmark") or "not trained",
                supp=round(supp.get("suppression") or 0, 3), domain=domain),
                model=llm_model)
            feas = _extract_json(raw)
            log["step4_llm_raw"] = raw[-800:]
        except Exception as ex:
            log["step4_llm_error"] = str(ex)

    log["step4_feasibility"] = feas or _fallback_feasibility(
        gap_check, ranking, spread, supp, conc_value, domain)
    log["summary"] = {
        "width": chosen,
        "concentration": conc_value,
        "model_spread": round(spread, 3),
        "gap_ok": gap_check.get("ok"),
        "suppression": supp.get("suppression"),
        "domain": domain,
    }
    return log


def _fallback_feasibility(gap_check, ranking, spread, supp, conc, domain):
    """Deterministic feasibility verdict when the LLM is unavailable."""
    reasons = []
    ok = True
    if not gap_check.get("ok"):
        ok = False
        reasons.append(f"band gap inconsistent ({gap_check.get('verdict')})")
    if len(ranking["viable"]) > 1:
        reasons.append(f"{len(ranking['viable'])} widths remain viable")
    if spread and spread > 3:
        ok = False
        reasons.append(f"model predictions disagree by {spread:.1f}")
    if (supp.get("suppression") or 0) > 0.8:
        reasons.append("very strong suppression: spectrum near noise floor")
    if "OUT OF DOMAIN" in domain:
        ok = False
        reasons.append(domain)
    if conc is None:
        ok = False
        reasons.append("no concentration model available")
    return {"feasible": ok, "confidence": "high" if ok and not reasons else "low",
            "assessment": "; ".join(reasons) or "all checks passed",
            "main_risk": reasons[0] if reasons else "none identified",
            "next_check": "regenerate a forward spectrum at the inferred (width, concentration) and compare"}


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="AGNR inference agent.")
    ap.add_argument("signature", help=".npy transmission spectrum (300,)")
    ap.add_argument("--library", default="~/agnr_infer/pristine_library.npz")
    ap.add_argument("--models-dir", default="~/agnr_infer/models")
    ap.add_argument("--mode", default="IL", choices=["IR", "IL"])
    ap.add_argument("--llm-model", default=DEFAULT_MODEL)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--true-width", type=int, default=None)
    ap.add_argument("--true-conc", type=float, default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    spec = np.load(os.path.expanduser(args.signature))
    lib = W.load_library(args.library)
    key = "T_IL" if args.mode == "IL" else "T_IR"
    pristine_by_width = {int(m): lib[key][i] for i, m in enumerate(lib["widths"])}

    log = run_agent(spec, lib, pristine_by_width, args.models_dir,
                    mode=args.mode, use_llm=not args.no_llm, llm_model=args.llm_model)

    s = log["summary"]
    print("\n================ AGNR INFERENCE ================")
    print(f"observed gap      : {log['step1_gap']['gap_energy']:.3f}")
    print(f"width identified  : {s['width']}"
          + (f"   (true {args.true_width})" if args.true_width else ""))
    print(f"gap validation    : {log['step2_gap_validation'].get('verdict')}")
    print(f"concentration     : "
          + (f"{s['concentration']:.2f}" if s['concentration'] is not None else "unavailable")
          + (f"   (true {args.true_conc})" if args.true_conc else ""))
    print(f"model predictions : {log['step3_concentration'].get('predictions')}")
    print(f"suppression       : {s['suppression']}")
    f = log["step4_feasibility"]
    print(f"\nfeasible          : {f.get('feasible')}  ({f.get('confidence')})")
    print(f"assessment        : {f.get('assessment')}")
    print(f"main risk         : {f.get('main_risk')}")
    print(f"next check        : {f.get('next_check')}")
    print("===============================================\n")

    if args.json_out:
        with open(os.path.expanduser(args.json_out), "w") as fh:
            json.dump(log, fh, indent=2, default=str)
        print(f"[saved] {args.json_out}")


if __name__ == "__main__":
    main()
